"""
TensorTerm to nn.Module compiler. Resolves op names from run scope and torch (no whitelist).

E2E tests for tensor-term ops (Linear, ReLU, transpose, view, sqrt, ArgMax, Concat, *, +, etc.)
live in tests/_feature_tests/test_e2e_single_ops.py.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

from relann.pydantic_classes import (
    ArithTerm,
    ContentEncode,
    TensorTerm,
    TensorOp,
    TransformDef,
    Var,
)
from relann.encode import EncodeTypeError, is_text_dtype, tensorize_column
from relann.smart_ops import (
    smart_matmul,
    smart_mul,
    smart_add,
    smart_sub,
    smart_div,
    smart_pow,
    smart_transpose,
    smart_view,
)


# ---------------------------------------------------------------------------
# Resolution: run scope -> torch.nn -> torch
# ---------------------------------------------------------------------------

# Sentinel for DSL-native unary ops (view, sqrt, transpose) so parser knows they "exist"
_DSL_NATIVE_UNARY = object()

# TensorTerm nodes that represent arithmetic (for shape/hyperparam conversion)
_ARITH_OP_STR = frozenset(("+", "-", "*", "/", "**"))

# Arithmetic ops whose sub-trees are evaluated as ctor hyperparams (not tensor operations)
_PURE_ARITH_OPS = frozenset(("+", "-", "*", "/", "@", "**"))


def tensor_term_to_arith_term(t: TensorTerm) -> ArithTerm:
    """
    Convert a TensorTerm (constant leaf or arithmetic subtree) to an ArithTerm.
    Used for shape/hyperparam evaluation in the compiler and in transform_def in the parser.
    """
    if t.op is None:
        v = getattr(t, "value", None)
        if isinstance(v, ArithTerm):
            return v
        return ArithTerm(value=v)
    if isinstance(t.op, TensorOp) and t.op.hyper_params is None and t.sons and str(t.op.op) in _ARITH_OP_STR:
        return ArithTerm(op=str(t.op.op), sons=[tensor_term_to_arith_term(s) for s in t.sons])
    raise ValueError(f"Cannot convert tensor term to arith: {t}")


def _arith_term_bool_var_to_primitive(a: ArithTerm) -> ArithTerm:
    """`True`/`False` parsed as CNAME vars (by Lark) become Python bool literals for ctor args."""
    if a.op is None and a.sons is None and isinstance(a.value, Var):
        if a.value.name == "True":
            return ArithTerm(value=True)
        if a.value.name == "False":
            return ArithTerm(value=False)
    return a


def resolve_op(name: str, globals_dict: Dict[str, Any]) -> Optional[Union[type, Callable, object]]:
    """
    Resolve a string name to a class or callable. Lookup order:
    1. DSL-native: view, sqrt, transpose (sentinel)
    2. globals_dict (run scope)
    3. RelNN built-ins (ArgMax, Concat, Tensor), defined in this module.
    4. torch.nn
    5. torch (functions)
    Returns None if not found.
    """
    name_lower = name.lower() if name else ""
    if name_lower in ("view", "sqrt", "transpose"):
        return _DSL_NATIVE_UNARY

    if name in globals_dict:
        obj = globals_dict[name]
        if isinstance(obj, type) and issubclass(obj, nn.Module):
            return obj
        if callable(obj) and not isinstance(obj, type):
            return obj
        if isinstance(obj, nn.Module):
            return obj

    # RelNN built-ins: ArgMax, Concat, Tensor (defined in this module)
    import sys
    _this = sys.modules[__name__]
    if name in ("ArgMax", "argmax"):
        return getattr(_this, "ArgMax")
    if name in ("Concat", "Tensor"):
        return getattr(_this, name)
    if name in ("HashBucketTextEncoder", "HashBucket"):
        from relann.encoders import HashBucketTextEncoder
        return HashBucketTextEncoder

    if hasattr(nn, name):
        obj = getattr(nn, name)
        if isinstance(obj, type) and issubclass(obj, nn.Module):
            return obj
        if callable(obj):
            return obj

    if hasattr(torch, name):
        obj = getattr(torch, name)
        if callable(obj):
            return obj

    return None


# ---------------------------------------------------------------------------
# Multi-arg input adapters (op-specific input normalization; keeps _MultiArgWrapper op-agnostic)
# ---------------------------------------------------------------------------

def _targets_to_class_indices(t: torch.Tensor) -> torch.Tensor:
    """Normalize targets to (N,) long for nn.CrossEntropyLoss. (N,1) -> squeeze; else ensure long."""
    if t.dim() == 2 and t.size(1) == 1:
        return t.squeeze(1).long()
    return t.long()


def _cross_entropy_adapter(predictions: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Adapter for CrossEntropyLoss: pass predictions through; normalize targets to class indices (N,) long."""
    return (predictions, _targets_to_class_indices(targets))


# Classification losses (CrossEntropyLoss, NLLLoss) expect class indices; adapters normalize (N,1) or (N,) to (N,) long.
MULTI_ARG_INPUT_ADAPTERS: Dict[str, Callable[..., Tuple[torch.Tensor, ...]]] = {
    "CrossEntropyLoss": _cross_entropy_adapter,
}


# ---------------------------------------------------------------------------
# Encode: content columns (RHS ``[...]``)
# ---------------------------------------------------------------------------

_AUTO_TENSORIZE_NOTICE_KEYS: set[tuple[int, str, str]] = set()


def replace_submodule(root: nn.Module, target: nn.Module, replacement: nn.Module) -> bool:
    """Replace ``target`` with ``replacement`` anywhere in the module tree.

    Handles ``ModuleList`` and ``ModuleDict`` parents correctly: ``named_children``
    yields names like ``"0"``, ``"1"`` for list slots, but ``setattr(list, "0", x)``
    does not update the slot — must use ``__setitem__``.
    """
    if isinstance(root, nn.ModuleList):
        for i, child in enumerate(root):
            if child is target:
                root[i] = replacement
                return True
            if replace_submodule(child, target, replacement):
                return True
        return False
    if isinstance(root, nn.ModuleDict):
        for key, child in root.items():
            if child is target:
                root[key] = replacement
                return True
            if replace_submodule(child, target, replacement):
                return True
        return False
    for name, child in list(root.named_children()):
        if child is target:
            setattr(root, name, replacement)
            return True
        if replace_submodule(child, target, replacement):
            return True
    return False


class _ColumnExtractModule(nn.Module):
    """
    Reads one column from the input ER's content DataFrame (injected via ``_source_er``).
    Numeric/bool/categorical -> tensor; text/object -> ``pd.Series`` for a wrapping encoder.
    """

    def __init__(self, column_name: str):
        super().__init__()
        self.column_name = column_name
        self._source_er: Any = None
        self._target_device: Optional[torch.device] = None
        self._target_dtype: Optional[torch.dtype] = None
        self._last_was_text: bool = False
        self._cached_result: Any = None
        self._cache_key: Optional[tuple] = None

    @staticmethod
    def _make_cache_key(df, data_version: int = 0) -> tuple:
        """
        Stable cache key that avoids false hits from Python id() reuse after GC.
        Uses object identity + row count + first index value so that a new DataFrame
        of different content but accidental same id() doesn't get a cache hit.
        ``data_version`` bumps when the relation source reloads (see ER ``data_version``).
        For minibatching, call invalidate() explicitly between batches.
        """
        first_idx = df.index[0] if len(df) > 0 else None
        return (id(df), len(df), first_idx, int(data_version))

    def clear_cache(self) -> None:
        """Reset the cache. Called by Transformation before each dummy-run and by
        future BatchSpec runners between mini-batches."""
        self._cached_result = None
        self._cache_key = None

    # Alias used by future BatchSpec code (same behaviour, more descriptive name).
    invalidate = clear_cache

    def forward(self, *inputs: Any) -> Any:
        er = self._source_er
        if er is None or getattr(er, "content", None) is None:
            raise RuntimeError(
                f"Column extract for {self.column_name!r}: no source relation with content "
                "(internal: Transformation must inject _source_er before forward)."
            )
        df = er.content
        if self.column_name not in df.columns:
            raise KeyError(f"Content column {self.column_name!r} not in relation columns {list(df.columns)}")
        series = df[self.column_name]
        data_version = int(getattr(er, "data_version", 0) or 0)
        cache_key = self._make_cache_key(df, data_version)
        if self._cached_result is not None and self._cache_key == cache_key:
            raw = self._cached_result
        else:
            vocabs = getattr(er, "column_vocabs", None) or {}
            col_vocab = vocabs.get(self.column_name)
            if col_vocab is None:
                col_vocab = vocabs.get(str(self.column_name))
            if is_text_dtype(series):
                self._last_was_text = True
                raw = series
            else:
                self._last_was_text = False
                col = str(self.column_name)
                notice_key = (id(self), col, str(series.dtype))
                if notice_key not in _AUTO_TENSORIZE_NOTICE_KEYS:
                    _AUTO_TENSORIZE_NOTICE_KEYS.add(notice_key)
                    if isinstance(series.dtype, pd.CategoricalDtype):
                        logger.info(
                            "Notice: column %r (categorical) auto-tensorized to (N,) long using column_vocabs / codes. "
                            "Wrap with an explicit encoder in brackets to silence this message.",
                            col,
                        )
                    elif pd.api.types.is_bool_dtype(series):
                        logger.info(
                            "Notice: column %r (bool) auto-tensorized to (N, 1) float32. "
                            "Wrap with an explicit encoder in brackets to silence this message.",
                            col,
                        )
                    elif pd.api.types.is_numeric_dtype(series):
                        logger.info(
                            "Notice: column %r (dtype %s) auto-tensorized to (N, 1) float32. "
                            "Wrap with an explicit encoder in brackets to silence this message.",
                            col,
                            series.dtype,
                        )
                    else:
                        logger.info(
                            "Notice: column %r (dtype %s) auto-tensorized (numeric coercion or similar). "
                            "Wrap with an explicit encoder in brackets to silence this message.",
                            col,
                            series.dtype,
                        )
                raw, _ = tensorize_column(series, col_vocab)
                if isinstance(raw, torch.Tensor):
                    raw = raw.detach()
            self._cached_result = raw
            self._cache_key = cache_key

        if isinstance(raw, torch.Tensor):
            dev = self._target_device or torch.device("cpu")
            dt = self._target_dtype or torch.float32
            if raw.dtype.is_floating_point:
                return raw.to(device=dev, dtype=dt)
            return raw.to(device=dev)
        return raw


class _EncodeWrapper(nn.Module):
    """Applies an ``nn.Module`` encoder to the output of ``_ColumnExtractModule``."""

    def __init__(self, extractor: _ColumnExtractModule, encoder: nn.Module):
        super().__init__()
        self.extractor = extractor
        self.encoder = encoder

    def forward(self, *inputs: Any) -> torch.Tensor:
        raw = self.extractor(*inputs)
        try:
            return self.encoder(raw)
        except Exception as e:
            raise RuntimeError(
                f"{e} (while encoding column {self.extractor.column_name!r})"
            ) from e


class _MultiEncodeModule(nn.Module):
    """Concatenates multiple bracket items: ``[age, Linear(1,d)(x)]``."""

    def __init__(self, item_modules: List[nn.Module]):
        super().__init__()
        self.item_modules = nn.ModuleList(item_modules)

    def forward(self, *inputs: Any) -> torch.Tensor:
        tensors: List[torch.Tensor] = []
        for mod in self.item_modules:
            result = mod(*inputs)
            if not isinstance(result, torch.Tensor):
                col = getattr(mod, "column_name", None)
                if col is None and isinstance(mod, _EncodeWrapper):
                    col = mod.extractor.column_name
                raise EncodeTypeError(
                    f"Text column {col!r} in a multi-item bracket must use an encoder on that column."
                )
            if result.dim() == 1:
                result = result.unsqueeze(-1)
            tensors.append(result)
        return torch.cat(tensors, dim=-1)


def collect_column_extract_leaves(module: nn.Module) -> List[_ColumnExtractModule]:
    return [m for m in module.modules() if isinstance(m, _ColumnExtractModule)]


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


class _InputSelector(nn.Module):
    """Selects one input from *args by index."""

    def __init__(self, index: int, name: Optional[str] = None):
        super().__init__()
        self.index = index
        self.name = name

    def extra_repr(self) -> str:
        return f'"{self.name}"' if self.name is not None else f'index={self.index}'

    def forward(self, *inputs):
        if self.index < len(inputs):
            return inputs[self.index]
        raise IndexError(f"Input index {self.index} out of range (got {len(inputs)} inputs)")


class _ConstantValue(nn.Module):
    """Returns a constant value (registered buffer for numbers, else attribute)."""

    def __init__(self, value):
        super().__init__()
        if isinstance(value, (int, float)):
            self.register_buffer("value", torch.tensor(value))
        else:
            self.value = value

    def extra_repr(self) -> str:
        if isinstance(getattr(self, "value", None), torch.Tensor):
            v = self.value
            if v.numel() <= 1:
                return f"value={v.item()}"
            return f"value=Tensor(shape={tuple(v.shape)})"
        return f"value={getattr(self, 'value', None)!r}"

    def forward(self, *inputs):
        return self.value


class Concat(nn.Module):
    """Concatenates inputs along dim=1."""

    def forward(self, *x):
        return torch.cat(x, dim=1)


# RelNN row-first wrapper; other torch op wrappers can follow this pattern (built-in here with Concat, Tensor).
class ArgMax(nn.Module):
    """
    A transformation module that computes argmax along dimension 1.
    Returns the index of the maximum value for each row, shape (N, 1) for row-first consistency.
    """
    def __init__(self):
        super(ArgMax, self).__init__()

    def forward(self, x):
        # x: [N, C] logits
        # Returns: [N, 1] class indices (row-first: one scalar per row)
        return torch.argmax(x, dim=1, keepdim=True)


def _coerce_computed_int_float(value: Any, source_arith_term: Any) -> Any:
    """Coerce an integer-valued float that *came from arithmetic* to int.

    Background — Python's ``/`` is true-division, so a DSL hyperparam like
    ``Tensor(d/h, d/h)`` with ``d=16, h=4`` evaluates to ``(4.0, 4.0)``. The
    ``_ParameterTensor`` constructor has a "last arg is float → fill value"
    heuristic for the pyHGT-faithful ``Tensor(1, 1.0)`` ones-init pattern;
    that heuristic mis-fires on computed floats and silently turns shape
    ``(4, 4)`` into ``(4,)`` filled with ``4.0``. The symptom surfaces
    downstream as the famous HGT ``transformation_L1`` "batch2 [1, 1] vs
    [1, 4]" matmul error.

    Fix: at the central chokepoint where shape-arith results are produced
    (``TensorTermCompiler._eval_hyperparams``), distinguish DSL **literal
    floats** from **computed** floats by inspecting the source ArithTerm.
    Coerce only computed integer-valued floats; preserve literal floats so
    callers like ``Tensor(1, 1.0)`` continue to get fill_value=1.0.

    Known edge case: if a user defines ``d = 4.0 .`` (a TransformDef whose
    body is a float literal) and uses ``Tensor(d, d)``, the resolved
    substitution still presents ``4.0`` as a literal-shaped ArithTerm and
    the bug returns. Today nobody writes scalar TransformDefs with float
    literals (verified by grep), so this is documented but unhandled.
    """
    if not (isinstance(value, float) and value.is_integer()):
        return value
    is_literal_float = (
        source_arith_term is not None
        and getattr(source_arith_term, "op", None) is None
        and getattr(source_arith_term, "sons", None) is None
        and isinstance(getattr(source_arith_term, "value", None), float)
    )
    return value if is_literal_float else int(value)


class _ParameterTensor(nn.Module):
    """
    Learnable tensor of given shape for use in @ and other ops.
    Weight is registered as nn.Parameter (per PyTorch: assignment as attribute registers it).
    Forward returns the parameter and ignores inputs.
    """

    def __init__(self, *args):
        super().__init__()
        if not args:
            raise ValueError("Tensor(shape) requires at least one dimension (e.g. Tensor(4, 2)).")
        # Optional fill_value: if the last arg is a Python float (not int), treat it as the
        # initialization value.  Shape dims from DSL arithmetic (hidden // num_heads, etc.)
        # are always Python int.  A DSL literal like 1.0 is Python float, so
        #   Tensor(1)       → shape (1,),    fill 0.0  (existing behaviour)
        #   Tensor(1, 1.0)  → shape (1,),    fill 1.0  (matches pyHGT ones init)
        #   Tensor(4, 2)    → shape (4, 2),  fill 0.0  (multi-dim, unchanged)
        if len(args) >= 2 and isinstance(args[-1], float):
            fill_value = float(args[-1])
            shape_args = args[:-1]
        else:
            fill_value = 0.0
            shape_args = args
        if not shape_args:
            raise ValueError("Tensor(shape) requires at least one shape dimension before the optional fill value.")
        shape = tuple(int(s) for s in shape_args)
        self.weight = nn.Parameter(torch.empty(*shape))
        nn.init.constant_(self.weight, fill_value)

    def extra_repr(self) -> str:
        return f"shape={tuple(self.weight.shape)}"

    def forward(self, *inputs):
        return self.weight


# Expose as Tensor for the DSL (e.g. Tensor(4, 2)() in rules).
Tensor = _ParameterTensor


# ---------------------------------------------------------------------------
# DSL-native unary: view, sqrt, transpose (name normalized in compile)
# ---------------------------------------------------------------------------

def _to_int_like(x: Any, *, name: str) -> int:
    if isinstance(x, bool):
        raise TypeError(f"{name} must be an int, got bool")
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        xi = int(x)
        if abs(x - xi) < 1e-9:
            return xi
        raise TypeError(f"{name} must be an int-like value, got {x!r}")
    raise TypeError(f"{name} must be an int-like value, got {type(x)}")


class _UnaryOp(nn.Module):
    """DSL-native unary: sqrt, transpose, view."""

    def __init__(
        self,
        op: str,
        children: list,
        shape: Optional[tuple] = None,
    ):
        super().__init__()
        self.op = op
        self.arg = children[0]
        self.shape = shape

    def extra_repr(self) -> str:
        s = f'op={self.op!r}'
        if self.shape is not None:
            s += f', shape={self.shape}'
        return s

    def forward(self, *inputs):
        x = self.arg(*inputs)
        if self.op == "sqrt":
            return torch.sqrt(x)
        if self.op == "transpose":
            return smart_transpose(x)
        if self.op == "view":
            if self.shape is None or len(self.shape) == 0:
                raise ValueError("view requires at least one dimension")
            return smart_view(x, *self.shape)
        raise NotImplementedError(self.op)


# ---------------------------------------------------------------------------
# Compiler
# ---------------------------------------------------------------------------


class TensorTermCompiler:
    """Compiles a TensorTerm tree to an nn.Module using engine for symbol table and ArithTerm evaluation."""

    def __init__(self, engine):
        self._engine = engine

    def compile(
        self,
        tterm: TensorTerm,
        var_to_input_index: Optional[Dict[str, int]] = None,
    ) -> nn.Module:
        return self._compile(tterm, var_to_input_index)

    def _compile(
        self,
        tterm: TensorTerm,
        var_to_input_index: Optional[Dict[str, int]] = None,
    ) -> nn.Module:
        if tterm.op is None:
            return self._compile_leaf(tterm, var_to_input_index)

        op_name = tterm.op.op
        child_modules = [
            self._compile(son, var_to_input_index)
            for son in (tterm.sons or [])
        ]

        # Arithmetic
        arithmetic_ops = {
            "*": smart_mul,
            "+": smart_add,
            "-": smart_sub,
            "/": smart_div,
            "@": smart_matmul,
            "**": smart_pow,
        }
        if op_name == "==":
            if len(child_modules) != 2:
                raise ValueError(f"Equality '==' requires 2 operands, got {len(child_modules)}")
            return _EqualityWrapper(child_modules)
        if op_name in arithmetic_ops:
            if len(child_modules) != 2:
                raise ValueError(f"Arithmetic '{op_name}' requires 2 operands, got {len(child_modules)}")
            return _ArithmeticWrapper(arithmetic_ops[op_name], child_modules)

        # DSL-native unary (normalize name)
        op_lower = op_name.lower() if op_name else ""
        if op_lower in ("sqrt", "transpose", "view"):
            if op_lower == "view":
                hp = tterm.op.hyper_params or []
                dims = tuple(
                    _to_int_like(self._engine.evaluate_arith_term_for_hyperparams(hp_i), name=f"view dim[{i}]")
                    for i, hp_i in enumerate(hp)
                )
                return _UnaryOp("view", child_modules, shape=dims)
            unary_op = op_lower
            return _UnaryOp(unary_op, child_modules)

        # TransformDef
        sym = self._engine.get_symbol(op_name)
        if sym is not None:
            typ, obj = sym
            if typ == TransformDef and getattr(obj, "tensor_term", None) is not None:
                inner = self._compile(obj.tensor_term, var_to_input_index)
                return _TransformDefWrapper(inner, child_modules)

        # Named op: resolve and instantiate
        resolved = resolve_op(op_name, self._engine.get_run_globals())
        if resolved is None:
            raise NotImplementedError(
                f"Op '{op_name}' not found in run scope or torch.nn/torch. "
                "Import the module (e.g. from torch.nn import Linear, ReLU) in the scope that calls session.run()."
            )
        if resolved is _DSL_NATIVE_UNARY:
            raise NotImplementedError(f"DSL-native op '{op_name}' should have been handled above")

        # Function-style callable ops (e.g., unsqueeze(z1, 1)) use sons as positional arguments.
        # Preserve backward compatibility for zero-arg callable factories from run_globals.
        if callable(resolved) and not inspect.isclass(resolved) and self._should_compile_as_function_call(
            resolved, len(child_modules)
        ):
            return _CallableFunctionWrapper(resolved, child_modules, op_name=op_name)

        # Classify sons: if all are ctor args (not runtime vars), promote to hyper_params.
        promoted = self._maybe_promote_ctor_args(tterm, resolved, var_to_input_index)
        if promoted is not tterm:
            tterm = promoted
            child_modules = []

        module = self._instantiate(resolved, tterm)
        # Tensor(4, 2) parsed as instance call: sons are shape dims, not input tensors
        if resolved is _ParameterTensor and not (tterm.op.hyper_params or []) and tterm.sons:
            child_modules = []
        child_names = self._child_names_from_sons(tterm.sons) if (tterm.sons and len(child_modules) > 1) else None
        return self._wrap_module(module, op_name, child_modules, child_names=child_names)

    def _child_names_from_sons(self, sons: Optional[List[TensorTerm]]) -> Optional[List[str]]:
        """Logical names for each son (TransformDef/op name or str(i)) for FQN/display."""
        if not sons:
            return None
        names: List[str] = []
        for i, son in enumerate(sons):
            op = getattr(son, "op", None)
            if isinstance(op, TensorOp) and isinstance(getattr(op, "op", None), str):
                names.append(str(op.op))
            else:
                names.append(str(i))
        return names

    def _compile_content_encode(self, enc: ContentEncode) -> nn.Module:
        """Compile ``ContentEncode`` (RHS bracket) to a module tree."""
        item_modules: List[nn.Module] = []
        g = self._engine.get_run_globals()
        for item in enc.items:
            leaf = _ColumnExtractModule(column_name=item.column.name)
            if item.encoder_name:
                resolved = resolve_op(item.encoder_name, g)
                if resolved is None:
                    raise ValueError(
                        f"Unknown encoder {item.encoder_name!r} in […] bracket. "
                        "Import the module in the run scope passed to session.define()."
                    )
                mock_tterm = TensorTerm(
                    op=TensorOp(op=item.encoder_name, hyper_params=item.encoder_params or []),
                )
                encoder_mod = self._instantiate(resolved, mock_tterm)
                item_modules.append(_EncodeWrapper(extractor=leaf, encoder=encoder_mod))
            else:
                item_modules.append(leaf)
        if len(item_modules) == 1:
            return item_modules[0]
        return _MultiEncodeModule(item_modules)

    def _son_kind(self, t: TensorTerm, runtime_vars: set) -> str:
        """Classify a TensorTerm son as 'ctor' or 'input'.

        Leaves:
          - Var in runtime_vars → 'input'
          - any other Var (symbol-table name, template param, True/False) → 'ctor'
          - numeric/string literal → 'ctor'
        Expressions:
          - pure-arithmetic sub-tree (+,-,*,/,@,**) → 'ctor' iff every leaf is 'ctor'
          - any other op (module call, transform def, etc.) → 'input'
        """
        if t.op is None:
            if isinstance(t.value, Var):
                return "input" if t.value.name in runtime_vars else "ctor"
            return "ctor"
        op_name = t.op.op if isinstance(t.op, TensorOp) else None
        if op_name in _PURE_ARITH_OPS:
            for s in (t.sons or []):
                if self._son_kind(s, runtime_vars) == "input":
                    return "input"
            return "ctor"
        return "input"

    def _maybe_promote_ctor_args(
        self,
        tterm: TensorTerm,
        resolved,
        var_to_input_index: Optional[Dict[str, int]],
    ) -> TensorTerm:
        """For an nn.Module class with no hyper_params and at least one son, classify each son.

        - All ctor  → promote sons to hyper_params (returns a new TensorTerm).
        - All input → return tterm unchanged (sons stay as tensor inputs).
        - Mixed     → raise; user must use the two-paren form `Module(ctor_args)(inputs)`.
        """
        if not (inspect.isclass(resolved) and issubclass(resolved, nn.Module)):
            return tterm
        if (tterm.op.hyper_params or []) or not tterm.sons:
            return tterm
        runtime_vars = set(var_to_input_index.keys()) if var_to_input_index else set()
        kinds = [self._son_kind(s, runtime_vars) for s in tterm.sons]
        if all(k == "ctor" for k in kinds):
            new_hp = [
                _arith_term_bool_var_to_primitive(tensor_term_to_arith_term(s))
                for s in tterm.sons
            ]
            return TensorTerm(
                op=TensorOp(
                    op=tterm.op.op,
                    template_args=tterm.op.template_args,
                    hyper_params=new_hp,
                ),
                sons=None,
                value=None,
            )
        if any(k == "ctor" for k in kinds):
            raise ValueError(
                f"Single-paren call to '{tterm.op.op}' mixes constructor args and tensor "
                f"inputs. Use the two-paren form `{tterm.op.op}(ctor_args)(inputs)`."
            )
        return tterm

    def _compile_leaf(
        self,
        tterm: TensorTerm,
        var_to_input_index: Optional[Dict[str, int]],
    ) -> nn.Module:
        if tterm.value is not None:
            if isinstance(tterm.value, ContentEncode):
                return self._compile_content_encode(tterm.value)
            if isinstance(tterm.value, Var):
                var_name = tterm.value.name
                if var_to_input_index is not None and var_name in var_to_input_index:
                    return _InputSelector(var_to_input_index[var_name], name=var_name)
                if var_name.startswith("z"):
                    try:
                        idx = int(var_name[1:]) - 1
                        if idx >= 0:
                            return _InputSelector(idx, name=var_name)
                    except (ValueError, AttributeError):
                        pass
                return _InputSelector(0)
            return _ConstantValue(tterm.value)
        return _InputSelector(0)

    def _tensor_term_to_hyperparam_value(self, t: TensorTerm) -> Any:
        """Evaluate a TensorTerm (constant or arith subtree) to a scalar for shape/hyperparams."""
        return self._engine.evaluate_arith_term_for_hyperparams(tensor_term_to_arith_term(t))

    def _eval_hyperparams(self, tterm: TensorTerm, resolved: Union[type, Callable, nn.Module]) -> list:
        """Evaluate hyperparams from tterm; for Tensor with no hyper_params, use sons as shape dims.

        Computed integer-valued floats (e.g. ``d/h`` evaluating to ``4.0``
        because Python's ``/`` is true-division) are coerced to int so they
        are read as shape dims, not as fill values by the
        ``_ParameterTensor`` heuristic. See ``_coerce_computed_int_float``.
        """
        # resolved = op for this term (class/callable/module from resolve_op); used to special-case Tensor(shape).
        hp_terms = tterm.op.hyper_params or []
        if hp_terms:
            return [
                _coerce_computed_int_float(
                    self._engine.evaluate_arith_term_for_hyperparams(hp), hp
                )
                for hp in hp_terms
            ]
        # Fallback for `_ParameterTensor` whose ctor args still live in `sons`
        # rather than `op.hyper_params`. In practice `_maybe_promote_ctor_args`
        # (called from `_compile`) promotes them to `op.hyper_params` upstream
        # whenever all sons are "ctor" kind, so this path is unreachable for
        # the DSL `Tensor(...)` invocation. Belt-and-suspenders: route values
        # through `_coerce_computed_int_float` anyway so a future refactor
        # that lands here doesn't silently reintroduce the `Tensor(d/h, d/h)`
        # bug shape.
        if resolved is _ParameterTensor and tterm.sons:
            return [
                _coerce_computed_int_float(
                    self._tensor_term_to_hyperparam_value(s),
                    tensor_term_to_arith_term(s),
                )
                for s in tterm.sons
            ]
        return []

    def _instantiate(self, resolved: Union[type, Callable, nn.Module], tterm: TensorTerm) -> nn.Module:
        if isinstance(resolved, nn.Module):
            return resolved
        eval_hp = self._eval_hyperparams(tterm, resolved)
        if resolved is _ParameterTensor:
            return _ParameterTensor(*eval_hp)
        target = resolved.__init__ if inspect.isclass(resolved) else resolved
        sig = inspect.signature(target)
        params = [
            (n, p) for n, p in sig.parameters.items()
            if n != "self" and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        names = [n for n, _ in params]

        def _coerce_for_param(value: Any, param: inspect.Parameter) -> Any:
            # DSL arithmetic (e.g. d / h) produces float; coerce integral
            # floats to int for ctor params annotated as int (nn.Linear, etc.)
            ann = param.annotation
            if isinstance(value, float) and float(value).is_integer() and ann is int:
                return int(value)
            return value

        coerced_hp = [
            _coerce_for_param(eval_hp[i], params[i][1]) if i < len(params) else eval_hp[i]
            for i in range(len(eval_hp))
        ]

        force_row_wise = (
            inspect.isclass(resolved)
            and issubclass(resolved, nn.modules.loss._Loss)
        )

        try:
            instance = resolved(*coerced_hp[: len(names)])
        except TypeError:
            kwargs = {names[i]: coerced_hp[i] for i in range(min(len(coerced_hp), len(names)))}
            instance = resolved(**kwargs)

        if force_row_wise and getattr(instance, "reduction", None) != "none":
            logger.debug("Forcing reduction='none' on %s for per-row semantics", resolved.__name__)
            instance.reduction = "none"

        return instance

    # Dispatch: 0 children -> error if module has .weight and is not _ParameterTensor, else _NoChildWrapper.
    # 1 child -> _SingleChildWrapper. 2+ children -> _MultiArgWrapper only if Concat or module arity >= 2;
    # else error (use explicit Concat).
    def _wrap_module(
        self,
        module: nn.Module,
        op_name: str,
        child_modules: list,
        child_names: Optional[List[str]] = None,
    ) -> nn.Module:
        if len(child_modules) == 0:
            if hasattr(module, "weight") and isinstance(getattr(module, "weight"), torch.Tensor):
                if type(module) is not _ParameterTensor:
                    raise NotImplementedError(
                        f"Use Tensor(shape) for a learnable matrix in @ (e.g. Tensor(4, 2)). "
                        f"Module {type(module).__name__} with no arguments is not supported."
                    )
            return _NoChildWrapper(module)
        if len(child_modules) == 1:
            return _SingleChildWrapper(module, child_modules, op_name=op_name)
        # Multiple children: Concat or module accepts multiple args -> _MultiArgWrapper; else error
        if op_name == "Concat" or self._forward_arity(module) >= 2:
            adapter = MULTI_ARG_INPUT_ADAPTERS.get(op_name)
            return _MultiArgWrapper(
                module, child_modules, op_name=op_name, input_adapter=adapter, child_names=child_names
            )
        raise NotImplementedError(
            f"Module {type(module).__name__} expects 1 argument but got {len(child_modules)}. "
            "Use Concat(z1, z2, ...) as the single argument."
        )

    def _forward_arity(self, module: nn.Module) -> int:
        sig = inspect.signature(module.forward)
        return sum(
            1 for n, p in sig.parameters.items()
            if n != "self" and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        )

    @staticmethod
    def _should_compile_as_function_call(resolved: Callable[..., Any], num_children: int) -> bool:
        """
        Decide whether a non-class callable should be treated as op(arg1, ...).
        For compatibility, callable factories with zero required positional args keep constructor path.
        """
        if num_children == 0:
            return False
        try:
            sig = inspect.signature(resolved)
        except (TypeError, ValueError):
            # Builtins without inspectable signatures are treated as function-style calls when args exist.
            return True
        if any(p.kind == p.VAR_POSITIONAL for p in sig.parameters.values()):
            # Varargs callables are function-style by construction when arguments are present.
            return True
        params = [
            p
            for p in sig.parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        required = [p for p in params if p.default is inspect._empty]
        if len(required) == 0:
            return False
        return True


# ---------------------------------------------------------------------------
# Wrappers
# ---------------------------------------------------------------------------


class _EqualityWrapper(nn.Module):
    def __init__(self, children):
        super().__init__()
        self.left = children[0]
        self.right = children[1]

    def forward(self, *inputs):
        left = self.left(*inputs)
        right = self.right(*inputs)
        return torch.eq(left, right).int()


_ARITH_OP_SYMBOL = {
    smart_mul: "*",
    smart_add: "+",
    smart_sub: "-",
    smart_div: "/",
    smart_matmul: "@",
    smart_pow: "**",
}


class _ArithmeticWrapper(nn.Module):
    def __init__(self, op_func, children):
        super().__init__()
        self.op_func = op_func
        self.left = children[0]
        self.right = children[1]

    def extra_repr(self) -> str:
        return f"op={_ARITH_OP_SYMBOL.get(self.op_func, self.op_func)!r}"

    def forward(self, *inputs):
        left = self.left(*inputs)
        right = self.right(*inputs)
        return self.op_func(left, right)


class _TransformDefWrapper(nn.Module):
    def __init__(self, inner_mod, children):
        super().__init__()
        self.inner = inner_mod
        if children and len(children) == 1:
            self.arg = children[0]
            self.children_modules = None
        elif children and len(children) > 1:
            self.arg = None
            self.children_modules = nn.ModuleList(children)
        else:
            self.arg = None
            self.children_modules = None

    def forward(self, *inputs):
        if self.arg is not None:
            return self.inner(self.arg(*inputs))
        if self.children_modules is not None:
            child_outputs = [c(*inputs) for c in self.children_modules]
            return self.inner(*child_outputs)
        return self.inner(*inputs)


class _NoChildWrapper(nn.Module):
    """Used when the tensor term has no children; forward passes join inputs to the module."""
    def __init__(self, module):
        super().__init__()
        self._module = module

    def extra_repr(self) -> str:
        return f'module={type(self._module).__name__}'

    def forward(self, *inputs):
        return self._module(*inputs)


class _CallableFunctionWrapper(nn.Module):
    """Used for function-style callables: forward calls op(*evaluated_child_outputs)."""

    def __init__(self, op_func: Callable[..., Any], children: list, op_name: str = ""):
        super().__init__()
        self._op_func = op_func
        self.children_modules = nn.ModuleList(children)
        self._op_name = op_name

    def extra_repr(self) -> str:
        return f'op={self._op_name!r}' if self._op_name else ""

    def forward(self, *inputs):
        args = [c(*inputs) for c in self.children_modules]
        return self._op_func(*args)


# Attribute name under which the single child is registered (for FQN mapping in engine when child is a TransformDef)
SINGLE_CHILD_ATTR = "input"


class _SingleChildWrapper(nn.Module):
    """Used when the tensor term has one child; forward is module(child(*inputs))."""

    def __init__(self, module, children, op_name: str = ""):
        super().__init__()
        self._module = module
        setattr(self, SINGLE_CHILD_ATTR, children[0])
        self._op_name = op_name

    def extra_repr(self) -> str:
        return f'op={self._op_name!r}' if self._op_name else ''

    def forward(self, *inputs):
        x = getattr(self, SINGLE_CHILD_ATTR)(*inputs)
        return self._module(x)


class _MultiArgWrapper(nn.Module):
    """Used when the tensor term has 2+ children and module accepts multiple args (e.g. Concat, CrossEntropyLoss).
    Optional input_adapter normalizes args before calling the module; registry: MULTI_ARG_INPUT_ADAPTERS.
    _child_names: logical names (e.g. TransformDef names) for FQN/display so params show as K._module.weight not children_modules.0._module.weight."""

    def __init__(
        self,
        module: nn.Module,
        children: list,
        op_name: str = "",
        input_adapter: Optional[Callable[..., Tuple[torch.Tensor, ...]]] = None,
        child_names: Optional[List[str]] = None,
    ):
        super().__init__()
        self._module = module
        self.children_modules = nn.ModuleList(children)
        self._op_name = op_name
        self._input_adapter = input_adapter
        self._child_names = child_names  # for parameter FQN rewrite (engine/relnn)

    def extra_repr(self) -> str:
        return f'op={self._op_name!r}' if self._op_name else ''

    def forward(self, *inputs):
        args = [c(*inputs) for c in self.children_modules]
        if self._input_adapter is not None:
            args = list(self._input_adapter(*args))
        return self._module(*args)


__all__ = [
    "resolve_op",
    "tensor_term_to_arith_term",
    "TensorTermCompiler",
    "ArgMax",
    "Concat",
    "Tensor",
    "SINGLE_CHILD_ATTR",
    "_InputSelector",
    "_ConstantValue",
    "collect_column_extract_leaves",
    "replace_submodule",
]
