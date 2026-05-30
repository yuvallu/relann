# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
from __future__ import annotations
# %% [markdown]
# # pydantic_classes
#
# > Fill in a module description here

# %%
if __name__ == "__main__":

    import numpy as np
    from pandas.api.extensions import ExtensionDtype
    from pandas.errors import AbstractMethodError

# %%
if __name__ == "__main__":


    def all_pandas_dtypes() -> list[str]:
        # 1. Core NumPy dtypes via the scalar-type registry
        core = sorted({np.dtype(t).name for t in set(np.sctypeDict.values())})
        
        # 2. Concrete pandas ExtensionDtypes
        ext = []
        for cls in ExtensionDtype.__subclasses__():
            try:
                name = cls().name
            except (AbstractMethodError, TypeError):
                # skip abstract or constructor-arg dtypes
                continue
            ext.append(name)
        ext = sorted(set(ext))
        
        return core + ext

# %%
import logging
from collections import Counter
from typing import List, Optional, Any, Union, Callable, Tuple, Dict, ForwardRef, Literal
from pydantic import BaseModel, ConfigDict, validator, Field

from relann.column_ref import ColumnRef
import torch
import torch.nn as nn
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# %%
# Allowed primitive types for type annotations in the DSL
PrimitiveType = Literal["int", "float", "bool", "str"]
REL_OP_NAME = Literal["Join", "Union"]
REL_OP = Literal[",", "|"]
COMP_OP = Literal["==", "!=", ">", ">=", "<", "<="]
ArithOp = Literal["+", "-", "*", "/", "**"]

# Define a Literal for content attribute dtypes (pandas/numpy dtypes for completeness)
DType = Literal[
    "int","bool","bytes","complex64","complex128","complex256",
    "datetime64","float","float16","float32","float64","float128",
    "int8","int16","int32","int64","uint8","uint16","uint32","uint64",
    "object","str","timedelta64","void","Sparse[float64, nan]","category"
]

class BoundedRHS(BaseModel):
    rel_op_name: REL_OP_NAME
    main_er: EmbeddedRelation
    bounding_ers: List[EmbeddedRelation] = Field(default_factory=list)
    bounding_conditions: List[ComparisonExpression] = Field(default_factory=list)  # Always a list, sometimes empty

class ERRef(BaseModel):
    name: str
    template_args: Optional[List[ArithTerm]] = None

class EmbeddedRelation(BaseModel):
    """
    Represents either an ER instance or a function call in the DSL.
    If 'arguments' is populated, this object represents a function call; otherwise, it is a standard ER instance.
    """
    name: str
    template_args: Optional[List[ArithTerm]] = None
    # Add semantic check when its a function call (arguments is not None).
    arguments: Optional[List[ERRef]] = None  # Only populated for function calls; None or empty for standard ERs
    content_attrs: List[Var] = Field(default_factory=list)
    embedding_var: Optional[Union[Var, int, float]] = None

class ComparisonExpression(BaseModel):
    lhs: ArithTerm    # e.g., 0
    comp_op: COMP_OP  # e.g., "=="
    rhs: ArithTerm    # e.g., i + j
    # TODO: add a semantic check that these could also be content_attrs

def _find_col_index(attrs: List[Any], name: str) -> Optional[int]:
    """Index of first attribute with .name == name (for Var), or None."""
    for i, a in enumerate(attrs or []):
        if getattr(a, "name", None) == name:
            return i
    return None


class RHS(BaseModel):
    ers: List[EmbeddedRelation]  # Only EmbeddedRelation or FunctionCall
    rel_ops: Optional[List[REL_OP]] = None  # e.g., [",", "|"], can be empty if only one ER
    filter_expressions: List[ComparisonExpression] = Field(default_factory=list)

    @property
    def join_conditions(self) -> List[Dict[str, Any]]:
        """Infer join conditions from overlapping column names across ers (natural join)."""
        if not self.ers or len(self.ers) < 2:
            return []
        rel_ops = self.rel_ops or []
        if "," not in rel_ops:
            return []
        col_counts: Counter = Counter()
        for er in self.ers:
            attrs = er.content_attrs or []
            names = [getattr(a, "name", None) for a in attrs if hasattr(a, "name")]
            col_counts.update(set(n for n in names if n is not None))
        key_names = sorted(col for col, count in col_counts.items() if count >= 2)
        result: List[Dict[str, Any]] = []
        for key_name in key_names:
            refs: List[ColumnRef] = []
            for input_idx, er in enumerate(self.ers):
                attrs = er.content_attrs or []
                col_idx = _find_col_index(attrs, key_name)
                if col_idx is not None:
                    refs.append(ColumnRef(input_idx, col_idx))
            if len(refs) >= 2:
                result.append({"key_name": key_name, "normalized_refs": refs})
        return result

    @property
    def input_schemas(self) -> List[List[str]]:
        """Get the schema (column names) for each input relation."""
        return [
            [attr.name for attr in (er.content_attrs or []) if hasattr(attr, "name")]
            for er in self.ers
        ]

    @property
    def merge_steps(self) -> List[Dict[str, Any]]:
        """
        Pre-compute merge steps for sequential joins.
        Each step merges the accumulated result (inputs[0..step-1]) with inputs[step].
        
        Returns a list of merge step dictionaries with:
        - step: int - which input to merge (1 = merge inputs[0] with inputs[1])
        - left_refs: List[ColumnRef] - ColumnRefs from accumulated side
        - right_refs: List[ColumnRef] - ColumnRefs from right side (inputs[step])
        - key_names: List[str] - key names for this step (for error messages)
        """
        if not self.ers or len(self.ers) < 2:
            return []
        
        rel_ops = self.rel_ops or []
        if "," not in rel_ops:
            return []
        
        join_conds = self.join_conditions
        if not join_conds:
            return []
        
        merge_steps = []
        
        # For each merge step (step i merges accumulated inputs[0..i-1] with inputs[i])
        for step in range(1, len(self.ers)):
            left_refs = []
            right_refs = []
            key_names = []
            
            # For each join condition, find refs that apply to this step
            for cond in join_conds:
                key_name = cond.get("key_name")
                normalized_refs = cond.get("normalized_refs", [])
                
                if not normalized_refs:
                    continue
                
                # Find the best left ref (from inputs[0..step-1], prefer latest)
                left_ref = None
                right_ref = None
                
                for ref in normalized_refs:
                    if not isinstance(ref, ColumnRef):
                        continue
                    
                    if ref.input_idx < step:
                        # From accumulated side
                        if left_ref is None or ref.input_idx > left_ref.input_idx:
                            left_ref = ref
                    elif ref.input_idx == step:
                        # From right side
                        right_ref = ref
                
                # If we have both left and right refs, add them to this step
                if left_ref is not None and right_ref is not None:
                    left_refs.append(left_ref)
                    right_refs.append(right_ref)
                    if key_name:
                        key_names.append(key_name)
            
            if left_refs and right_refs:
                merge_steps.append({
                    "step": step,
                    "left_refs": left_refs,
                    "right_refs": right_refs,
                    "key_names": key_names
                })
        
        return merge_steps

    @property
    def output_content_attrs(self) -> List[Var]:
        """
        Compute the output content_attrs for a join operation using aliased names from join_conditions.
        
        This property:
        - Uses key_name aliases from join_conditions (not original column names)
        - Collapses duplicate join keys (only adds each key once)
        - Adds non-key columns from all input relations
        - Handles name collisions by suffixing with input index
        
        Returns a list of Var objects representing the output schema.
        """
        if not self.ers or len(self.ers) < 2:
            # Single relation: return its content_attrs
            if self.ers:
                return list(self.ers[0].content_attrs or [])
            return []
        
        rel_ops = self.rel_ops or []
        if "," not in rel_ops:
            # Not a join: return first relation's schema
            return list(self.ers[0].content_attrs or [])
        
        join_conds = self.join_conditions
        if not join_conds:
            # No join conditions: concatenate all schemas (with collision handling)
            used = set()
            out = []
            for i, er in enumerate(self.ers):
                for attr in (er.content_attrs or []):
                    if isinstance(attr, Var):
                        name = attr.name
                        if name in used:
                            name = f"{name}_{i}"
                        used.add(name)
                        out.append(Var(name=name))
            return out
        
        # Build key_map: (input_idx, col_idx) -> key_name (aliased name)
        key_map = {}
        for jc in join_conds:
            key_name = jc.get("key_name")
            if not key_name:
                continue
            for ref in (jc.get("normalized_refs") or []):
                if isinstance(ref, ColumnRef):
                    key_map[(ref.input_idx, ref.column_idx)] = key_name
        
        used = set()  # Track names already added
        out = []
        
        def add_var(attr: Var, input_idx: int):
            """Add a Var to output, handling name collisions."""
            name = attr.name
            if name in used:
                name = f"{name}_{input_idx}"
            used.add(name)
            out.append(Var(name=name))
        
        # Process first relation: use key_name if it's a join key, otherwise use original name
        for j, attr in enumerate(self.ers[0].content_attrs or []):
            if not isinstance(attr, Var):
                continue
            key_name = key_map.get((0, j))
            if key_name:
                # This is a join key - use the aliased name
                if key_name not in used:
                    used.add(key_name)
                    out.append(Var(name=key_name))
                # else: key already added, skip
            else:
                # Not a join key - add as regular column
                add_var(attr, 0)
        
        # Process subsequent relations
        for i in range(1, len(self.ers)):
            for j, attr in enumerate(self.ers[i].content_attrs or []):
                if not isinstance(attr, Var):
                    continue
                key_name = key_map.get((i, j))
                if key_name:
                    # This is a join key - use aliased name, but only add once
                    if key_name not in used:
                        used.add(key_name)
                        out.append(Var(name=key_name))
                    # else: key already added from earlier relation, skip
                else:
                    # Not a join key - add as regular column with collision handling
                    add_var(attr, i)
        
        return out


class EmbeddingExpression(BaseModel):
    aggregation_fn: Optional[str] = None  # e.g., "avg", "sum", "min", "max", or None/"identity"
    tensor_term: Optional[TensorTerm] = None

class DerivedER(BaseModel):
    name: str
    template_params: Optional[List[Any]] = None
    derived_content_attrs: List[Union[Primitive, Var, "ContentDecode"]]
    embedding_expression: EmbeddingExpression

    @property
    def group_by_column_names(self) -> List[str]:
        """Column names to group by (from LHS content attrs). Used by aggregation."""
        result: List[str] = []
        for attr in self.derived_content_attrs or []:
            if isinstance(attr, ContentDecode):
                result.append(attr.column.name)
            elif isinstance(attr, Var):
                result.append(attr.name)
            else:
                result.append(str(attr))
        return result

class Rule(BaseModel):
    lhs: DerivedER
    rhs: Union[RHS, BoundedRHS]

class ERSchema(BaseModel):
    content_attr_types: List[DType]
    embedding_dims: Optional[List[int]] = None

class ErParam(BaseModel):
    name: str
    er_schema: Optional[ERSchema] = None

class FunctionDef(BaseModel):
    name: str
    template_params: Optional[List[Union[Primitive, Var]]] = None
    er_params: List[ErParam] = Field(default_factory=list)
    return_type: Optional[ERSchema] = None
    function_body: List[Union[TransformDef, Rule]] = Field(default_factory=list)

def is_var(obj: Any) -> bool:
    return isinstance(obj, Var)

class Var(BaseModel):
    name: str
    
class VarTemplated(BaseModel):
    name: str
    template_params: List[Union["Primitive", "Var"]]

Primitive = Union[int, float, str, bool]

class ArithTerm(BaseModel):
    op: Optional[ArithOp] = None
    sons: Optional[List[ArithTerm]] = None  # sub-terms (children)
    value: Optional[Union[Primitive, Var]] = None  # for constants or variable names

class TensorOp(BaseModel):
    op: str  # e.g., "Linear", "*", "+", etc.
    hyper_params: Optional[List[ArithTerm]] = None  # e.g., [16, 64] for Linear(16,64)
    template_args: Optional[List[ArithTerm]] = None  # e.g., [64] for Lin<64>(z)

class EncodeItem(BaseModel):
    """One item inside RHS encode brackets: bare column or encoder-wrapped column."""
    column: Var
    encoder_name: Optional[str] = None
    encoder_params: Optional[List[ArithTerm]] = None


class ContentEncode(BaseModel):
    """RHS ``[...]`` bracket: one or more encode items, concatenated."""
    items: List[EncodeItem]


class ContentDecode(BaseModel):
    """LHS ``[...]`` in predict rules: decode embedding tensor to content."""
    column: Var
    decoder_name: Optional[str] = None
    decoder_params: Optional[List[ArithTerm]] = None


class TensorTerm(BaseModel):
    op: Optional[TensorOp] = None
    sons: Optional[List[TensorTerm]] = None  # sub-terms
    value: Optional[Union[Primitive, Var, VarTemplated, ContentEncode]] = None

# Names the Lark grammar emits as ``Var`` but which are really Python literals,
# not formal parameters. Kept in sync with
# ``relann.tensor_term_compiler._arith_term_bool_var_to_primitive``.
_RESERVED_VAR_LITERAL_NAMES = frozenset({"True", "False", "None"})


def collect_formal_vars(body: Optional["TensorTerm"]) -> List[str]:
    """Ordered unique ``Var`` names that appear as leaves in *body*, treating
    reserved literal names (``True``, ``False``, ``None``) as non-formals.

    A ``TransformDef`` body is a lambda; this returns its formal parameters.
    After the engine has resolved scalar Vars (`d = 96`) and op-name aliases
    (`L1`, `Mu_L2`), any ``Var`` leaf that remains in the body — *except* the
    reserved literals the parser also encodes as ``Var`` — is a formal that
    should be β-substituted with the call-site argument.
    """
    seen: List[str] = []
    def walk(t: Optional["TensorTerm"]) -> None:
        if t is None:
            return
        if (
            t.value is not None
            and isinstance(t.value, Var)
            and t.value.name not in _RESERVED_VAR_LITERAL_NAMES
            and t.value.name not in seen
        ):
            seen.append(t.value.name)
        for s in (t.sons or ()):
            walk(s)
    walk(body)
    return seen


class TransformDef(BaseModel):
    name: str
    template_params: Optional[List[Union[Primitive, Var]]] = None
    tensor_term: TensorTerm
    # NOTE: there is no `formal_params` field on the data model. Formals are
    # inferred fresh from the *resolved* body at substitution time via
    # ``collect_formal_vars`` — see ``relann/engine.py::_apply_call_argument``
    # for why pre-storing them would be stale after template materialization.

class FitStatement(BaseModel):
    fit_params: Dict[str, ArithTerm] = Field(default_factory=dict)
    # Add semantic check that the rule has zero content_attrs and a loss function.
    rule: Rule

class PredictStatement(BaseModel):
    rule: Rule

# TODO - maybe we should have a list of names and not all the Statements.
class Program(BaseModel):
    statements: List[Union[
        TransformDef, FunctionDef, Rule, FitStatement, PredictStatement
    ]] = Field(default_factory=list)

# Allowed aggregation functions for embedding expressions
ALLOWED_AGGREGATIONS = ['avg','min','max','sum', 'mean', 'add', 'count']

# %%
if __name__ == "__main__":
    # Demonstration of RHS.output_content_attrs property
    # This shows how output_content_attrs elegantly computes join output schemas using aliased names

    print("=" * 60)
    print("Demonstrating RHS.output_content_attrs property")
    print("=" * 60)

    # Example 1: (a,b) join (b,c) should output [a,b,c]
    print("\n1. (a,b) join (b,c):")
    er1 = EmbeddedRelation(name="R1", content_attrs=[Var(name="a"), Var(name="b")])
    er2 = EmbeddedRelation(name="R2", content_attrs=[Var(name="b"), Var(name="c")])
    rhs1 = RHS(ers=[er1, er2], rel_ops=[","])
    output1 = [v.name for v in rhs1.output_content_attrs]
    print(f"   Input: R1(a,b), R2(b,c)")
    print(f"   Output: {output1}")
    expected1 = ['a', 'b', 'c']
    print(f"   ✓ Expected: {expected1}")
    assert output1 == expected1, f"Output does not match expected: {output1} != {expected1}"

    # Example 2: (a,b) join (a,b) join (b,c) should output [a,b,c]
    print("\n2. (a,b) join (a,b) join (b,c):")
    er3 = EmbeddedRelation(name="R1", content_attrs=[Var(name="a"), Var(name="b")])
    er4 = EmbeddedRelation(name="R2", content_attrs=[Var(name="a"), Var(name="b")])
    er5 = EmbeddedRelation(name="R3", content_attrs=[Var(name="b"), Var(name="c")])
    rhs2 = RHS(ers=[er3, er4, er5], rel_ops=[",", ","])
    output2 = [v.name for v in rhs2.output_content_attrs]
    print(f"   Input: R1(a,b), R2(a,b), R3(b,c)")
    print(f"   Output: {output2}")
    expected2 = ['a', 'b', 'c']
    print(f"   ✓ Expected: {expected2}")
    assert output2 == expected2, f"Output does not match expected: {output2} != {expected2}"

    # Example 3: (a,b) join (b) should output [a,b]
    print("\n3. (a,b) join (b):")
    er6 = EmbeddedRelation(name="R1", content_attrs=[Var(name="a"), Var(name="b")])
    er7 = EmbeddedRelation(name="R2", content_attrs=[Var(name="b")])
    rhs3 = RHS(ers=[er6, er7], rel_ops=[","])
    output3 = [v.name for v in rhs3.output_content_attrs]
    print(f"   Input: R1(a,b), R2(b)")
    print(f"   Output: {output3}")
    expected3 = ['a', 'b']
    print(f"   ✓ Expected: {expected3}")
    assert output3 == expected3, f"Output does not match expected: {output3} != {expected3}"

    # Example 4: Chain join (a,b) join (b,c) join (c,d)
    print("\n4. Chain join (a,b) join (b,c) join (c,d):")
    er8 = EmbeddedRelation(name="R1", content_attrs=[Var(name="a"), Var(name="b")])
    er9 = EmbeddedRelation(name="R2", content_attrs=[Var(name="b"), Var(name="c")])
    er10 = EmbeddedRelation(name="R3", content_attrs=[Var(name="c"), Var(name="d")])
    rhs4 = RHS(ers=[er8, er9, er10], rel_ops=[",", ","])
    output4 = [v.name for v in rhs4.output_content_attrs]
    print(f"   Input: R1(a,b), R2(b,c), R3(c,d)")
    print(f"   Output: {output4}")
    expected4 = ['a', 'b', 'c', 'd']
    print(f"   ✓ Expected: {expected4}")
    assert output4 == expected4, f"Output does not match expected: {output4} != {expected4}"

    # Example 5: Composite key join
    print("\n5. Composite key (K1,K2,A) join (K1,K2,B):")
    er11 = EmbeddedRelation(name="R1", content_attrs=[Var(name="K1"), Var(name="K2"), Var(name="A")])
    er12 = EmbeddedRelation(name="R2", content_attrs=[Var(name="K1"), Var(name="K2"), Var(name="B")])
    rhs5 = RHS(ers=[er11, er12], rel_ops=[","])
    output5 = [v.name for v in rhs5.output_content_attrs]
    print(f"   Input: R1(K1,K2,A), R2(K1,K2,B)")
    print(f"   Output: {output5}")
    expected5 = ['K1', 'K2', 'A', 'B']
    print(f"   ✓ Expected: {expected5} (keys collapsed)")
    assert output5 == expected5, f"Output does not match expected: {output5} != {expected5}"

    print("\n" + "=" * 60)
    print("All examples demonstrate that output_content_attrs:")
    print("  • Uses aliased names from join_conditions")
    print("  • Collapses duplicate join keys (only adds each key once)")
    print("  • Adds non-key columns from all input relations")
    print("=" * 60)
