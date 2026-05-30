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
# # Comparison
#
# > Session-level comparison harness for RelNN vs PyG models

# %%

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from relann.session import Session

logger = logging.getLogger(__name__)

# %%
@dataclass
class TrainResult:
    losses: List[float]
    wall_time_s: float
    final_loss: float

@dataclass
class EvalResult:
    accuracy: float
    n_correct: int
    n_total: int

@dataclass
class ForwardResult:
    max_diff: float
    mean_diff: float
    passed: bool
    shape: Tuple[int, ...]

@dataclass
class SyncResult:
    n_synced: int
    n_skipped: int
    warnings: List[str] = field(default_factory=list)

# %%
class SessionComparison:
    """Compare a Session-level RelNN model with a PyG nn.Module on a dataset."""

    def __init__(self, name: str, verbose: bool = True):
        self.name = name
        self.verbose = verbose
        self._pyg_model: Optional[nn.Module] = None
        self._session: Optional[Session] = None
        self._mapping: Dict[str, str] = {}
        self._pyg_train_result: Optional[TrainResult] = None
        self._pyg_eval_result: Optional[EvalResult] = None
        self._relnn_train_result: Optional[TrainResult] = None
        self._relnn_eval_result: Optional[EvalResult] = None

    # -- Setup ---------------------------------------------------------------

    def set_pyg_model(self, model: nn.Module) -> None:
        self._pyg_model = model

    def set_relnn_session(self, session: Session) -> None:
        self._session = session

    # -- Parameter inspection ------------------------------------------------

    def _pyg_params(self) -> Dict[str, torch.Tensor]:
        assert self._pyg_model is not None, "Call set_pyg_model first"
        return dict(self._pyg_model.named_parameters())

    def _relnn_params(self) -> Dict[str, torch.Tensor]:
        assert self._session is not None, "Call set_relnn_session first"
        return dict(self._session.engine.parameter_store)

    def print_params(self) -> None:
        """Print both parameter stores side by side."""
        pyg_p = self._pyg_params()
        rn_p = self._relnn_params()

        header = f"=== {self.name}: Parameter Overview ==="
        print(header)
        print("-" * len(header))

        fmt = "  {name:<50s} {shape:>15s} {numel:>8s}"
        print(fmt.format(name="Name", shape="Shape", numel="#Params"))
        print(fmt.format(name="----", shape="-----", numel="-------"))

        print(f"\n  [PyG] ({sum(p.numel() for p in pyg_p.values())} total)")
        for name, p in pyg_p.items():
            print(fmt.format(name=name, shape=str(tuple(p.shape)), numel=str(p.numel())))

        print(f"\n  [RelNN] ({sum(p.numel() for p in rn_p.values())} total)")
        for fqn, p in rn_p.items():
            print(fmt.format(name=fqn, shape=str(tuple(p.shape)), numel=str(p.numel())))
        print()

    def assert_param_count(self) -> None:
        """Assert total trainable parameter count is identical."""
        pyg_total = sum(p.numel() for p in self._pyg_params().values())
        rn_total = sum(p.numel() for p in self._relnn_params().values())
        if self.verbose:
            print(f"  Param count -- PyG: {pyg_total}, RelNN: {rn_total}", end="")
        if pyg_total == rn_total:
            if self.verbose:
                print("  [MATCH]")
        else:
            if self.verbose:
                print(f"  [MISMATCH delta={abs(pyg_total - rn_total)}]")
            raise AssertionError(
                f"Param count mismatch: PyG={pyg_total}, RelNN={rn_total}"
            )

    # -- Weight mapping ------------------------------------------------------

    def set_mapping(self, mapping: Dict[str, str]) -> None:
        """Set explicit {pyg_param_name: relnn_fqn} mapping."""
        self._mapping = dict(mapping)

    def print_mapping(self) -> None:
        """Print the mapping with shapes for verification."""
        if not self._mapping:
            print("  (no mapping set)")
            return
        pyg_p = self._pyg_params()
        rn_p = self._relnn_params()

        print(f"  Weight mapping ({len(self._mapping)} entries):")
        fmt = "    {pyg:<45s} -> {rn:<45s}  {shapes}"
        for pyg_name, rn_fqn in self._mapping.items():
            pyg_shape = str(tuple(pyg_p[pyg_name].shape)) if pyg_name in pyg_p else "???"
            rn_shape = str(tuple(rn_p[rn_fqn].shape)) if rn_fqn in rn_p else "???"
            ok = "[OK]" if pyg_shape == rn_shape and pyg_shape != "???" else "[WARN]"
            print(fmt.format(pyg=pyg_name, rn=rn_fqn, shapes=f"{pyg_shape} -> {rn_shape} {ok}"))

    def sync_weights(self, direction: str = "pyg_to_relnn") -> SyncResult:
        """Copy weights between models based on the set mapping.

        direction: 'pyg_to_relnn' (default) or 'relnn_to_pyg'.
        """
        if not self._mapping:
            raise ValueError("No mapping set. Call set_mapping() first.")

        if direction == "pyg_to_relnn":
            src_params = self._pyg_params()
            dst_store = self._session.engine.parameter_store
        elif direction == "relnn_to_pyg":
            src_params = self._relnn_params()
            dst_store = self._pyg_params()
        else:
            raise ValueError(f"Unknown direction: {direction}")

        warnings = []
        synced = 0
        skipped = 0

        for pyg_name, rn_fqn in self._mapping.items():
            src_key = pyg_name if direction == "pyg_to_relnn" else rn_fqn
            dst_key = rn_fqn if direction == "pyg_to_relnn" else pyg_name

            if src_key not in src_params:
                warnings.append(f"Source key not found: {src_key}")
                skipped += 1
                continue

            if direction == "pyg_to_relnn":
                if dst_key not in dst_store:
                    warnings.append(f"RelNN FQN not found: {dst_key}")
                    skipped += 1
                    continue
                src_data = src_params[src_key]
                dst_data = dst_store[dst_key]
            else:
                dst_data = dst_store.get(dst_key)
                if dst_data is None:
                    warnings.append(f"PyG param not found: {dst_key}")
                    skipped += 1
                    continue
                src_data = src_params[src_key]

            if src_data.shape != dst_data.shape:
                warnings.append(
                    f"Shape mismatch: {src_key} {tuple(src_data.shape)} vs {dst_key} {tuple(dst_data.shape)}"
                )
                skipped += 1
                continue

            dst_data.data.copy_(src_data.data)
            synced += 1

        result = SyncResult(n_synced=synced, n_skipped=skipped, warnings=warnings)
        if self.verbose:
            print(f"  Sync ({direction}): {synced} synced, {skipped} skipped")
            for w in warnings:
                print(f"    [WARN] {w}")
        return result

    # -- Independent runs ----------------------------------------------------

    def train_pyg(self, train_fn: Callable, epochs: int = 100) -> TrainResult:
        """Train PyG model. train_fn(model, epoch) -> loss (float)."""
        assert self._pyg_model is not None
        losses = []
        t0 = time.perf_counter()
        for epoch in range(1, epochs + 1):
            loss = train_fn(self._pyg_model, epoch)
            losses.append(float(loss))
        wall = time.perf_counter() - t0
        result = TrainResult(losses=losses, wall_time_s=wall, final_loss=losses[-1])
        self._pyg_train_result = result
        if self.verbose:
            print(f"  PyG train: {epochs} epochs, final_loss={result.final_loss:.4f}, time={wall:.1f}s")
        return result

    def eval_pyg(self, eval_fn: Callable) -> EvalResult:
        """Evaluate PyG model. eval_fn(model) -> (n_correct, n_total)."""
        assert self._pyg_model is not None
        n_correct, n_total = eval_fn(self._pyg_model)
        result = EvalResult(accuracy=n_correct / n_total, n_correct=n_correct, n_total=n_total)
        self._pyg_eval_result = result
        if self.verbose:
            print(f"  PyG eval: {result.accuracy:.1%} ({n_correct}/{n_total})")
        return result

    def train_relnn(self, fit_dsl: str) -> TrainResult:
        """Train RelNN via session.run(fit_dsl).

        Loss history is extracted from engine.trained_modules (populated by
        Engine.fit after training completes).
        """
        assert self._session is not None
        t0 = time.perf_counter()
        self._session.run(fit_dsl)
        wall = time.perf_counter() - t0

        losses: List[float] = []
        for info in self._session.engine.trained_modules.values():
            history = info.get("loss_history", [])
            if history:
                losses = list(history)
                break

        final_loss = losses[-1] if losses else float("nan")
        result = TrainResult(losses=losses, wall_time_s=wall, final_loss=final_loss)
        self._relnn_train_result = result
        if self.verbose:
            print(f"  RelNN train: final_loss={result.final_loss:.4f}, time={wall:.1f}s")
        return result

    def eval_relnn(self, pred_dsl: str, eval_fn: Callable) -> EvalResult:
        """Predict via session.run(pred_dsl), then eval_fn(pred_result) -> (n_correct, n_total)."""
        assert self._session is not None
        pred = self._session.run(pred_dsl)
        n_correct, n_total = eval_fn(pred)
        result = EvalResult(accuracy=n_correct / n_total, n_correct=n_correct, n_total=n_total)
        self._relnn_eval_result = result
        if self.verbose:
            print(f"  RelNN eval: {result.accuracy:.1%} ({n_correct}/{n_total})")
        return result

    # -- Comparison ----------------------------------------------------------

    def compare_forward(
        self,
        pyg_fn: Callable,
        relnn_pred_dsl: str,
        align_fn: Optional[Callable] = None,
        tolerance: float = 1e-5,
    ) -> ForwardResult:
        """Weight-synced forward comparison.

        1. sync_weights() must be called first.
        2. pyg_fn(model) -> Tensor of shape (N, d)
        3. session.run(relnn_pred_dsl) -> EmbeddedRelation
        4. align_fn(relnn_result, pyg_output) -> Tensor aligned to pyg order
           If None, assumes row order matches.
        """
        assert self._pyg_model is not None
        assert self._session is not None

        self._pyg_model.eval()
        with torch.no_grad():
            pyg_out = pyg_fn(self._pyg_model)

        relnn_pred = self._session.run(relnn_pred_dsl)
        assert relnn_pred is not None and relnn_pred.embeddings is not None

        rn_out = relnn_pred.embeddings[0]
        if align_fn is not None:
            rn_out = align_fn(relnn_pred, pyg_out)

        max_diff = (pyg_out - rn_out).abs().max().item()
        mean_diff = (pyg_out - rn_out).abs().mean().item()
        passed = max_diff < tolerance

        result = ForwardResult(
            max_diff=max_diff,
            mean_diff=mean_diff,
            passed=passed,
            shape=tuple(pyg_out.shape),
        )

        if self.verbose:
            status = "[OK]" if passed else "[FAIL]"
            print(f"  Forward compare (shape {result.shape}): "
                  f"max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}  {status}")

        return result

    def compare_training(self) -> Dict[str, Any]:
        """Compare training results from train_pyg and train_relnn.

        Returns a dict with keys: pyg_final_loss, relnn_final_loss, loss_delta,
        pyg_time, relnn_time, speedup.
        """
        pyg_tr = self._pyg_train_result
        rn_tr = self._relnn_train_result
        if pyg_tr is None or rn_tr is None:
            raise ValueError("Run train_pyg and train_relnn first.")

        info = {
            "pyg_final_loss": pyg_tr.final_loss,
            "relnn_final_loss": rn_tr.final_loss,
            "loss_delta": abs(pyg_tr.final_loss - rn_tr.final_loss),
            "pyg_time": pyg_tr.wall_time_s,
            "relnn_time": rn_tr.wall_time_s,
            "speedup": rn_tr.wall_time_s / pyg_tr.wall_time_s if pyg_tr.wall_time_s > 0 else float("inf"),
        }

        if self.verbose:
            print(f"  Training comparison:")
            print(f"    Final loss -- PyG: {info['pyg_final_loss']:.4f}, RelNN: {info['relnn_final_loss']:.4f}  (delta={info['loss_delta']:.4f})")
            print(f"    Wall time  -- PyG: {info['pyg_time']:.1f}s, RelNN: {info['relnn_time']:.1f}s  (ratio={info['speedup']:.2f}x)")

        return info

    def summary(self) -> str:
        """Print and return a final summary table."""
        lines = [
            "",
            f"=== {self.name}: Summary ===",
            "-" * 50,
        ]

        pyg_total = sum(p.numel() for p in self._pyg_params().values())
        rn_total = sum(p.numel() for p in self._relnn_params().values())
        param_ok = "MATCH" if pyg_total == rn_total else "MISMATCH"
        lines.append(f"  Params: PyG={pyg_total}, RelNN={rn_total}  [{param_ok}]")

        if self._pyg_train_result and self._relnn_train_result:
            lines.append(f"  Training loss: PyG={self._pyg_train_result.final_loss:.4f}, "
                         f"RelNN={self._relnn_train_result.final_loss:.4f}")
            lines.append(f"  Training time: PyG={self._pyg_train_result.wall_time_s:.1f}s, "
                         f"RelNN={self._relnn_train_result.wall_time_s:.1f}s")

        if self._pyg_eval_result and self._relnn_eval_result:
            acc_delta = abs(self._pyg_eval_result.accuracy - self._relnn_eval_result.accuracy)
            lines.append(f"  Accuracy: PyG={self._pyg_eval_result.accuracy:.1%}, "
                         f"RelNN={self._relnn_eval_result.accuracy:.1%}, Delta={acc_delta:.1%}")

        lines.append("-" * 50)
        text = "\n".join(lines)
        print(text)
        return text
