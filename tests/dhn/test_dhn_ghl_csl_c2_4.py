"""Smoke tests for canonical GHL RelNN (dhn_ghl_csl_c2_4.relnn) with edge-join."""

import sys
from pathlib import Path

import pytest
import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_RELNN_DIR := _REPO / "research" / "paper_experiments" / "dhn") not in sys.path:
    sys.path.insert(0, str(_RELNN_DIR))

sys.path.insert(0, str(_REPO / "tests" / "dhn"))

from dhn_utils import build_dhn_db_edge, load_csl_dataset  # noqa: E402
from relann.session import Session  # noqa: E402
from relann.torch_utils import full_seed  # noqa: E402

_RELNN_FILE = _RELNN_DIR / "dhn_ghl_csl_c2_4.relnn"


def _split_relnn_file(path: Path) -> tuple[str, str, str]:
    """Split .relnn file into [define, fit, pred] blocks."""
    text = path.read_text(encoding="utf-8")
    parts = text.split("#lang:relnn")
    blocks: list[str] = []
    for chunk in parts[1:]:
        chunk = chunk.strip()
        if chunk:
            blocks.append("#lang:relnn\n" + chunk)
    if len(blocks) != 3:
        raise ValueError(
            f"Expected 3 #lang:relnn blocks in {path}, found {len(blocks)}"
        )
    return blocks[0], blocks[1], blocks[2]


def test_split_relnn_file_three_blocks():
    define_p, fit_p, pred_p = _split_relnn_file(_RELNN_FILE)
    assert "#lang:relnn" in define_p
    assert "?fit" in fit_p
    assert "?pred" in pred_p


def test_csl_ghl_c2_4_forward_small_subset():
    if not _RELNN_FILE.is_file():
        pytest.skip(f"missing {_RELNN_FILE}")

    graphs, labels, _nc, _nf = load_csl_dataset(root="./data")
    graphs = graphs[:4]
    labels = labels[:4]

    db = build_dhn_db_edge(graphs, labels=labels, graph_ids=list(range(4)))

    define_prog, fit_prog, pred_prog = _split_relnn_file(_RELNN_FILE)
    fit_prog = fit_prog.replace("<epochs=500,", "<epochs=3,")

    full_seed(0)
    session = Session(db=db)
    session.run(define_prog)
    session.run(fit_prog)
    result = session.run(pred_prog)

    # ?pred returns ArgMax class index per graph, shape (num_graphs, 1)
    preds = result.embeddings[0].view(-1)
    assert preds.shape == (4,)
    assert int(preds.max()) < 10 and int(preds.min()) >= 0
