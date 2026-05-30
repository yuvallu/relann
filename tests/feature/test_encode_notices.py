"""Tests for one-shot INFO notices on implicit auto-tensorize (numeric / categorical)."""

import logging
import sys
from pathlib import Path

import pandas as pd
from relann.embedded_relation import EmbeddedRelation
from relann.tensor_term_compiler import _ColumnExtractModule

def test_auto_tensorize_notice_emitted_once_for_numeric(caplog):
    caplog.set_level(logging.INFO)
    df = pd.DataFrame({"age": [1.0, 2.0]})
    er = EmbeddedRelation(content_schema=["age"], embedding_shapes=[], content=df)
    m = _ColumnExtractModule("age")
    m._source_er = er
    _ = m()
    _ = m()
    msgs = [r.message for r in caplog.records if r.levelname == "INFO" and "auto-tensorized" in r.message]
    assert len(msgs) == 1
    assert "age" in msgs[0]

def test_auto_tensorize_notice_emitted_once_for_categorical(caplog):
    caplog.set_level(logging.INFO)
    df = pd.DataFrame({"dept": pd.Categorical(["a", "b", "a"])})
    er = EmbeddedRelation(content_schema=["dept"], embedding_shapes=[], content=df)
    m = _ColumnExtractModule("dept")
    m._source_er = er
    _ = m()
    _ = m()
    msgs = [r.message for r in caplog.records if r.levelname == "INFO" and "auto-tensorized" in r.message]
    assert len(msgs) == 1
    assert "dept" in msgs[0]
