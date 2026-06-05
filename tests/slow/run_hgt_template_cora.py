"""Templated HGT-style architecture on Cora: validates all 3 template levels.

Level 1: Per-head TransformDef templates (K<head>, Q<head>, M<head>, Mu<head>)
Level 2: Per-head computation via templated Rules (ATT_Head<head>, MSG_Head<head>)
Level 3: Layer stacking via templated FunctionDef (HGTLayer<layer>)

Uses a simplified attention mechanism (element-wise instead of matmul-based)
to avoid shape issues with the current engine. The template nesting behaviour
is identical to what a full HGT would exercise.

Run from repo root:
    python tests/slow/run_hgt_template_cora.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from relann.session import Session
from relann.torch_utils import full_seed

full_seed(42)

# Cora is auto-downloaded by `load_cora_dataset` on first call via PyG's
# Planetoid -- no need for a pre-existence skip. Removed because the prior
# `if not path.exists(): sys.exit(0)` made the CI step silently pass when
# Cora wasn't present, defeating the e2e guard for the V2 optimizer's
# homogeneous-HGT regression.
from relann.datasets import load_cora_dataset

data = load_cora_dataset()
db = {k: data[k] for k in ("Papers", "Citation", "Labels")}

papers_df, papers_z = db["Papers"]
cite_df, cite_w = db["Citation"]

d = 6
h = 3
in_features = 1433
n_classes = 7

# ── Single-layer templated HGT (Levels 1 + 2) ─────────────────────────────

print("=" * 60)
print("Test 1: Single-layer templated HGT-style (Levels 1 + 2)")
print("=" * 60)

session1 = Session(db=db)

session1.run(f"""
#lang:relnn
d = {d} .
h = {h} .

K<i> = Linear(d, d/h) .
Q<i> = Linear(d, d/h) .
M<i> = Linear(d, d/h) .
Mu<i> = Tensor(1) .
A_LIN = Linear(d, d) .
Classifier = Linear(d, {n_classes}) .

Papers_Emb(s; Linear({in_features}, d, False)(z)) :- Papers(s;z) .

ATT_Head<i>(s,t; K<i>(z1) * Q<i>(z2) * Mu<i>) :-
    Papers_Emb(s;z1), Citation(s, t; w), Papers_Emb(t;z2) .

ATT_Con(s,t; Concat(z1, z2, z3)) :- ATT_Head<1>(s,t;z1), ATT_Head<2>(s,t;z2), ATT_Head<3>(s,t;z3) .

MSG_Src(s,t; z) :- Papers_Emb(s;z), Citation(s, t; w) .
MSG_Head<i>(s,t; M<i>(z)) :- MSG_Src(s,t;z) .
MSG(s,t; Concat(z1, z2, z3)) :- MSG_Head<1>(s,t;z1), MSG_Head<2>(s,t;z2), MSG_Head<3>(s,t;z3) .

AGG_MSG(t; sum(z2 * z1)) :- MSG(s,t; z2), ATT_Con(s,t; z1) .
Output(t; A_LIN(ReLU(z1)) + z2) :- AGG_MSG(t; z1), Papers_Emb(t; z2) .
""")

session1.run(f"""
#lang:relnn
?fit <epochs=10, lr=0.01, weight_decay=0.0005>
Loss(; CrossEntropyLoss()(Classifier(z_pred), z)) :- Output(target_id; z_pred), Labels(target_id; z) .
""")

result1 = session1.run("""
#lang:relnn
?pred Predictions(t; z) :- Output(t; z) .
""")

assert result1 is not None and result1.embeddings is not None
print(f"  Output shape: {tuple(result1.embeddings[0].shape)}")
assert result1.embeddings[0].shape[1] == d, f"Expected d={d}, got {result1.embeddings[0].shape[1]}"

cache1 = session1.engine._template_instance_cache
print(f"  Template cache keys: {sorted(cache1.keys())}")
for head_idx in [1, 2, 3]:
    assert f"K<{head_idx}>" in cache1, f"Missing K<{head_idx}>"
    assert f"Q<{head_idx}>" in cache1, f"Missing Q<{head_idx}>"
    assert f"M<{head_idx}>" in cache1, f"Missing M<{head_idx}>"
    assert f"Mu<{head_idx}>" in cache1, f"Missing Mu<{head_idx}>"

pred1 = session1.run("""
#lang:relnn
?pred Cls(t; ArgMax()(Classifier(z))) :- Output(t; z) .
""")
from relann.datasets import evaluate_node_classification
acc1 = evaluate_node_classification(data, pred1, return_value=True)
print(f"  Accuracy: {acc1:.1%}")
print("  PASS: Single-layer templated HGT-style")

# ── Two-layer templated HGT (Level 3: FunctionDef template) ────────────────

print()
print("=" * 60)
print("Test 2: Two-layer templated HGT-style (Level 3 — FunctionDef)")
print("=" * 60)

session2 = Session(db=db)

session2.run(f"""
#lang:relnn
d = {d} .
h = {h} .

K<layer, head> = Linear(d, d/h) .
Q<layer, head> = Linear(d, d/h) .
M<layer, head> = Linear(d, d/h) .
Mu<layer, head> = Tensor(1) .
A_LIN<layer> = Linear(d, d) .
Classifier = Linear(d, {n_classes}) .

Papers_Emb(s; Linear({in_features}, d, False)(z)) :- Papers(s;z) .

def HGTLayer<layer>(InputNodes, Edges):
    Head<head>(s,t; K<layer,head>(z1) * Q<layer,head>(z2) * Mu<layer,head>) :- InputNodes(s;z1), Edges(s, t; w), InputNodes(t;z2) .
    Head_Con(s,t; Concat(z1, z2, z3)) :- Head<1>(s,t;z1), Head<2>(s,t;z2), Head<3>(s,t;z3) .
    Head_Agg(t; sum(z)) :- Head_Con(s,t; z) .

    MSG_Src(s,t; z) :- InputNodes(s;z), Edges(s, t; w) .
    MSG_Head<head>(s,t; M<layer,head>(z)) :- MSG_Src(s,t;z) .
    MSG(s,t; Concat(z1, z2, z3)) :- MSG_Head<1>(s,t;z1), MSG_Head<2>(s,t;z2), MSG_Head<3>(s,t;z3) .
    AGG_MSG(t; sum(z)) :- MSG(s,t; z) .

    Out(t; A_LIN<layer>(ReLU(z1 + z3)) + z2) :- AGG_MSG(t; z1), InputNodes(t; z2), Head_Agg(t; z3) .
enddef

L1(t; z) :- HGTLayer<1>(Papers_Emb, Citation)(t; z) .
Output(t; z) :- HGTLayer<2>(L1, Citation)(t; z) .
""")

session2.run(f"""
#lang:relnn
?fit <epochs=10, lr=0.01, weight_decay=0.0005>
Loss(; CrossEntropyLoss()(Classifier(z_pred), z)) :- Output(target_id; z_pred), Labels(target_id; z) .
""")

result2 = session2.run("""
#lang:relnn
?pred Predictions(t; z) :- Output(t; z) .
""")

assert result2 is not None and result2.embeddings is not None
print(f"  Output shape: {tuple(result2.embeddings[0].shape)}")
assert result2.embeddings[0].shape[1] == d, f"Expected d={d}, got {result2.embeddings[0].shape[1]}"

cache2 = session2.engine._template_instance_cache
print(f"  Template cache keys: {sorted(cache2.keys())}")
for layer_idx in [1, 2]:
    for head_idx in [1, 2, 3]:
        assert f"K<{layer_idx},{head_idx}>" in cache2, f"Missing K<{layer_idx},{head_idx}>"
    assert f"A_LIN<{layer_idx}>" in cache2, f"Missing A_LIN<{layer_idx}>"
assert "HGTLayer<1>" in cache2
assert "HGTLayer<2>" in cache2

pred2 = session2.run("""
#lang:relnn
?pred Cls(t; ArgMax()(Classifier(z))) :- Output(t; z) .
""")
acc2 = evaluate_node_classification(data, pred2, return_value=True)
print(f"  Accuracy: {acc2:.1%}")
print("  PASS: Two-layer templated HGT-style")

print()
print("OK: All templated HGT tests passed.")
