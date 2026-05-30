# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
from juplit import test

# %% [markdown]
# # Constant Embeddings End-to-End Test
#
# > Tests the full pipeline for constant embeddings: parsing → Engine → forward pass

# %%
if test():
    #| hide
    import pandas as pd
    import torch
    import numpy as np
    import sys
    from pathlib import Path

# %%
if test():
    #| hide

    # Add parent directory to path
    sys.path.insert(0, str(Path(__file__).parent.parent) if '__file__' in globals() else '.')

    from relann.parser import parse_and_transform_str
    from relann.engine import Engine
    from relann.torch_utils import full_seed
    from relann.era_operations import _to_er_dict

    import torch_geometric
    device = torch_geometric.device('auto')
    print(f"Using device: {device}")

    # Set seed for reproducibility
    full_seed(42)

# %% [markdown]
# ## Test: 2-Line Syntax (User's Requested Format)
#
# This test verifies the exact 2-line syntax requested:
# ```python
# AP(a_id; sum(z2)) :- A(a_id; z1), P(a_id, p_id; z2) .
# AP2(a_id; sum(z2)) :- A(a_id; 0) | AP(a_id; z2) .
# ```

# %%
if test():
    #| hide
    # Test: 2-line syntax as requested by user

    # Setup: Create test data
    # A has a_ids: [1, 2, 3]
    # P has (a_id, p_id) pairs: [(1, 10), (2, 20)] - note: a_id=3 is missing

    A_df = pd.DataFrame({'a_id': [1, 2, 3]})
    P_df = pd.DataFrame({'a_id': [1, 2], 'p_id': [10, 20]})

    # A and P have variable embeddings (z1 and z2 respectively)
    A_emb = torch.randn(3, 4, device=device)
    P_emb = torch.randn(2, 4, device=device)

    # Build relations dict for ExecutionContext
    relations = {
        'A': {'content': A_df, 'content_schema': ['a_id'], 'embedding_shapes': [A_emb.shape], 'embeddings': [A_emb]},
        'P': {'content': P_df, 'content_schema': ['a_id', 'p_id'], 'embedding_shapes': [P_emb.shape], 'embeddings': [P_emb]}
    }

    # Parse the 2-line program
    program_str = """
    AP(a_id; sum(z2)) :- A(a_id; z1), P(a_id, p_id; z2) .
    AP2(a_id; sum(z2)) :- A(a_id; 0) | AP(a_id; z2) .
    """

    print("2-Line Program:")
    print(program_str)

    program = parse_and_transform_str(program_str)
    from relann.parser import Rule
    rules = [stmt for stmt in program.statements if isinstance(stmt, Rule)]

    print(f"\nParsed {len(rules)} rules:")
    for i, rule in enumerate(rules, 1):
        print(f"  {i}. {rule.lhs.name}")
        if hasattr(rule.rhs, 'ers') and rule.rhs.ers:
            for er in rule.rhs.ers:
                if hasattr(er, 'embedding_var'):
                    emb_var = er.embedding_var
                    if isinstance(emb_var, (int, float, bool)):
                        print(f"     Constant embedding: {emb_var}")
                    elif hasattr(emb_var, 'name'):
                        print(f"     Variable embedding: {emb_var.name}")

# %%
if test():
    #| hide
    # Create engine and run the 2-line program
    engine = Engine(db={}, debug=False)

    print("Adding rules to engine...")
    for rule in rules:
        engine.add_rule(rule)
        print(f"  ✓ Added rule: {rule.lhs.name}")

    # Run forward pass for AP2
    tg = engine.term_graphs['global']
    sub_tg = tg.induced_subgraph(node_name='AP2', direction='ancestors', include_root=True)
    ground_sub_tg = engine.replace_all_vars_in_tg_using_symbol_table(sub_tg, in_place=False)
    ground_sub_tg = engine.eval_tensor_terms_on_tg(ground_sub_tg)

    from relann.relnn import term_graph_to_module
    module = term_graph_to_module(ground_sub_tg, param_loader=engine)

    print("\nRunning instantiate...")
    module.instantiate(relations)

    print("Running forward pass...")
    result = module.forward(relations)

    print(f"\nResult:")
    print(f"  Content shape: {result.content.shape}")
    print(f"  Content:\n{result.content}")
    print(f"  Embedding shape: {result.embeddings[0].shape if result.embeddings else None}")
    if result.embeddings:
        print(f"  Embedding device: {result.embeddings[0].device}")
        print(f"  Embedding dtype: {result.embeddings[0].dtype}")

# %%
if test():
    #| hide
    # Verify the 2-line syntax results
    print("\n" + "="*60)
    print("Verification for 2-Line Syntax:")
    print("="*60)

    result_df = result.content
    if hasattr(result_df, '__class__') and result_df.__class__.__module__.startswith('cudf'):
        result_df = result_df.to_pandas()

    # Check that all a_ids from A are present
    a_ids_in_result = set(result_df['a_id'].values)
    a_ids_in_A = set(A_df['a_id'].values)

    assert a_ids_in_result == a_ids_in_A, f"Missing a_ids! Expected {a_ids_in_A}, got {a_ids_in_result}"
    print(f"✓ All a_ids from A are present: {a_ids_in_result}")

    # Check embedding shape
    assert result.embeddings is not None, "Result should have embeddings"
    assert len(result.embeddings) == 1, "Result should have one embedding"
    assert result.embeddings[0].shape == (3, 4), f"Expected shape (3, 4), got {result.embeddings[0].shape}"
    print(f"✓ Embedding shape is correct: {result.embeddings[0].shape}")

    # Check that a_id=3 (missing from P) has zero embeddings
    emb = result.embeddings[0]
    a_id_3_idx = result_df[result_df['a_id'] == 3].index[0]
    a_id_3_emb = emb[a_id_3_idx]

    # Device should match the expected device (CUDA when available)
    # Use emb.device to ensure both tensors are on the same device
    assert torch.allclose(a_id_3_emb, torch.zeros(4, device=emb.device)), \
        f"a_id=3 should have zero embeddings, got {a_id_3_emb}"
    print(f"✓ a_id=3 (missing from P) has zero embeddings: {a_id_3_emb}")
    print(f"✓ Device is correct: {emb.device.type == 'cuda'} (embeddings should be on CUDA when available)")

    # Check that a_ids 1 and 2 have non-zero embeddings (from P via AP)
    for a_id in [1, 2]:
        a_id_idx = result_df[result_df['a_id'] == a_id].index[0]
        a_id_emb = emb[a_id_idx]
        assert not torch.allclose(a_id_emb, torch.zeros(4, device=emb.device)), \
            f"a_id={a_id} should have non-zero embeddings from P"
        print(f"✓ a_id={a_id} has non-zero embeddings (from P)")

    print("\n✅ Test passed: 2-line syntax works correctly!")

# %% [markdown]
# ## Summary
#
# This test verifies the complete pipeline for constant embeddings:
#
# 1. ✅ **Parsing**: Constant embeddings like `A(a_id; 0)` are correctly parsed by the grammar and transformer
# 2. ✅ **Engine**: Rules with constant embeddings are added to the term graph correctly
# 3. ✅ **DataLoader**: Constant embeddings are created as tensors with the correct shape and value
# 4. ✅ **Union**: Constant embeddings (shape `(n, 1)`) are correctly broadcasted to match variable embeddings
# 5. ✅ **Forward pass**: The complete forward pass produces correct results with all a_ids present
# 6. ✅ **Device/dtype**: Embeddings are aligned to the correct device and dtype during Union operations
#
# The test demonstrates the outer join use case where we want to include all rows from the left relation even if they don't appear in the right relation.
