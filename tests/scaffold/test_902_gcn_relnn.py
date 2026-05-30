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

# %%
if test():
    import pandas as pd
    import torch
    import torch.nn.functional as F
    import numpy as np
    import os.path as osp
    from pathlib import Path
    import sys

    # Add parent directory to path if needed
    sys.path.insert(0, str(Path(__file__).parent.parent) if '__file__' in globals() else '.')

    from relann.parser import parse_and_transform_str
    from relann.engine import Engine, pretty_print_params
    from relann.torch_utils import full_seed, get_project_root, get_model_weights
    from relann.pydantic_classes import FitStatement, PredictStatement

    import torch_geometric
    import torch_geometric.transforms as T
    from torch_geometric.datasets import Planetoid
    from torch_geometric.nn import GCNConv
    from torch_geometric.utils import add_self_loops, degree

# %% [markdown]
# # GCN Test Flow: RelNN

# %%
if test():
    # Set random seed for reproducibility
    full_seed(42)

    # Configuration
    args = {
        'dataset': 'Cora',
        'hidden_channels': 16,
        'lr': 0.01,
        'epochs': 200,
        'weight_decay': 5e-4
    }

    device = torch_geometric.device('auto')
    print(f"Using device: {device}")

    # Load Cora dataset
    project_root = get_project_root()
    path = project_root / 'data' / 'Planetoid'
    dataset = Planetoid(str(path), args['dataset'], transform=T.NormalizeFeatures())
    data = dataset[0].to(device)

    print(f"Dataset: {args['dataset']}")
    print(f"Number of nodes: {data.x.size(0)}")
    print(f"Number of features: {data.x.size(1)}")
    print(f"Number of classes: {dataset.num_classes}")
    print(f"Number of edges: {data.edge_index.size(1)}")

# %%
if test():
    # Create nodes relation: DataFrame with node_id, label, masks + node features tensor
    num_nodes = data.x.size(0)

    node_df = pd.DataFrame({
        'node_id': range(num_nodes),
        'label': data.y.cpu().numpy(),
        'is_train': data.train_mask.cpu().numpy(),
        'is_val': data.val_mask.cpu().numpy(),
        'is_test': data.test_mask.cpu().numpy()
    })
    node_features_tensor = data.x  # Shape: [num_nodes, num_features]

    # Create separate train and test node tables
    train_nodes = node_df[node_df["is_train"] == True].copy()
    test_nodes = node_df[node_df["is_test"] == True].copy()

    # Create train nodes relation: DataFrame with node_id + node features tensor (only train nodes)
    nodes_train_df = train_nodes[["node_id"]].copy()
    # Extract features for train nodes only
    train_indices = torch.as_tensor(train_nodes["node_id"].values, dtype=torch.long, device=node_features_tensor.device)
    nodes_train_tensor = node_features_tensor[train_indices]  # Shape: [num_train_nodes, num_features]

    # Create test nodes relation: DataFrame with node_id + node features tensor (only test nodes)
    nodes_test_df = test_nodes[["node_id"]].copy()
    # Extract features for test nodes only
    test_indices = torch.as_tensor(test_nodes["node_id"].values, dtype=torch.long, device=node_features_tensor.device)
    nodes_test_tensor = node_features_tensor[test_indices]  # Shape: [num_test_nodes, num_features]

    # Create nodes_all relation: DataFrame with node_id + node features tensor (ALL nodes)
    nodes_all_df = node_df[["node_id"]].copy()
    nodes_all_tensor = node_features_tensor  # Shape: [num_nodes, num_features] - all nodes

    # Create edges relation: DataFrame with source_id, target_id + edge normalization weights tensor
    # Add self-loops (required for GCN)
    edge_index, _ = add_self_loops(data.edge_index)

    edge_df = pd.DataFrame({
        'source_id': edge_index[0].cpu().numpy(),
        'target_id': edge_index[1].cpu().numpy()
    })

    # Compute normalization factors exactly like PyG's gcn_norm
    source, target = edge_index
    deg = degree(source, data.x.size(0))
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    norm_factor = deg_inv_sqrt[source] * deg_inv_sqrt[target]

    # Create edge features tensor with the normalized factors
    edge_features_tensor = norm_factor.unsqueeze(-1)  # Shape: [num_edges, 1]

    print(f"Nodes DataFrame shape: {node_df.shape}")
    print(f"Train nodes DataFrame shape: {nodes_train_df.shape}")
    print(f"Train nodes tensor shape: {nodes_train_tensor.shape}")
    print(f"Test nodes DataFrame shape: {nodes_test_df.shape}")
    print(f"Test nodes tensor shape: {nodes_test_tensor.shape}")
    print(f"Nodes all DataFrame shape: {nodes_all_df.shape}")
    print(f"Nodes all tensor shape: {nodes_all_tensor.shape}")
    print(f"Edges DataFrame shape: {edge_df.shape}")
    print(f"Edge features tensor shape: {edge_features_tensor.shape}")
    print(f"Normalization factor range: [{norm_factor.min():.4f}, {norm_factor.max():.4f}]")

    # Create Labels relation: extract labels from training nodes (class indices for CrossEntropyLoss)
    labels_df = train_nodes[["node_id"]].rename(columns={"node_id": "target_id"})
    num_classes = dataset.num_classes
    label_indices = torch.as_tensor(train_nodes["label"].values, dtype=torch.long, device=node_features_tensor.device)
    labels_tensor = label_indices.unsqueeze(1)  # Shape: [N, 1] - class indices

    print(f"\nLabels relation:")
    print(f"  Labels DataFrame shape: {labels_df.shape}")
    print(f"  Labels tensor shape: {labels_tensor.shape}")

# %%
if test():
    # Define the RelNN DSL programs for GCN - split into training and prediction

    # Training program:
    # 1. Alias rule: maps nodes to nodes_all for training phase (use ALL nodes like PyG)
    # 2. Layer 1: Linear transformation -> Message passing (join + multiply + aggregate) -> ReLU
    # 3. Layer 2: Linear transformation -> Message passing (join + multiply + aggregate)
    # 4. Loss computation for training (join predictions with labels)
    # 5. Fit statement for training

    train_program_str = f"""
    num_features = {dataset.num_features} .
    num_classes = {dataset.num_classes} .
    h = {args['hidden_channels']} .
    weight_decay = {args['weight_decay']} .
    lr = {args['lr']} .
    epochs = {args['epochs']} .

    # Alias rule: map nodes to nodes_all for training phase
    nodes(node_id; z) :- nodes_all(node_id; z) .

    # Layer 1: Initial node embedding
    NodesEmbedding1(node_id; Linear(num_features, h, False)(z)) :- nodes(node_id; z) .

    # Layer 1: Message passing - join nodes with edges, multiply embeddings by edge weights, aggregate by target_id
    MsgPassing1(target_id; sum(z * w)) :- NodesEmbedding1(node_id; z), edges(node_id, target_id; w) .

    # Layer 1: ReLU activation
    ReLULayer(node_id; ReLU(z)) :- MsgPassing1(node_id; z) .

    # Layer 2: Second node embedding
    NodesEmbedding2(node_id; Linear(h, num_classes, False)(z)) :- ReLULayer(node_id; z) .

    # Layer 2: Message passing - join nodes with edges, multiply embeddings by edge weights, aggregate by target_id
    MsgPassing2(target_id; sum(z * w)) :- NodesEmbedding2(node_id; z), edges(node_id, target_id; w) .

    # Training
    # Loss computation: use CrossEntropy transformation on predictions and labels
    ?fit <epochs=epochs, lr=lr, weight_decay=weight_decay> Loss(; CrossEntropyLoss()(z_pred, z)) :- MsgPassing2(target_id; z_pred), Labels(target_id; z) .
    """

    # Prediction program:
    # 1. Alias rule: maps nodes to nodes_all for prediction phase (use ALL nodes like PyG)
    # 2. Predict statement for inference

    predict_program_str = """
    # Prediction
    ?pred Output(target_id; z) :- MsgPassing2(target_id; z) .
    """

    print("Training Program:")
    print(train_program_str)
    print("\n" + "="*50 + "\n")
    print("Prediction Program:")
    print(predict_program_str)

# %%
if test():
    # Parse the training and prediction programs
    train_program = parse_and_transform_str(train_program_str)
    predict_program = parse_and_transform_str(predict_program_str)

    print("Training Program Statements:")
    for i, stmt in enumerate(train_program.statements):
        print(f"  Statement {i}: {type(stmt).__name__}")

    print(f"\nPrediction Program Statements:")
    for i, stmt in enumerate(predict_program.statements):
        print(f"  Statement {i}: {type(stmt).__name__}")

# %%
if test():
    # Create engine with database containing nodes_all, nodes_train, nodes_test, edges, and Labels
    engine = Engine(db={
        "nodes_all": (nodes_all_df, nodes_all_tensor),  # Add this - use all nodes
        "nodes_train": (nodes_train_df, nodes_train_tensor),
        "nodes_test": (nodes_test_df, nodes_test_tensor),
        "edges": (edge_df, edge_features_tensor),
        "Labels": (labels_df, labels_tensor)
    }, debug=True)

    # Execute training program
    print("Executing training program (this may take a while)...")
    engine.add_program(train_program)
    print("✓ Training completed")

    # Compute training accuracy before removing nodes_train
    print("\nComputing training accuracy...")
    train_predict_program_str = """
    # Prediction on training data (using nodes alias which points to nodes_train)
    ?pred TrainOutput(target_id; z) :- MsgPassing2(target_id; z) .
    """
    train_predict_program = parse_and_transform_str(train_predict_program_str)
    train_output = engine.add_program(train_predict_program)

    # Compute training accuracy
    if hasattr(train_output, 'embeddings') and train_output.embeddings:
        train_logits = train_output.embeddings[0]  # May include val/test neighbors
        train_pred = train_logits.argmax(dim=1).cpu().numpy()
        train_output_df = train_output.content.copy() if train_output.content is not None else __import__('pandas').DataFrame()
        if hasattr(train_output_df, '__class__') and train_output_df.__class__.__module__.startswith('cudf'):
            train_output_df = train_output_df.to_pandas()
        if 'target_id' not in train_output_df.columns and 'node_id' not in train_output_df.columns and len(train_output_df) > 0:
            train_output_df = train_output_df.copy(); train_output_df['node_id'] = list(train_output_df.index) if hasattr(train_output_df, 'index') else list(range(len(train_output_df)))
        key_col = 'target_id' if 'target_id' in train_output_df.columns else 'node_id'
        if key_col not in train_output_df.columns and len(train_output_df) > 0:
            train_output_df = train_output_df.copy(); train_output_df[key_col] = list(train_output_df.index) if hasattr(train_output_df, 'index') else list(range(len(train_output_df)))
        train_merged = train_output_df[[key_col]].merge(
            node_df[['node_id', 'label', 'is_train']],
            left_on=key_col,
            right_on='node_id',
            how='left'
        )
        # CRITICAL: Filter to only train nodes (MsgPassing2 includes all nodes that receive messages)
        train_mask = train_merged['is_train'].fillna(False).astype(bool)
        train_labels = train_merged.loc[train_mask, 'label'].values
        train_pred_filtered = train_pred[train_mask]
        
        if len(train_labels) > 0:
            train_acc = (train_pred_filtered == train_labels).mean()
            expected_train_count = node_df['is_train'].sum()
            print(f"✓ Training Accuracy: {train_acc:.4f} ({np.sum(train_pred_filtered == train_labels)}/{len(train_labels)} correct)")
            print(f"  (Filtered from {len(train_pred)} total predictions to {len(train_pred_filtered)} train nodes)")
            if len(train_pred_filtered) == expected_train_count:
                print(f"  ✓ Correct: Found all {expected_train_count} train nodes")
        else:
            train_acc = None
            print("⚠ No train nodes found in predictions")
    else:
        train_acc = None
        print("⚠ Could not compute training accuracy")

    # Execute prediction program (uses all nodes, will filter to test nodes for accuracy)
    print("\nExecuting prediction program (uses all nodes, will filter to test nodes for accuracy)...")
    output = engine.add_program(predict_program)
    print("✓ Prediction completed")

    print(f"\n✓ Program executed successfully")
    print(f"  Fit completed: {'Loss' in engine.trained_modules}")
    print(f"  Predict completed: output is {type(output).__name__}")

# %%
if test():
    engine.trained_modules['Loss']['module']

# %%
if test():
    from relann.term_graph import preety_draw_tg

    preety_draw_tg(engine.term_graphs['global'])

# %%
if test():
    output.embeddings, output.content

# %%
if test():
    # Display training results: loss history
    if 'Loss' in engine.trained_modules:
        loss_history = engine.trained_modules['Loss']['loss_history']
        print("Training Loss History:")
        print(f"  Epoch 1: {loss_history[0]:.6f}")
        if len(loss_history) > 1:
            print(f"  Epoch 10: {loss_history[9]:.6f}" if len(loss_history) > 9 else f"  Epoch {len(loss_history)}: {loss_history[-1]:.6f}")
            print(f"  Final loss: {loss_history[-1]:.6f}")
            print(f"  Loss change: {loss_history[0] - loss_history[-1]:.6f}")
        print(f"\n  Total epochs: {len(loss_history)}")
        # Show every 20th epoch for brevity
        if len(loss_history) > 20:
            print(f"  Sample (every 20 epochs): {loss_history[::20]}")
        else:
            print(f"  Full history: {loss_history}")
    else:
        print("No training loss history found")

# %%
if test():
    # Display model's parameters in a pretty format
    pretty_print_params(engine, show_stats=False)

# %%
if test():
    # Display trained parameters
    print("Trained Parameters:")
    print(f"  Number of parameter groups: {len(engine.parameter_store)}")
    for param_name, param_value in engine.parameter_store.items():
        print(f"  {param_name}: shape {tuple(param_value.shape)}")
        # Show a sample of the parameter values
        if param_value.numel() <= 20:
            print(f"    values: {param_value.data.flatten().tolist()}")
        else:
            print(f"    sample (first 5): {param_value.data.flatten()[:5].tolist()}")

# %%
if test():
    # Display prediction results
    print("Prediction Output:")
    print(f"  Type: {type(output)}")
    if hasattr(output, 'content'):
        print(f"  Content DataFrame shape: {output.content.shape}")
        print(f"  Content columns: {list(output.content.columns)}")
        print(f"  First few rows:\n{output.content.head()}")
    if hasattr(output, 'embeddings'):
        print(f"  Embeddings shapes: {output.embedding_shapes}")
        if output.embeddings:
            print(f"  First embedding shape: {output.embeddings[0].shape}")
            print(f"  First embedding sample (first 5 rows):\n{output.embeddings[0][:5]}")

# %%
if test():
    # Compute accuracy on test set
    # Note: MsgPassing2 aggregates by target_id, which includes ALL nodes that receive messages
    # from test nodes (including train/val neighbors). We need to filter to only test nodes.
    print("="*60)
    print("ACCURACY SUMMARY")
    print("="*60)

    if hasattr(output, 'embeddings') and output.embeddings:
        logits = output.embeddings[0]  # [num_output_nodes, num_classes] - may include train/val neighbors
        
        # Get predictions
        pred = logits.argmax(dim=1).cpu().numpy()
        
        # Align predictions with node IDs
        output_df = output.content
        if hasattr(output_df, '__class__') and output_df.__class__.__module__.startswith('cudf'):
            output_df = output_df.to_pandas()
        if 'target_id' not in output_df.columns and 'node_id' not in output_df.columns and len(output_df) > 0:
            output_df = output_df.copy(); output_df['node_id'] = list(output_df.index) if hasattr(output_df, 'index') else list(range(len(output_df)))
        # The output has target_id (from MsgPassing2), merge with node_df to get labels and masks
        key_col = 'target_id' if 'target_id' in output_df.columns else 'node_id'
        merged = output_df[[key_col]].merge(
            node_df[['node_id', 'label', 'is_test']],
            left_on=key_col,
            right_on='node_id',
            how='left'
        )
        
        # CRITICAL: Filter to only test nodes (MsgPassing2 includes all nodes that receive messages)
        test_mask = merged['is_test'].fillna(False).astype(bool)
        test_labels = merged.loc[test_mask, 'label'].values
        test_preds = pred[test_mask]
        
        if len(test_labels) > 0:
            test_accuracy = (test_preds == test_labels).mean()
            print(f"Test Accuracy:  {test_accuracy:.4f} ({np.sum(test_preds == test_labels)}/{len(test_labels)} correct)")
            print(f"  (Filtered from {len(pred)} total predictions to {len(test_preds)} test nodes)")

            # Verify we have the correct number of test nodes
            expected_test_count = node_df['is_test'].sum()
            if len(test_preds) != expected_test_count:
                raise AssertionError(f"Expected {expected_test_count} test nodes, got {len(test_preds)}")
            print(f"  ✓ Correct: Found all {expected_test_count} test nodes")
        else:
            test_accuracy = None
            print("No test nodes found in predictions")
        
        # Display training accuracy if computed (from previous cell)
        # Note: train_acc is already computed and filtered in above Cell
        try:
            if train_acc is not None:
                correct_count = np.sum(train_pred_filtered == train_labels)
                total_count = len(train_labels)
                print(f"Train Accuracy: {train_acc:.4f} ({correct_count}/{total_count} correct)")
        except NameError:
            raise NameError("train_acc is not computed or not defined in this scope.")
        
        print("="*60)
        
        # Basic assertions for test validation
        print("\nTest Assertions:")
        assert 'Loss' in engine.trained_modules, "Training should have completed"
        assert test_accuracy is not None, "Test accuracy should be computed"
        assert 0.0 <= test_accuracy <= 1.0, "Test accuracy should be between 0 and 1"
        assert test_accuracy > 0.3, f"Test accuracy should be reasonable (got {test_accuracy:.4f})"
        try:
            if train_acc is not None:
                assert 0.0 <= train_acc <= 1.0, "Train accuracy should be between 0 and 1"
                assert train_acc > 0.3, f"Train accuracy should be reasonable (got {train_acc:.4f})"
        except NameError:
            pass  # train_acc not computed
        print("✓ All assertions passed")
        
    else:
        print("No embeddings found in output")
        raise AssertionError("Output should contain embeddings")


# %%
# --- CI completion sentinel (see .github/workflows/test.yaml) ---
# This file runs its checks at import time under `if test():` and defines no
# `def test_*` of its own, so pytest would otherwise collect 0 tests and exit 5
# ("no tests collected") even if the body never ran (file moved, collection
# skipped, etc.). We record that import reached the end (every prior
# `if test():` block ran without raising) and assert it in a real test below,
# so a green CI step means the checks actually executed.
_SCAFFOLD_902_COMPLETED = False
if test():
    _SCAFFOLD_902_COMPLETED = True


# %%
def test_902_ran_to_completion():
    """Real pytest test: lets CI tell 'checks ran and passed' apart from
    'no tests collected'. The heavy work runs above at import under
    `if test():`; this asserts it reached the end rather than being skipped."""
    assert _SCAFFOLD_902_COMPLETED, (
        "scaffold test_902 inline checks did not run to completion - a prior "
        "`if test():` block was skipped or raised before the end of the module"
    )
