# Scaffold System for Comparing PyG and RelNN Implementations

## Overview

The scaffold system provides automatic comparison, weight synchronization, and debugging capabilities for comparing your custom RelNN implementations with PyG (PyTorch Geometric) implementations. It is designed to be:

- **Robust**: Comprehensive error checking and detailed mismatch reporting
- **Flexible**: Supports automatic and manual weight mapping, custom hooks, and various output types
- **Easy to Use**: Minimal code intrusion with context managers and decorators
- **Easy to Debug**: Detailed error messages with statistics about mismatches
- **Automatic**: Automatic weight mapping by shape, automatic hook registration, and automatic comparison

## RelNN ModuleDict Naming Update

RelNN now uses sanitized internal `ModuleDict` keys (for example, `op_0007__...`) that are not guaranteed to match term-graph node ids.

- For **operator/module lookup by graph node id**, use `relnn_model.module_for_node("MyNodeId")`.
- For scaffold output hooks, use **`node.<graph_node_id>`** paths (for example, `node.orderby_MsgPassing1`).
- Do not rely on `relnn_model.ops["<graph_node_id>"]` as a stable API.

Note: weight-mapping APIs still work with parameter names reported by `named_parameters()`; those names may contain internal prefixes such as `ops.`.

## Quick Start

```python
from relann.scaffold import Scaffold

# Create scaffold
scaffold = Scaffold(pyg_model, relnn_model)

# Auto-map and sync weights
scaffold.auto_map_weights()
scaffold.sync_weights()

# Add hooks for intermediate outputs by graph node id
scaffold.add_output_hook("conv1", "node.layer1")

# Compare forward passes
with scaffold.compare_forward():
    pyg_out = pyg_model(x, edge_index)
    relnn_out = relnn_model(relations)

# Assert all match (raises on mismatch)
scaffold.assert_all_match()
```

## Core Components

### 1. Scaffold Class

The main class that orchestrates comparison between PyG and RelNN models.

**Key Methods:**
- `auto_map_weights()`: Automatically map weights by matching shapes
- `add_weight_mapping(relnn_path, pyg_path)`: Manually add weight mappings
- `sync_weights()`: Copy weights from PyG to RelNN
- `add_output_hook(pyg_path, relnn_path)`: Add hooks to capture intermediate outputs
- `compare_forward()`: Context manager for capturing forward pass outputs
- `compare_outputs(pyg_out, relnn_out, name)`: Compare two outputs
- `assert_all_match()`: Assert all comparisons match (raises on failure)

### 2. WeightMapper Class

Handles mapping and copying weights between models.

**Features:**
- Automatic mapping by shape
- Manual mapping support
- Shape validation
- Device/dtype handling

### 3. ForwardHook Class

Captures intermediate outputs during forward pass.

**Features:**
- Automatic registration on modules
- Detached tensor cloning
- Optional gradient capture

### 4. ComparisonResult Class

Stores comparison results with detailed statistics.

**Fields:**
- `success`: Whether comparison passed
- `message`: Human-readable message
- `max_diff`: Maximum difference
- `mean_diff`: Mean difference
- `details`: Additional statistics

## Usage Patterns

### Pattern 1: Basic Comparison

```python
scaffold = Scaffold(pyg_model, relnn_model)
scaffold.auto_map_weights()
scaffold.sync_weights()

with scaffold.compare_forward():
    pyg_out = pyg_model(x, edge_index)
    relnn_out = relnn_model(relations)

result = scaffold.compare_outputs(pyg_out, relnn_out, "output")
scaffold.assert_all_match()
```

### Pattern 2: With Intermediate Output Hooks

```python
scaffold = Scaffold(pyg_model, relnn_model)
scaffold.auto_map_weights()
scaffold.sync_weights()

# Add hooks for each layer via graph node ids
scaffold.add_output_hook("conv1", "node.layer1", hook_name="layer1")
scaffold.add_output_hook("conv2", "node.layer2", hook_name="layer2")

with scaffold.compare_forward():
    pyg_out = pyg_model(x, edge_index)
    relnn_out = relnn_model(relations)

# Automatically compares all hook outputs
scaffold.assert_all_match()
```

### Pattern 3: Context Manager (Automatic Cleanup)

```python
with Scaffold(pyg_model, relnn_model) as scaffold:
    scaffold.auto_map_weights()
    scaffold.sync_weights()
    scaffold.add_output_hook("conv1", "node.layer1")
    
    with scaffold.compare_forward():
        pyg_out = pyg_model(x, edge_index)
        relnn_out = relnn_model(relations)
    
    scaffold.assert_all_match()
```

### Pattern 4: Using Decorator (Minimal Code)

```python
from relann.scaffold import scaffold_decorator

@scaffold_decorator(
    pyg_model, relnn_model,
    weight_mappings={
        # Use exact RelNN parameter names from relnn_model.named_parameters()
        "ops.op_0003__NodesEmbedding1.transformation._module.weight": "conv1.lin.weight",
    },
    output_hooks=[("conv1", "node.layer1")],
    auto_map=True
)
def train_step(x, edge_index, relations):
    pyg_out = pyg_model(x, edge_index)
    relnn_out = relnn_model(relations)
    return pyg_out, relnn_out

# Automatically handles everything
pyg_out, relnn_out = train_step(x, edge_index, relations)
```

## Weight Mapping and Comparison

### When to Copy vs Compare

**Copy weights (`sync_weights()`)**: Use when you want to ensure both models start with the same weights for a fair comparison of forward passes. This is typically done before comparing outputs to ensure any differences are due to implementation differences, not different initializations.

**Compare weights (`compare_weights()`)**: Use when you just want to verify that weights are the same without modifying them. Useful for:
- Checking if weights are already synchronized
- After training to verify both models learned similar weights
- Comparing weights at any point without side effects

### Automatic Mapping

```python
scaffold.auto_map_weights()  # Maps by shape
scaffold.auto_map_weights(prefix_filter="ops.op_")  # Only map specific RelNN ops prefix
```

### Manual Mapping

```python
scaffold.add_weight_mapping(
    "ops.op_0003__NodesEmbedding1.transformation._module.weight",  # RelNN path
    "conv1.lin.weight"                  # PyG path
)
```

### Copying Weights (for Fair Comparison)

```python
# Copy weights from PyG to RelNN (modifies RelNN model)
scaffold.sync_weights()  # Use before comparing forward passes
```

### Comparing Weights (Without Modifying)

```python
# Just compare weights (doesn't modify anything)
weight_results = scaffold.compare_weights()
for param_name, result in weight_results.items():
    if not result.success:
        print(f"Mismatch in {param_name}: max_diff={result.max_diff}")

# Or assert all weights match
scaffold.assert_weights_match()  # Raises AssertionError on mismatch
```

### View Mapping Summary

```python
print(scaffold.weight_mapper.get_mapping_summary())
```

## Output Comparison

### Supported Output Types

1. **Tensors**: Direct tensor comparison
2. **Dictionaries**: Recursive comparison of dict values (e.g., heterogeneous graphs)
3. **Tuples/Lists**: Element-wise comparison

### Comparison Options

```python
# Standard comparison
result = scaffold.compare_outputs(pyg_out, relnn_out, "output")

# With permutation tolerance (for order-invariant outputs)
result = scaffold.compare_outputs(
    pyg_out, relnn_out, "output", 
    use_permutation=True
)
```

### Accessing Comparison Results

```python
result = scaffold.compare_outputs(pyg_out, relnn_out, "output")
print(f"Success: {result.success}")
print(f"Max diff: {result.max_diff}")
print(f"Mean diff: {result.mean_diff}")
print(f"Details: {result.details}")
```

## Error Handling and Debugging

### Detailed Error Messages

When a mismatch occurs, `assert_all_match()` provides:

```
============================================================
SCAFFOLD COMPARISON FAILED
============================================================

❌ layer1: Tensor comparison for layer1: FAIL
   Max diff: 1.234567e-05, Mean diff: 5.432109e-06
   Details: {'max_diff': 1.234567e-05, 'mean_diff': 5.432109e-06, ...}

============================================================
```

### Verbose Mode

```python
scaffold = Scaffold(pyg_model, relnn_model, verbose=True)  # Default
# Prints detailed information about mismatches
```

### Non-Strict Mode

```python
# Don't raise on mismatch, just return False
scaffold.assert_all_match(raise_on_fail=False)
```

## Advanced Features

### Custom Tolerances

```python
scaffold = Scaffold(
    pyg_model, relnn_model,
    atol=1e-6,  # Absolute tolerance
    rtol=1e-6   # Relative tolerance
)
```

### Comparing Weights After Training

```python
# After training both models - just compare without modifying
weight_results = scaffold.compare_weights()
for param_name, result in weight_results.items():
    if not result.success:
        print(f"Mismatch in {param_name}: {result.max_diff}")

# Or use the assertion method
scaffold.assert_weights_match()  # Raises on mismatch with detailed error
```

### Module Path Resolution

The scaffold supports various module path formats:

```python
# RelNN graph node lookup (recommended)
scaffold.add_output_hook("conv1", "node.layer1")

# Nested PyG attributes still work on the PyG side
scaffold.add_output_hook("layers.0", "node.layer1")

# Indexed PyG access
scaffold.add_output_hook("layers[0]", "node.layer1")
```

## Best Practices

1. **Understand when to copy vs compare weights**:
   - Use `sync_weights()` before comparing forward passes to ensure fair comparison
   - Use `compare_weights()` to verify weights without modifying models
2. **Use automatic mapping first**: Try `auto_map_weights()` before manual mapping
3. **Add hooks for debugging**: Add hooks at key points to identify where mismatches occur
4. **Use context managers**: Use `with Scaffold(...)` for automatic cleanup
5. **Check mapping summary**: Print `get_mapping_summary()` to verify weight mappings
6. **Set appropriate tolerances**: Adjust `atol` and `rtol` based on your precision requirements

## Troubleshooting

### "Unmapped RelNN parameters" Error

**Solution**: Either add manual mappings or use `strict=False` in `sync_weights()`:

```python
scaffold.sync_weights(strict=False)  # Only copy mapped parameters
```

### "Module not found" Error

**Solution**: Check module paths. Use `print(model)` to see the module structure:

```python
print(pyg_model)
print(relnn_model)
```

### Shape Mismatches

**Solution**: Verify your models have the same architecture, or use manual weight mappings for different architectures.

### Large Differences in Outputs

**Solution**: 
1. Check that weights are properly synced
2. Verify both models are in the same mode (train/eval)
3. Check for numerical stability issues
4. Increase tolerances if appropriate

## Example: Complete Workflow

```python
from relann.scaffold import Scaffold
from relann.torch_utils import full_seed

# Set seed for reproducibility
full_seed(42)

# Create models
pyg_model = SimpleGCN(...).to(device)
relnn_model = create_relnn_model(...).to(device)

# Create scaffold
with Scaffold(pyg_model, relnn_model, atol=1e-5, rtol=1e-5) as scaffold:
    # Setup weight mapping
    scaffold.auto_map_weights()
    print(scaffold.weight_mapper.get_mapping_summary())
    
    # Option 1: Copy weights for fair forward pass comparison
    scaffold.sync_weights()  # Use this before comparing outputs
    
    # Option 2: Just compare weights without modifying
    # scaffold.assert_weights_match()  # Use this to verify weights are the same
    
    # Add hooks for intermediate outputs
    scaffold.add_output_hook("conv1", "node.layer1", hook_name="layer1")
    scaffold.add_output_hook("conv2", "node.layer2", hook_name="layer2")
    
    # Compare forward pass
    with scaffold.compare_forward():
        pyg_out = pyg_model(x, edge_index)
        relnn_out = relnn_model(relations)
    
    # Compare final output
    result = scaffold.compare_outputs(pyg_out, relnn_out, "final_output")
    print(f"Final output match: {result.success}")
    
    # Assert all comparisons pass
    scaffold.assert_all_match()
    
    print("✓ All comparisons passed!")
```

## API Reference

See `relann/scaffold.py` for complete API documentation.

