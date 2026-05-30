"""
Scaffold mechanism for comparing custom RelNN implementations with PyG.

This module provides automatic comparison, weight synchronization, and debugging
capabilities with minimal code intrusion.
"""

from __future__ import annotations

import logging
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from collections import defaultdict, OrderedDict
import contextlib
from dataclasses import dataclass, field
import warnings
import traceback

from relann.torch_utils import equal_up_to_permutation

logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    """Result of a comparison operation."""
    success: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    max_diff: Optional[float] = None
    mean_diff: Optional[float] = None
    shape_match: bool = True


class ForwardHook:
    """Hook to capture intermediate outputs during forward pass."""
    
    def __init__(self, name: str, capture_grad: bool = False):
        self.name = name
        self.capture_grad = capture_grad
        self.output = None
        self.grad = None
        self.handle = None
        self.embedded_relation = None  # Store full EmbeddedRelation if output is one
    
    def __call__(self, module, input, output):
        """Store the output (and optionally gradient) of a module."""
        # Handle EmbeddedRelation objects (from RelNN operators)
        if hasattr(output, 'embeddings') and output.embeddings:
            # Extract first embedding tensor from EmbeddedRelation
            self.output = output.embeddings[0].detach().clone()
            # Also store the full EmbeddedRelation for potential alignment later
            self.embedded_relation = output
        elif isinstance(output, torch.Tensor):
            self.output = output.detach().clone()
            self.embedded_relation = None
        else:
            self.output = output
            self.embedded_relation = None
        return output
    
    def register(self, module: nn.Module):
        """Register this hook on a module."""
        self.handle = module.register_forward_hook(self)
        return self.handle
    
    def remove(self):
        """Remove the hook."""
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


class InputHook:
    """Hook to capture inputs to a module (using forward_pre_hook)."""
    
    def __init__(self, name: str):
        self.name = name
        self.inputs = {}  # Store multiple inputs (e.g., 'x', 'edge_index' for PyG)
        self.input = None  # For backward compatibility
        self.handle = None
    
    def __call__(self, module, input):
        """Store the input to a module."""
        # Input is typically a tuple for PyG modules (x, edge_index, ...)
        if isinstance(input, (tuple, list)):
            # For PyG modules, typically: (x, edge_index) or (x, edge_index, edge_weight, ...)
            if len(input) >= 1 and isinstance(input[0], torch.Tensor):
                self.inputs['x'] = input[0].detach().clone()
                self.input = input[0].detach().clone()  # For backward compatibility
            if len(input) >= 2 and isinstance(input[1], torch.Tensor):
                self.inputs['edge_index'] = input[1].detach().clone()
            # Store full tuple for other cases
            if not self.inputs:
                self.input = input[0] if len(input) == 1 else input
        elif isinstance(input, torch.Tensor):
            self.input = input.detach().clone()
            self.inputs['input'] = input.detach().clone()
        elif isinstance(input, dict):
            # For RelNN, input might be a dict of relations
            self.input = input
            self.inputs = input.copy() if hasattr(input, 'copy') else input
        else:
            self.input = input
            self.inputs['input'] = input
        return input
    
    def register(self, module: nn.Module):
        """Register this hook on a module."""
        self.handle = module.register_forward_pre_hook(self)
        return self.handle
    
    def remove(self):
        """Remove the hook."""
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


class WeightMapper:
    """Maps weights between PyG and RelNN models."""
    
    def __init__(self, pyg_model: nn.Module, relnn_model: nn.Module):
        self.pyg_model = pyg_model
        self.relnn_model = relnn_model
        self.mapping: Dict[str, str] = {}  # relnn_path -> pyg_path
        self._pyg_params: Dict[str, torch.nn.Parameter] = {}
        self._relnn_params: Dict[str, torch.nn.Parameter] = {}
        self._build_param_dicts()
    
    def _build_param_dicts(self):
        """Build dictionaries of all parameters in both models."""
        for name, param in self.pyg_model.named_parameters():
            self._pyg_params[name] = param
        
        for name, param in self.relnn_model.named_parameters():
            self._relnn_params[name] = param
    
    def add_mapping(self, relnn_path: str, pyg_path: str):
        """Add a manual mapping between parameter paths."""
        if relnn_path not in self._relnn_params:
            raise ValueError(f"RelNN parameter '{relnn_path}' not found")
        if pyg_path not in self._pyg_params:
            raise ValueError(f"PyG parameter '{pyg_path}' not found")
        
        self.mapping[relnn_path] = pyg_path
    
    def auto_map_by_shape(self, prefix_filter: Optional[str] = None):
        """
        Automatically map parameters by matching shapes.
        
        Parameters
        ----------
        prefix_filter : str, optional
            Only consider parameters with this prefix in RelNN model.
        """
        # Group PyG params by shape
        pyg_by_shape: Dict[Tuple, List[str]] = defaultdict(list)
        for name, param in self._pyg_params.items():
            pyg_by_shape[param.shape].append(name)
        
        # Try to match RelNN params
        for relnn_name, relnn_param in self._relnn_params.items():
            if prefix_filter and not relnn_name.startswith(prefix_filter):
                continue
            
            if relnn_name in self.mapping:
                continue  # Already mapped
            
            shape = relnn_param.shape
            if shape in pyg_by_shape:
                # Use the first match (could be improved with heuristics)
                if len(pyg_by_shape[shape]) > 0:
                    pyg_name = pyg_by_shape[shape].pop(0)
                    self.mapping[relnn_name] = pyg_name
    
    def copy_weights_pyg_to_relnn(self, strict: bool = True):
        """
        Copy weights from PyG model to RelNN model.
        
        **Note**: This modifies the RelNN model's weights. If you just want to
        compare weights without modifying them, use `compare_weights()` instead.
        
        Parameters
        ----------
        strict : bool
            If True, raise error if any RelNN parameter is not mapped.
            If False, only copy mapped parameters.
        """
        unmapped = []
        with torch.no_grad():
            for relnn_name, relnn_param in self._relnn_params.items():
                if relnn_name not in self.mapping:
                    if strict:
                        unmapped.append(relnn_name)
                    continue
                
                pyg_name = self.mapping[relnn_name]
                pyg_param = self._pyg_params[pyg_name]
                
                if relnn_param.shape != pyg_param.shape:
                    raise ValueError(
                        f"Shape mismatch for '{relnn_name}' -> '{pyg_name}': "
                        f"{relnn_param.shape} vs {pyg_param.shape}"
                    )
                
                relnn_param.data.copy_(pyg_param.data.to(
                    relnn_param.device, 
                    relnn_param.dtype
                ))
        
        if unmapped:
            raise ValueError(
                f"Unmapped RelNN parameters (strict=True): {unmapped}"
            )
    
    def get_mapping_summary(self) -> str:
        """Get a summary of the current mapping."""
        lines = ["Weight Mapping Summary:"]
        lines.append(f"  Total PyG parameters: {len(self._pyg_params)}")
        lines.append(f"  Total RelNN parameters: {len(self._relnn_params)}")
        lines.append(f"  Mapped parameters: {len(self.mapping)}")
        lines.append("\nMappings:")
        for relnn_name, pyg_name in sorted(self.mapping.items()):
            relnn_shape = self._relnn_params[relnn_name].shape
            pyg_shape = self._pyg_params[pyg_name].shape
            lines.append(f"  {relnn_name} ({relnn_shape}) <- {pyg_name} ({pyg_shape})")
        
        unmapped_relnn = set(self._relnn_params.keys()) - set(self.mapping.keys())
        if unmapped_relnn:
            lines.append(f"\nUnmapped RelNN parameters ({len(unmapped_relnn)}):")
            for name in sorted(unmapped_relnn):
                lines.append(f"  {name} ({self._relnn_params[name].shape})")
        
        return "\n".join(lines)


class Scaffold:
    """
    Main scaffold class for comparing PyG and RelNN implementations.
    
    This class provides automatic hooking, weight synchronization, and comparison
    capabilities with minimal code intrusion.
    
    Example
    -------
    >>> scaffold = Scaffold(pyg_model, relnn_model)
    >>> scaffold.add_output_hook("conv1", "layer1")
    >>> scaffold.sync_weights()
    >>> with scaffold.compare_forward():
    ...     pyg_out = pyg_model(x, edge_index)
    ...     relnn_out = relnn_model(relations)
    >>> scaffold.assert_all_match()
    """
    
    def __init__(
        self,
        pyg_model: nn.Module,
        relnn_model: nn.Module,
        atol: float = 1e-5,
        rtol: float = 1e-5,
        verbose: bool = True
    ):
        """
        Initialize the scaffold.
        
        Parameters
        ----------
        pyg_model : nn.Module
            The PyG reference model.
        relnn_model : nn.Module
            The custom RelNN model to compare.
        atol : float
            Absolute tolerance for tensor comparisons.
        rtol : float
            Relative tolerance for tensor comparisons.
        verbose : bool
            Whether to print detailed comparison information.
        """
        self.pyg_model = pyg_model
        self.relnn_model = relnn_model
        self.atol = atol
        self.rtol = rtol
        self.verbose = verbose
        
        # Weight mapping
        self.weight_mapper = WeightMapper(pyg_model, relnn_model)
        
        # Output hooks
        self.pyg_hooks: Dict[str, ForwardHook] = {}
        self.relnn_hooks: Dict[str, ForwardHook] = {}
        
        # Input hooks (to automatically capture inputs to first layers)
        self.pyg_input_hooks: Dict[str, InputHook] = {}
        self.relnn_input_hooks: Dict[str, InputHook] = {}
        
        # Captured outputs
        self.pyg_outputs: Dict[str, Any] = {}
        self.relnn_outputs: Dict[str, Any] = {}
        
        # Captured inputs (for debug visualization)
        self.pyg_inputs: Dict[str, Any] = {}
        self.relnn_inputs: Dict[str, Any] = {}
        
        # Comparison results
        self.comparison_results: Dict[str, ComparisonResult] = {}
        
        # Training state
        self._training_mode = False
        self._forward_captured = False
        
        # Store original forward methods for restoration
        self._pyg_original_forward = None
        self._relnn_original_forward = None
        
        # Hook path mapping for cache-based fallback capture
        self._hook_path_mapping: Dict[str, str] = {}
    
    def add_output_hook(
        self,
        pyg_module_path: str,
        relnn_module_path: str,
        hook_name: Optional[str] = None
    ):
        """
        Add hooks to capture outputs from corresponding modules.
        
        Note: RelNN operators are called via op(sons) which triggers PyTorch's __call__
        mechanism, so forward hooks should work correctly. However, if hooks don't capture
        outputs for some reason, you can use `register_relnn_output()` to manually register
        RelNN outputs.
        
        Parameters
        ----------
        pyg_module_path : str
            Path to PyG module (e.g., "conv1" or "layers.0").
        relnn_module_path : str
            Path to RelNN module (e.g., "ops.layer1").
        hook_name : str, optional
            Name for this hook pair. If None, uses pyg_module_path.
        """
        if hook_name is None:
            hook_name = pyg_module_path
        
        # Get PyG module
        pyg_module = self._get_module_by_path(self.pyg_model, pyg_module_path)
        if pyg_module is None:
            raise ValueError(f"PyG module '{pyg_module_path}' not found")
        
        # Get RelNN module
        relnn_module = self._get_module_by_path(self.relnn_model, relnn_module_path)
        if relnn_module is None:
            raise ValueError(f"RelNN module '{relnn_module_path}' not found")
        
        # Create and register hooks
        pyg_hook = ForwardHook(f"pyg_{hook_name}")
        relnn_hook = ForwardHook(f"relnn_{hook_name}")
        
        pyg_hook.register(pyg_module)
        relnn_hook.register(relnn_module)
        
        self.pyg_hooks[hook_name] = pyg_hook
        self.relnn_hooks[hook_name] = relnn_hook
        
        # Store path mapping for cache-based fallback capture
        self._hook_path_mapping[hook_name] = relnn_module_path
        
        # Also register input hooks on the first layer to automatically capture inputs
        # Only register if this is the first hook (to avoid duplicate captures)
        if len(self.pyg_hooks) == 1:
            pyg_input_hook = InputHook(f"pyg_input_{hook_name}")
            pyg_input_hook.register(pyg_module)
            self.pyg_input_hooks[hook_name] = pyg_input_hook
        
        if len(self.relnn_hooks) == 1:
            relnn_input_hook = InputHook(f"relnn_input_{hook_name}")
            relnn_input_hook.register(relnn_module)
            self.relnn_input_hooks[hook_name] = relnn_input_hook
    
    def register_relnn_output(self, hook_name: str, output: Any):
        """
        Manually register an output from a RelNN module.
        
        This is a fallback method if forward hooks don't capture outputs correctly
        for some reason. Normally, hooks should work since RelNN operators are called
        via op(sons) which triggers PyTorch's __call__ mechanism.
        
        Parameters
        ----------
        hook_name : str
            Name of the hook (should match the hook_name used in add_output_hook).
        output : Any
            The output to register (typically a torch.Tensor or EmbeddedRelation).
        """
        # Extract tensor from EmbeddedRelation if needed
        if hasattr(output, 'embeddings') and output.embeddings:
            # Use first embedding tensor
            tensor_output = output.embeddings[0]
        elif isinstance(output, torch.Tensor):
            tensor_output = output
        else:
            tensor_output = output
        
        # Store in relnn_outputs (will be used by compare_all_hooks)
        self.relnn_outputs[hook_name] = tensor_output
    
    def register_pyg_inputs(self, **inputs):
        """
        Register PyG model inputs for debug visualization.
        
        Parameters
        ----------
        **inputs : dict
            Named inputs to register. Common names: 'x' (node features), 
            'edge_index' (edge connectivity), etc.
        
        Example
        -------
        >>> with scaffold.compare_forward():
        ...     scaffold.register_pyg_inputs(x=data.x, edge_index=edge_index)
        ...     pyg_out = pyg_model(data.x, edge_index)
        """
        self.pyg_inputs.update(inputs)
    
    def register_relnn_inputs(self, **inputs):
        """
        Register RelNN model inputs for debug visualization.
        
        Parameters
        ----------
        **inputs : dict
            Named inputs to register. Common names: 'nodes_all', 'edges', etc.
            Values can be tensors, dicts, or EmbeddedRelations.
        
        Example
        -------
        >>> with scaffold.compare_forward():
        ...     scaffold.register_relnn_inputs(nodes_all=nodes_all_tensor, edges=edge_tensor)
        ...     relnn_out = relnn_model(relations)
        """
        # Extract tensors from EmbeddedRelations or dicts if needed
        processed_inputs = {}
        for name, value in inputs.items():
            if isinstance(value, dict):
                # Try to extract tensor from dict (e.g., {'embeddings': [tensor], 'content': df})
                if 'embeddings' in value and value['embeddings']:
                    processed_inputs[name] = value['embeddings'][0] if isinstance(value['embeddings'], list) else value['embeddings']
                else:
                    processed_inputs[name] = value
            elif hasattr(value, 'embeddings') and value.embeddings:
                # EmbeddedRelation object
                processed_inputs[name] = value.embeddings[0] if isinstance(value.embeddings, list) else value.embeddings
            elif isinstance(value, torch.Tensor):
                processed_inputs[name] = value
            else:
                processed_inputs[name] = value
        
        self.relnn_inputs.update(processed_inputs)
    
    def _get_module_by_path(self, model: nn.Module, path: str) -> Optional[nn.Module]:
        """Get a module by its path (e.g., "conv1" or "layers.0")."""
        # Preferred node-addressing for RelNN nodes.
        if path.startswith("node.") and hasattr(model, "module_for_node"):
            node_id = path[len("node."):]
            if node_id:
                try:
                    resolved = model.module_for_node(node_id)
                    if isinstance(resolved, nn.Module):
                        return resolved
                except KeyError:
                    pass
        parts = path.split('.')
        current = model
        for part in parts:
            if hasattr(current, part):
                current = getattr(current, part)
            elif hasattr(current, '__getitem__'):
                try:
                    current = current[int(part)]
                except (ValueError, KeyError, IndexError):
                    return None
            else:
                return None
        return current if isinstance(current, nn.Module) else None
    
    def add_weight_mapping(self, relnn_path: str, pyg_path: str):
        """Add a manual weight mapping."""
        self.weight_mapper.add_mapping(relnn_path, pyg_path)
    
    def auto_map_weights(self, prefix_filter: Optional[str] = None):
        """Automatically map weights by shape."""
        self.weight_mapper.auto_map_by_shape(prefix_filter)
        if self.verbose:
            print(self.weight_mapper.get_mapping_summary())
    
    def sync_weights(self, strict: bool = True):
        """
        Copy weights from PyG to RelNN model.
        
        **When to use**: Use this when you want to ensure both models start with
        the same weights for a fair comparison of forward passes. This is typically
        done before comparing outputs to ensure any differences are due to
        implementation differences, not different initializations.
        
        **When NOT to use**: If you just want to compare existing weights without
        modifying them, use `compare_weights()` instead.
        
        Parameters
        ----------
        strict : bool
            If True, raise error if any RelNN parameter is not mapped.
            If False, only copy mapped parameters (unmapped ones are left unchanged).
        """
        self.weight_mapper.copy_weights_pyg_to_relnn(strict=strict)
        if self.verbose:
            print("[OK] Weights synchronized from PyG to RelNN")
    
    def _wrap_pyg_forward(self, original_forward):
        """Wrapper to capture PyG model inputs automatically."""
        def wrapped_forward(*args, **kwargs):
            # Capture inputs: PyG models typically take (x, edge_index, ...) as positional args
            if len(args) >= 1 and isinstance(args[0], torch.Tensor):
                self.pyg_inputs['x'] = args[0].detach().clone()
            if len(args) >= 2 and isinstance(args[1], torch.Tensor):
                self.pyg_inputs['edge_index'] = args[1].detach().clone()
            # Also capture any keyword arguments
            for key, value in kwargs.items():
                if isinstance(value, torch.Tensor):
                    self.pyg_inputs[key] = value.detach().clone()
            # Call original forward
            return original_forward(*args, **kwargs)
        return wrapped_forward
    
    def _wrap_relnn_forward(self, original_forward):
        """Wrapper to capture RelNN model inputs automatically."""
        def wrapped_forward(relations: Optional[Dict[str, dict]] = None, *args, **kwargs):
            # Capture inputs: RelNN models take a dict of relations
            if relations:
                # Process each relation in the dict
                for name, relation in relations.items():
                    # Extract tensor from EmbeddedRelation dict format
                    if isinstance(relation, dict):
                        if 'embeddings' in relation and relation['embeddings']:
                            # Extract first embedding tensor
                            emb = relation['embeddings']
                            if isinstance(emb, list) and len(emb) > 0:
                                self.relnn_inputs[name] = emb[0].detach().clone() if isinstance(emb[0], torch.Tensor) else emb[0]
                            elif isinstance(emb, torch.Tensor):
                                self.relnn_inputs[name] = emb.detach().clone()
                        else:
                            # Store the relation dict as-is if no embeddings found
                            self.relnn_inputs[name] = relation
                    elif hasattr(relation, 'embeddings') and relation.embeddings:
                        # EmbeddedRelation object
                        if isinstance(relation.embeddings, list) and len(relation.embeddings) > 0:
                            self.relnn_inputs[name] = relation.embeddings[0].detach().clone() if isinstance(relation.embeddings[0], torch.Tensor) else relation.embeddings[0]
                        elif isinstance(relation.embeddings, torch.Tensor):
                            self.relnn_inputs[name] = relation.embeddings.detach().clone()
                    elif isinstance(relation, torch.Tensor):
                        self.relnn_inputs[name] = relation.detach().clone()
                    else:
                        self.relnn_inputs[name] = relation
            # Call original forward
            return original_forward(relations, *args, **kwargs)
        return wrapped_forward
    
    @contextlib.contextmanager
    def compare_forward(self):
        """
        Context manager to capture outputs during forward pass.
        Automatically captures inputs from both models' forward calls.
        
        Example
        -------
        >>> with scaffold.compare_forward():
        ...     pyg_out = pyg_model(data.x, edge_index)
        ...     relnn_out = relnn_model(relations)
        """
        # Clear previous captures
        self.pyg_outputs.clear()
        self.relnn_outputs.clear()
        self.pyg_inputs.clear()
        self.relnn_inputs.clear()
        self.comparison_results.clear()
        self._forward_captured = False
        
        # Store original forward methods and wrap them to capture inputs
        self._pyg_original_forward = self.pyg_model.forward
        self._relnn_original_forward = self.relnn_model.forward
        
        # Wrap forward methods to automatically capture inputs
        self.pyg_model.forward = self._wrap_pyg_forward(self._pyg_original_forward)
        self.relnn_model.forward = self._wrap_relnn_forward(self._relnn_original_forward)
        
        try:
            yield self
        finally:
            # Restore original forward methods
            if self._pyg_original_forward is not None:
                self.pyg_model.forward = self._pyg_original_forward
                self._pyg_original_forward = None
            if self._relnn_original_forward is not None:
                self.relnn_model.forward = self._relnn_original_forward
                self._relnn_original_forward = None
            
            # Capture outputs from hooks
            for name, hook in self.pyg_hooks.items():
                if hook.output is not None:
                    self.pyg_outputs[name] = hook.output
            
            for name, hook in self.relnn_hooks.items():
                if hook.output is not None:
                    self.relnn_outputs[name] = hook.output
            
            # Fallback: For RelNN, hooks often don't fire due to custom evaluation
            # Try to extract outputs from RelNN's internal cache
            self._capture_relnn_from_cache()
            
            self._forward_captured = True
    
    def _capture_relnn_from_cache(self):
        """
        Capture RelNN outputs from its internal _cache_forward.
        This is a fallback when forward hooks don't fire due to RelNN's
        custom evaluation mechanism.
        """
        if not hasattr(self.relnn_model, '_cache_forward'):
            return
        
        cache = self.relnn_model._cache_forward
        if not cache:
            return
        
        # For each registered RelNN hook that didn't capture, try to get from cache
        for hook_name, hook in self.relnn_hooks.items():
            if hook_name in self.relnn_outputs:
                continue  # Already captured via hook
            
            # Extract the RelNN module path to find the cache key
            # The hook was registered with a path like "ops.agg_Output"
            # The cache key is the node_id like "agg_Output"
            
            # Try to find matching cache entry
            # Check all registered hook paths
            for pyg_path, relnn_path in self._get_hook_path_mapping().items():
                if pyg_path == hook_name or relnn_path.endswith(hook_name):
                    # Extract node name from path (e.g., "ops.agg_Output" -> "agg_Output")
                    cache_key = relnn_path.split('.')[-1] if '.' in relnn_path else relnn_path
                    
                    if cache_key in cache:
                        cached_result = cache[cache_key]
                        # Extract tensor from EmbeddedRelation
                        if hasattr(cached_result, 'embeddings') and cached_result.embeddings:
                            tensor = cached_result.embeddings[0]
                            if isinstance(tensor, torch.Tensor):
                                self.relnn_outputs[hook_name] = tensor.detach().clone()
                        elif isinstance(cached_result, torch.Tensor):
                            self.relnn_outputs[hook_name] = cached_result.detach().clone()
                        break
            
            # If still not found, try direct lookup with hook_name as cache key
            if hook_name not in self.relnn_outputs:
                # Try various cache key patterns
                for cache_key in [hook_name, f"agg_{hook_name}", f"transformation_{hook_name}"]:
                    if cache_key in cache:
                        cached_result = cache[cache_key]
                        if hasattr(cached_result, 'embeddings') and cached_result.embeddings:
                            tensor = cached_result.embeddings[0]
                            if isinstance(tensor, torch.Tensor):
                                self.relnn_outputs[hook_name] = tensor.detach().clone()
                                break
                        elif isinstance(cached_result, torch.Tensor):
                            self.relnn_outputs[hook_name] = cached_result.detach().clone()
                            break
    
    def _get_hook_path_mapping(self) -> Dict[str, str]:
        """Get mapping of hook names to their RelNN module paths."""
        return self._hook_path_mapping
    
    def compare_outputs(
        self,
        pyg_output: Any,
        relnn_output: Any,
        name: str = "output",
        use_permutation: bool = False
    ) -> ComparisonResult:
        """
        Compare two outputs (final or intermediate).
        
        Parameters
        ----------
        pyg_output : Any
            Output from PyG model.
        relnn_output : Any
            Output from RelNN model.
        name : str
            Name for this comparison (for error messages).
        use_permutation : bool
            If True, use equal_up_to_permutation for comparison.
        
        Returns
        -------
        ComparisonResult
            Result of the comparison.
        """
        # Handle tensor outputs
        if isinstance(pyg_output, torch.Tensor) and isinstance(relnn_output, torch.Tensor):
            return self._compare_tensors(pyg_output, relnn_output, name, use_permutation)
        
        # Handle dict outputs (e.g., heterogeneous graphs)
        elif isinstance(pyg_output, dict) and isinstance(relnn_output, dict):
            results = []
            all_match = True
            for key in set(pyg_output.keys()) | set(relnn_output.keys()):
                if key not in pyg_output:
                    result = ComparisonResult(
                        False, f"Key '{key}' missing in PyG output",
                        {"missing_in": "pyg"}
                    )
                    all_match = False
                elif key not in relnn_output:
                    result = ComparisonResult(
                        False, f"Key '{key}' missing in RelNN output",
                        {"missing_in": "relnn"}
                    )
                    all_match = False
                else:
                    result = self.compare_outputs(
                        pyg_output[key], relnn_output[key],
                        f"{name}.{key}", use_permutation
                    )
                    if not result.success:
                        all_match = False
                results.append(result)
            
            return ComparisonResult(
                all_match,
                f"Dict comparison for {name}: {'PASS' if all_match else 'FAIL'}",
                {"sub_results": results}
            )
        
        # Handle tuple outputs
        elif isinstance(pyg_output, (tuple, list)) and isinstance(relnn_output, (tuple, list)):
            if len(pyg_output) != len(relnn_output):
                return ComparisonResult(
                    False,
                    f"Length mismatch for {name}: {len(pyg_output)} vs {len(relnn_output)}"
                )
            
            results = []
            all_match = True
            for i, (pyg_item, relnn_item) in enumerate(zip(pyg_output, relnn_output)):
                result = self.compare_outputs(
                    pyg_item, relnn_item, f"{name}[{i}]", use_permutation
                )
                if not result.success:
                    all_match = False
                results.append(result)
            
            return ComparisonResult(
                all_match,
                f"Tuple/list comparison for {name}: {'PASS' if all_match else 'FAIL'}",
                {"sub_results": results}
            )
        
        else:
            return ComparisonResult(
                False,
                f"Unsupported output types for {name}: {type(pyg_output)} vs {type(relnn_output)}"
            )
    
    def _compare_tensors(
        self,
        pyg_tensor: torch.Tensor,
        relnn_tensor: torch.Tensor,
        name: str,
        use_permutation: bool
    ) -> ComparisonResult:
        """Compare two tensors."""
        # Shape check
        if pyg_tensor.shape != relnn_tensor.shape:
            return ComparisonResult(
                False,
                f"Shape mismatch for {name}: {pyg_tensor.shape} vs {relnn_tensor.shape}",
                {
                    "pyg_shape": tuple(pyg_tensor.shape),
                    "relnn_shape": tuple(relnn_tensor.shape)
                },
                shape_match=False
            )
        
        # Device and dtype check
        if pyg_tensor.device != relnn_tensor.device:
            relnn_tensor = relnn_tensor.to(pyg_tensor.device)
        if pyg_tensor.dtype != relnn_tensor.dtype:
            relnn_tensor = relnn_tensor.to(pyg_tensor.dtype)
        
        # Compute differences
        diff = torch.abs(pyg_tensor - relnn_tensor)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        # Comparison
        if use_permutation:
            match = equal_up_to_permutation(
                pyg_tensor, relnn_tensor, atol=self.atol, rtol=self.rtol
            )
        else:
            match = torch.allclose(pyg_tensor, relnn_tensor, atol=self.atol, rtol=self.rtol)
        
        result = ComparisonResult(
            match,
            f"Tensor comparison for {name}: {'PASS' if match else 'FAIL'}",
            {
                "max_diff": max_diff,
                "mean_diff": mean_diff,
                "pyg_stats": {
                    "min": pyg_tensor.min().item(),
                    "max": pyg_tensor.max().item(),
                    "mean": pyg_tensor.mean().item(),
                    "std": pyg_tensor.std().item()
                },
                "relnn_stats": {
                    "min": relnn_tensor.min().item(),
                    "max": relnn_tensor.max().item(),
                    "mean": relnn_tensor.mean().item(),
                    "std": relnn_tensor.std().item()
                }
            },
            max_diff=max_diff,
            mean_diff=mean_diff
        )
        
        if self.verbose and not match:
            print(f"\n[FAIL] Mismatch in {name}:")
            print(f"   Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}")
            print(f"   PyG stats: min={result.details['pyg_stats']['min']:.4f}, "
                  f"max={result.details['pyg_stats']['max']:.4f}, "
                  f"mean={result.details['pyg_stats']['mean']:.4f}")
            print(f"   RelNN stats: min={result.details['relnn_stats']['min']:.4f}, "
                  f"max={result.details['relnn_stats']['max']:.4f}, "
                  f"mean={result.details['relnn_stats']['mean']:.4f}")
        
        return result
    
    def compare_all_hooks(self) -> Dict[str, ComparisonResult]:
        """Compare all captured hook outputs."""
        results = {}
        # Use registered hook names, not just captured outputs
        all_names = set(self.pyg_hooks.keys()) | set(self.relnn_hooks.keys())

        for name in all_names:
            if name not in self.pyg_outputs:
                # Check if hook was registered but didn't capture
                hook_info = ""
                if name in self.pyg_hooks:
                    hook = self.pyg_hooks[name]
                    hook_info = f" (hook registered, handle valid: {hook.handle is not None}, output: {type(hook.output).__name__ if hook.output is not None else 'None'})"
                results[name] = ComparisonResult(
                    False, f"Hook '{name}' not captured in PyG model{hook_info}"
                )
            elif name not in self.relnn_outputs:
                # Check if hook was registered but didn't capture
                hook_info = ""
                if name in self.relnn_hooks:
                    hook = self.relnn_hooks[name]
                    hook_info = f" (hook registered, handle valid: {hook.handle is not None}, output: {type(hook.output).__name__ if hook.output is not None else 'None'})"
                results[name] = ComparisonResult(
                    False, f"Hook '{name}' not captured in RelNN model{hook_info}"
                )
            else:
                results[name] = self.compare_outputs(
                    self.pyg_outputs[name],
                    self.relnn_outputs[name],
                    name
                )

        self.comparison_results.update(results)
        return results
    
    def verify_hooks(self) -> None:
        """
        Print diagnostic information about registered hooks.
        Useful for debugging when hooks aren't capturing outputs.
        """
        print(f"\n{'='*60}")
        print("HOOK VERIFICATION")
        print(f"{'='*60}")
        
        print(f"\nRegistered PyG hooks ({len(self.pyg_hooks)}):")
        for name, hook in self.pyg_hooks.items():
            handle_valid = hook.handle is not None
            output_captured = hook.output is not None
            output_type = type(hook.output).__name__ if hook.output is not None else "None"
            print(f"  {name}: handle_valid={handle_valid}, output_captured={output_captured}, output_type={output_type}")
        
        print(f"\nRegistered RelNN hooks ({len(self.relnn_hooks)}):")
        for name, hook in self.relnn_hooks.items():
            handle_valid = hook.handle is not None
            output_captured = hook.output is not None
            output_type = type(hook.output).__name__ if hook.output is not None else "None"
            print(f"  {name}: handle_valid={handle_valid}, output_captured={output_captured}, output_type={output_type}")
        
        print(f"\nCaptured PyG outputs ({len(self.pyg_outputs)}):")
        for name, output in self.pyg_outputs.items():
            if isinstance(output, torch.Tensor):
                print(f"  {name}: Tensor{list(output.shape)}")
            else:
                print(f"  {name}: {type(output).__name__}")
        
        print(f"\nCaptured RelNN outputs ({len(self.relnn_outputs)}):")
        for name, output in self.relnn_outputs.items():
            if isinstance(output, torch.Tensor):
                print(f"  {name}: Tensor{list(output.shape)}")
            else:
                print(f"  {name}: {type(output).__name__}")
        
        print(f"\n{'='*60}")
    
    def _format_tensor_debug(
        self,
        pyg_tensor: torch.Tensor,
        relnn_tensor: torch.Tensor,
        name: str,
        max_rows: int = 5,
        max_cols: int = 8
    ) -> str:
        """
        Format a visual comparison of two tensors for debugging.
        
        Parameters
        ----------
        pyg_tensor : torch.Tensor
            PyG tensor output.
        relnn_tensor : torch.Tensor
            RelNN tensor output.
        name : str
            Name of the comparison.
        max_rows : int
            Maximum number of rows to display.
        max_cols : int
            Maximum number of columns to display.
        
        Returns
        -------
        str
            Formatted string showing tensor comparison.
        """
        # Ensure tensors are on same device and dtype
        if pyg_tensor.device != relnn_tensor.device:
            relnn_tensor = relnn_tensor.to(pyg_tensor.device)
        if pyg_tensor.dtype != relnn_tensor.dtype:
            relnn_tensor = relnn_tensor.to(pyg_tensor.dtype)
        
        # Require exact shape match: comparison is for exact PyG vs RelNN implementation parity
        shape_pyg = pyg_tensor.shape
        shape_relnn = relnn_tensor.shape
        if shape_pyg != shape_relnn:
            raise ValueError(
                f"Shape mismatch at {name!r}: PyG {shape_pyg} vs RelNN {shape_relnn}. "
                "Scaffold comparison expects exact implementation parity."
            )
        shape = shape_pyg
        num_dims = len(shape)
        
        # Compute differences
        diff = torch.abs(pyg_tensor - relnn_tensor)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        lines = []
        lines.append(f"\n{'─'*80}")
        lines.append(f"[OUT] {name}")
        lines.append(f"{'─'*80}")
        lines.append(f"Shape: {shape}")
        lines.append(f"Max diff: {max_diff:.6e} | Mean diff: {mean_diff:.6e}")
        
        # Show tensor values
        if num_dims == 1:
            # 1D tensor: show first max_rows elements
            n_show = min(max_rows, shape[0])
            pyg_vals = pyg_tensor[:n_show].cpu().numpy()
            relnn_vals = relnn_tensor[:n_show].cpu().numpy()
            diff_vals = diff[:n_show].cpu().numpy()
            
            lines.append(f"\nFirst {n_show} elements:")
            lines.append(f"{'Index':<8} {'PyG':<15} {'RelNN':<15} {'|Diff|':<15}")
            lines.append(f"{'─'*60}")
            for i in range(n_show):
                lines.append(f"{i:<8} {pyg_vals[i]:<15.6f} {relnn_vals[i]:<15.6f} {diff_vals[i]:<15.6e}")
            
            if shape[0] > n_show:
                lines.append(f"... ({shape[0] - n_show} more elements)")
        
        elif num_dims == 2:
            # 2D tensor: show first max_rows x max_cols (shapes already match)
            n_rows = min(max_rows, shape[0])
            n_cols = min(max_cols, shape[1])
            
            pyg_slice = pyg_tensor[:n_rows, :n_cols].cpu().numpy()
            relnn_slice = relnn_tensor[:n_rows, :n_cols].cpu().numpy()
            diff_slice = diff[:n_rows, :n_cols].cpu().numpy()
            
            lines.append(f"\nFirst {n_rows}×{n_cols} values:")
            lines.append(f"\nPyG values:")
            for i in range(n_rows):
                row_str = " ".join(f"{val:8.4f}" for val in pyg_slice[i][:n_cols])
                if n_cols < shape[1]:
                    row_str += " ..."
                lines.append(f"  [{i:2d}] {row_str}")
            
            lines.append(f"\nRelNN values:")
            for i in range(n_rows):
                row_str = " ".join(f"{val:8.4f}" for val in relnn_slice[i][:n_cols])
                if n_cols < shape[1]:
                    row_str += " ..."
                lines.append(f"  [{i:2d}] {row_str}")
            
            lines.append(f"\nAbsolute differences:")
            for i in range(n_rows):
                row_str = " ".join(f"{val:8.2e}" for val in diff_slice[i][:n_cols])
                if n_cols < shape[1]:
                    row_str += " ..."
                lines.append(f"  [{i:2d}] {row_str}")
            
            if shape[0] > n_rows or shape[1] > n_cols:
                lines.append(f"\n... ({shape[0] - n_rows if shape[0] > n_rows else 0} more rows, "
                           f"{shape[1] - n_cols if shape[1] > n_cols else 0} more cols)")
        
        else:
            # Higher dimensional: flatten and show first max_rows*max_cols elements
            pyg_flat = pyg_tensor.flatten()[:max_rows * max_cols].cpu().numpy()
            relnn_flat = relnn_tensor.flatten()[:max_rows * max_cols].cpu().numpy()
            diff_flat = diff.flatten()[:max_rows * max_cols].cpu().numpy()
            
            lines.append(f"\nFirst {len(pyg_flat)} elements (flattened):")
            lines.append(f"{'Index':<8} {'PyG':<15} {'RelNN':<15} {'|Diff|':<15}")
            lines.append(f"{'─'*60}")
            for i in range(len(pyg_flat)):
                lines.append(f"{i:<8} {pyg_flat[i]:<15.6f} {relnn_flat[i]:<15.6f} {diff_flat[i]:<15.6e}")
            
            total_elements = pyg_tensor.numel()
            if total_elements > len(pyg_flat):
                lines.append(f"... ({total_elements - len(pyg_flat)} more elements)")
        
        # Add statistics
        lines.append(f"\nStatistics:")
        lines.append(f"  PyG:    min={pyg_tensor.min().item():.6f}, max={pyg_tensor.max().item():.6f}, "
                    f"mean={pyg_tensor.mean().item():.6f}, std={pyg_tensor.std().item():.6f}")
        lines.append(f"  RelNN:  min={relnn_tensor.min().item():.6f}, max={relnn_tensor.max().item():.6f}, "
                    f"mean={relnn_tensor.mean().item():.6f}, std={relnn_tensor.std().item():.6f}")
        
        lines.append(f"{'─'*80}")
        
        return "\n".join(lines)
    
    def _format_single_tensor_debug(
        self,
        tensor: torch.Tensor,
        name: str,
        max_rows: int = 5,
        max_cols: int = 8
    ) -> str:
        """
        Format a visual representation of a single tensor for debugging.
        
        Parameters
        ----------
        tensor : torch.Tensor
            Tensor to visualize.
        name : str
            Name of the tensor.
        max_rows : int
            Maximum number of rows to display.
        max_cols : int
            Maximum number of columns to display.
        
        Returns
        -------
        str
            Formatted string showing tensor values.
        """
        shape = tensor.shape
        num_dims = len(shape)
        
        lines = []
        lines.append(f"\n{'─'*80}")
        lines.append(f"[IN] {name} (Input)")
        lines.append(f"{'─'*80}")
        lines.append(f"Shape: {shape}")
        lines.append(f"Device: {tensor.device}, Dtype: {tensor.dtype}")
        
        # Show tensor values
        if num_dims == 1:
            # 1D tensor: show first max_rows elements
            n_show = min(max_rows, shape[0])
            vals = tensor[:n_show].cpu().numpy()
            
            lines.append(f"\nFirst {n_show} elements:")
            lines.append(f"{'Index':<8} {'Value':<15}")
            lines.append(f"{'─'*30}")
            for i in range(n_show):
                lines.append(f"{i:<8} {vals[i]:<15.6f}")
            
            if shape[0] > n_show:
                lines.append(f"... ({shape[0] - n_show} more elements)")
        
        elif num_dims == 2:
            # 2D tensor: show first max_rows x max_cols
            n_rows = min(max_rows, shape[0])
            n_cols = min(max_cols, shape[1])
            
            slice_vals = tensor[:n_rows, :n_cols].cpu().numpy()
            
            lines.append(f"\nFirst {n_rows}×{n_cols} values:")
            for i in range(n_rows):
                row_str = " ".join(f"{val:8.4f}" for val in slice_vals[i])
                if n_cols < shape[1]:
                    row_str += " ..."
                lines.append(f"  [{i:2d}] {row_str}")
            
            if shape[0] > n_rows or shape[1] > n_cols:
                lines.append(f"\n... ({shape[0] - n_rows if shape[0] > n_rows else 0} more rows, "
                           f"{shape[1] - n_cols if shape[1] > n_cols else 0} more cols)")
        
        else:
            # Higher dimensional: flatten and show first max_rows*max_cols elements
            flat = tensor.flatten()[:max_rows * max_cols].cpu().numpy()
            
            lines.append(f"\nFirst {len(flat)} elements (flattened):")
            lines.append(f"{'Index':<8} {'Value':<15}")
            lines.append(f"{'─'*30}")
            for i in range(len(flat)):
                lines.append(f"{i:<8} {flat[i]:<15.6f}")
            
            total_elements = tensor.numel()
            if total_elements > len(flat):
                lines.append(f"... ({total_elements - len(flat)} more elements)")
        
        # Add statistics
        lines.append(f"\nStatistics:")
        lines.append(f"  min={tensor.min().item():.6f}, max={tensor.max().item():.6f}, "
                    f"mean={tensor.float().mean().item():.6f}, std={tensor.float().std().item():.6f}")
        
        lines.append(f"{'─'*80}")
        
        return "\n".join(lines)
    
    def assert_all_match(self, raise_on_fail: bool = True, debug: bool = False):
        """
        Assert that all comparisons match.
        
        Parameters
        ----------
        raise_on_fail : bool
            If True, raise AssertionError on mismatch. Otherwise, return False.
        debug : bool
            If True, print detailed visual comparison of intermediate tensors
            showing actual values (not just stats) for easy visual inspection.
            Also automatically shows input tensors (captured automatically from
            the models' forward calls).
        
        Returns
        -------
        bool
            True if all match, False otherwise.
        
        Example
        -------
        >>> # Inputs are automatically captured from forward calls
        >>> with scaffold.compare_forward():
        ...     pyg_out = pyg_model(data.x, edge_index)
        ...     relnn_out = relnn_model(relations)
        ... scaffold.assert_all_match(debug=True)  # Automatically shows inputs and outputs
        """
        if not self._forward_captured:
            warnings.warn("Forward pass not captured. Call compare_forward() first.")
            return True
        
        # Compare all hooks
        hook_results = self.compare_all_hooks()
        
        # If debug mode, print visual comparisons for all hooks
        if debug:
            print(f"\n{'='*80}")
            print("DEBUG MODE: Visual Tensor Comparison")
            print(f"{'='*80}")
            
            # First, show inputs if they were registered
            if self.pyg_inputs or self.relnn_inputs:
                print(f"\n{'='*80}")
                print("INPUT TENSORS")
                print(f"{'='*80}")
                
                # Show PyG inputs
                if self.pyg_inputs:
                    print("\n[PyG] Inputs:")
                    for input_name, input_value in self.pyg_inputs.items():
                        if isinstance(input_value, torch.Tensor):
                            print(self._format_single_tensor_debug(input_value, f"PyG.{input_name}"))
                        else:
                            print(f"\n  {input_name}: {type(input_value).__name__} (non-tensor, skipping visualization)")
                
                # Show RelNN inputs
                if self.relnn_inputs:
                    print("\n[RelNN] Inputs:")
                    for input_name, input_value in self.relnn_inputs.items():
                        if isinstance(input_value, torch.Tensor):
                            print(self._format_single_tensor_debug(input_value, f"RelNN.{input_name}"))
                        else:
                            print(f"\n  {input_name}: {type(input_value).__name__} (non-tensor, skipping visualization)")
            
            # Show weights comparison
            if self.weight_mapper.mapping:
                print(f"\n{'='*80}")
                print("MODEL WEIGHTS")
                print(f"{'='*80}")
                
                for relnn_name, pyg_name in sorted(self.weight_mapper.mapping.items()):
                    relnn_param = self.weight_mapper._relnn_params[relnn_name]
                    pyg_param = self.weight_mapper._pyg_params[pyg_name]
                    
                    # Create a shorter display name
                    display_name = relnn_name.split('.')[-1] if '.' in relnn_name else relnn_name
                    print(self._format_tensor_debug(
                        pyg_param.data, relnn_param.data, 
                        f"weight: {display_name}"
                    ))
            
            # Then show intermediate outputs (hook results)
            print(f"\n{'='*80}")
            print("INTERMEDIATE OUTPUTS (Hook Results)")
            print(f"{'='*80}")
            
            for name, result in hook_results.items():
                # Get the actual tensors
                if name in self.pyg_outputs and name in self.relnn_outputs:
                    pyg_output = self.pyg_outputs[name]
                    relnn_output = self.relnn_outputs[name]
                    
                    # Extract tensors if needed
                    if isinstance(pyg_output, torch.Tensor) and isinstance(relnn_output, torch.Tensor):
                        print(self._format_tensor_debug(pyg_output, relnn_output, name))
                    elif isinstance(pyg_output, dict) and isinstance(relnn_output, dict):
                        # Require exact same keys (parity with compare_outputs)
                        if set(pyg_output.keys()) != set(relnn_output.keys()):
                            print(f"\n[FAIL] {name}: Key mismatch -- PyG keys {sorted(pyg_output.keys())} vs RelNN keys {sorted(relnn_output.keys())}")
                        else:
                            for key in sorted(pyg_output.keys()):
                                pyg_val = pyg_output[key]
                                relnn_val = relnn_output[key]
                                if isinstance(pyg_val, torch.Tensor) and isinstance(relnn_val, torch.Tensor):
                                    print(self._format_tensor_debug(pyg_val, relnn_val, f"{name}.{key}"))
                    elif isinstance(pyg_output, (tuple, list)) and isinstance(relnn_output, (tuple, list)):
                        # Require same length (parity with compare_outputs)
                        if len(pyg_output) != len(relnn_output):
                            print(f"\n[FAIL] {name}: Length mismatch -- PyG {len(pyg_output)} vs RelNN {len(relnn_output)}")
                        else:
                            for i, (pyg_val, relnn_val) in enumerate(zip(pyg_output, relnn_output)):
                                if isinstance(pyg_val, torch.Tensor) and isinstance(relnn_val, torch.Tensor):
                                    print(self._format_tensor_debug(pyg_val, relnn_val, f"{name}[{i}]"))
                else:
                    # Hook not captured
                    status = "[OK]" if result.success else "[FAIL]"
                    print(f"\n{status} {name}: {result.message}")
                    if name not in self.pyg_outputs:
                        print(f"  [WARN] PyG output not captured")
                    if name not in self.relnn_outputs:
                        print(f"  [WARN] RelNN output not captured")
        
        # Check all results
        all_match = True
        failures = []
        
        for name, result in hook_results.items():
            if not result.success:
                all_match = False
                failures.append((name, result))
        
        if not all_match:
            error_msg = f"\n{'='*60}\n"
            error_msg += "SCAFFOLD COMPARISON FAILED\n"
            error_msg += f"{'='*60}\n"
            for name, result in failures:
                error_msg += f"\n[FAIL] {name}: {result.message}\n"
                if result.max_diff is not None:
                    error_msg += f"   Max diff: {result.max_diff:.6e}, Mean diff: {result.mean_diff:.6e}\n"
                if result.details:
                    error_msg += f"   Details: {result.details}\n"
            error_msg += f"\n{'='*60}\n"
            
            if raise_on_fail:
                raise AssertionError(error_msg)
            else:
                if self.verbose:
                    print(error_msg)
                return False
        
        if self.verbose:
            print(f"\n[OK] All comparisons passed ({len(hook_results)} checks)")
        
        return True
    
    def compare_weights(self) -> Dict[str, ComparisonResult]:
        """
        Compare weights between PyG and RelNN models without copying.
        
        This is useful when you want to verify that weights are the same
        (e.g., after training, or to check if they're already synchronized).
        
        **When to use**: 
        - To verify weights are already the same (without modifying them)
        - After training to check if both models learned similar weights
        - To compare weights at any point without side effects
        
        **When NOT to use**: 
        - If you want to ensure both models start with the same weights for
          a fair forward pass comparison, use `sync_weights()` instead.
        
        Returns
        -------
        Dict[str, ComparisonResult]
            Dictionary mapping parameter names to comparison results.
        """
        if not self.weight_mapper.mapping:
            warnings.warn(
                "No weight mappings found. Call auto_map_weights() or add_weight_mapping() first."
            )
            return {}
        
        results = {}
        
        for relnn_name, pyg_name in self.weight_mapper.mapping.items():
            relnn_param = self.weight_mapper._relnn_params[relnn_name]
            pyg_param = self.weight_mapper._pyg_params[pyg_name]
            
            result = self._compare_tensors(
                pyg_param.data, relnn_param.data,
                f"weight.{relnn_name}", use_permutation=False
            )
            results[relnn_name] = result
        
        return results
    
    def assert_weights_match(self, raise_on_fail: bool = True) -> bool:
        """
        Assert that all mapped weights match between PyG and RelNN models.
        
        This compares weights without modifying them. Use this to verify
        that weights are already synchronized or to check weights after training.
        
        Parameters
        ----------
        raise_on_fail : bool
            If True, raise AssertionError on mismatch. Otherwise, return False.
        
        Returns
        -------
        bool
            True if all weights match, False otherwise.
        """
        weight_results = self.compare_weights()
        
        if not weight_results:
            if self.verbose:
                print("No weight mappings found. Nothing to compare.")
            return True
        
        all_match = True
        failures = []
        
        for param_name, result in weight_results.items():
            if not result.success:
                all_match = False
                failures.append((param_name, result))
        
        if not all_match:
            error_msg = f"\n{'='*60}\n"
            error_msg += "WEIGHT COMPARISON FAILED\n"
            error_msg += f"{'='*60}\n"
            for param_name, result in failures:
                error_msg += f"\n[FAIL] {param_name}: {result.message}\n"
                if result.max_diff is not None:
                    error_msg += f"   Max diff: {result.max_diff:.6e}, Mean diff: {result.mean_diff:.6e}\n"
            error_msg += f"\n{'='*60}\n"
            
            if raise_on_fail:
                raise AssertionError(error_msg)
            else:
                if self.verbose:
                    print(error_msg)
                return False
        
        if self.verbose:
            print(f"\n[OK] All weights match ({len(weight_results)} parameters)")
        
        return True
    
    def compare_weights_after_training(self) -> Dict[str, ComparisonResult]:
        """
        Compare weights after training (useful for gradient checking).
        
        This is an alias for compare_weights() for backward compatibility.
        """
        return self.compare_weights()
    
    def cleanup(self):
        """Remove all hooks and clean up."""
        for hook in list(self.pyg_hooks.values()):
            hook.remove()
        for hook in list(self.relnn_hooks.values()):
            hook.remove()
        for hook in list(self.pyg_input_hooks.values()):
            hook.remove()
        for hook in list(self.relnn_input_hooks.values()):
            hook.remove()
        self.pyg_hooks.clear()
        self.relnn_hooks.clear()
        self.pyg_input_hooks.clear()
        self.relnn_input_hooks.clear()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False


def scaffold_decorator(
    pyg_model: nn.Module,
    relnn_model: nn.Module,
    weight_mappings: Optional[Dict[str, str]] = None,
    output_hooks: Optional[List[Tuple[str, str]]] = None,
    auto_map: bool = True,
    atol: float = 1e-5,
    rtol: float = 1e-5
):
    """
    Decorator to automatically scaffold a function that runs both models.
    
    Parameters
    ----------
    pyg_model : nn.Module
        PyG reference model.
    relnn_model : nn.Module
        RelNN model to compare.
    weight_mappings : dict, optional
        Manual weight mappings {relnn_path: pyg_path}.
    output_hooks : list, optional
        List of (pyg_path, relnn_path) tuples for output hooks.
    auto_map : bool
        Whether to automatically map weights by shape.
    atol, rtol : float
        Tolerances for comparison.
    
    Example
    -------
    >>> @scaffold_decorator(pyg_model, relnn_model, output_hooks=[("conv1", "ops.layer1")])
    ... def train_step():
    ...     pyg_out = pyg_model(x, edge_index)
    ...     relnn_out = relnn_model(relations)
    ...     return pyg_out, relnn_out
    """
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            scaffold = Scaffold(pyg_model, relnn_model, atol=atol, rtol=rtol)
            
            # Setup weight mappings
            if weight_mappings:
                for relnn_path, pyg_path in weight_mappings.items():
                    scaffold.add_weight_mapping(relnn_path, pyg_path)
            
            if auto_map:
                scaffold.auto_map_weights()
            
            scaffold.sync_weights(strict=not auto_map)
            
            # Setup output hooks
            if output_hooks:
                for pyg_path, relnn_path in output_hooks:
                    scaffold.add_output_hook(pyg_path, relnn_path)
            
            # Run function with scaffold
            with scaffold.compare_forward():
                result = func(*args, **kwargs)
            
            # Assert all match
            scaffold.assert_all_match()
            
            scaffold.cleanup()
            return result
        
        return wrapper
    return decorator

