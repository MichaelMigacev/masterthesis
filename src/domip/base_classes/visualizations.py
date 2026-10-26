import inspect
import time
import traceback
from abc import abstractmethod
from threading import Lock
from typing import Dict, Type, List, Callable, Tuple, Union

import torch
from tensordict import TensorDict

from domip.base_classes.goals import visualize_goal_derivatives

from domip.base_classes.connections import Connection
from domip.base_classes.derivatives import DerivativeDict, compute_derivatives, \
    get_steepest_gradient_from_derivative_block
from domip.base_classes.goals import evaluate_goals, Goal
from domip.base_classes.util import collect_args, collect_inputs, collect_derivatives, gather_partial_derivatives, \
    generate_outputdict, collect_shapes, collect_required_signs
from domip.tower_of_london_direct.global_params import NUM_BLOCKS

import matplotlib.pyplot as plt
import numpy as np

import datetime


def visualize_component_derivatives_external(backward_derivatives, component_name):
    """
    Visualizes all backward derivatives in the component as heatmaps.
    
    Args:
        backward_derivatives: DerivativeDict containing the backward derivatives.
        component_name: Name of the component for display purposes.
    """
    
    if len(backward_derivatives.block_dict) == 0:
        return
    
    # Determine maximum number of traces across all quantities
    max_traces = 0
    for block in backward_derivatives.block_dict.values():
        if hasattr(block, 'quantity_traces') and block.quantity_traces is not None:
            max_traces = max(max_traces, len(block.quantity_traces))
    
    if max_traces == 0:
        print("No quantity_traces found in backward_derivatives blocks")
        return
    
    num_quantities = len(backward_derivatives.block_dict)
    fig, axes = plt.subplots(max_traces, num_quantities, figsize=(6 * num_quantities, 5 * max_traces))
    # coerce into ndarray with the right shape
    axes = np.array(axes)
    try:
        axes = axes.reshape(max_traces, num_quantities)
    except Exception:
        axes = np.array([[axes]])
    
    for col_idx, (quantity_name, block) in enumerate(backward_derivatives.block_dict.items()):
        tensor = block.derivatives_tensor
        
        if tensor is None or tensor.numel() == 0:
            continue
        
        data = tensor.detach().cpu().numpy()
        traces = block.quantity_traces if hasattr(block, 'quantity_traces') else []
        
        # Process each trace
        for row_idx, trace in enumerate(traces):
            # Extract 2D data for this trace from the first dimension
            if len(data.shape) == 3:
                # Shape: (num_traces, from_dim, to_dim)
                viz_data = data[row_idx, :, :]
            elif len(data.shape) == 4:
                # Shape: (num_traces, channels, from_dim, to_dim)
                viz_data = data[row_idx, 0, :, :]
            elif len(data.shape) == 5:
                # Shape: (num_traces, ?, ?, from_dim, to_dim)
                viz_data = data[row_idx, 0, 0, :, :]
            else:
                # Fallback: try to get last two dimensions
                try:
                    viz_data = data[row_idx, ..., -2:]
                except:
                    continue
            
            if viz_data.size == 0:
                continue
            
            ax = axes[row_idx, col_idx]
            
            # Create heatmap
            im = ax.imshow(viz_data, cmap='coolwarm', aspect='equal',
                            vmin=-np.max(np.abs(viz_data)), vmax=np.max(np.abs(viz_data)))
            cbar = plt.colorbar(im, ax=ax)
            cbar.ax.tick_params(labelsize=8)
            
            # Create title with quantity name and trace
            trace_label = ' -> '.join(trace).replace('likelihood_', '')
            title = f'{quantity_name}\n{trace_label}'
            ax.set_title(title, fontsize=10, weight='bold')
            ax.set_xlabel('To Dimension', fontsize=8)
            ax.set_ylabel('From Dimension', fontsize=8)
            ax.tick_params(axis='both', which='major', labelsize=7)
            ax.grid(False)
            
            # Add text annotations only if the matrix is small enough
            if viz_data.shape[0] <= 10 and viz_data.shape[1] <= 10:
                for i in range(viz_data.shape[0]):
                    for j in range(viz_data.shape[1]):
                        val = viz_data[i, j]
                        text_color = 'white' if abs(val) > np.max(np.abs(viz_data)) / 2 else 'black'
                        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=text_color, fontsize=9, weight='bold')
    
    plt.suptitle(f'Backward Derivatives - {component_name}', fontsize=12, weight='bold')
    plt.tight_layout()
    plt.show()


def visualize_partial_derivatives_between_external(
        partial_derivatives,
        component_name,
        from_class: str,
        to_class: str,
    second_input_shape: Union[None, int, Tuple[int, ...]] = None,
    first_input_shape: Union[None, int, Tuple[int, ...]] = None,
    max_elements: int = 24,
    compact: bool = True,
    annotate: bool = True):
    """
    Visualizes partial derivatives for one selected pair of classes.

    This function is manual-shape only: you must provide both first_input_shape
    and second_input_shape explicitly.

    For each element in the second input (to_class), this plots the derivative
    of the full first input (from_class) w.r.t. that single second-input element.

    Args:
        partial_derivatives: TensorDict with nested partial derivative tensors.
        component_name: Name of the component for display purposes.
        from_class: First key element (e.g. 'likelihood_slots').
        to_class: Second key element (e.g. 'likelihood_below_valid').
        second_input_shape: Explicit shape of to_class input. Examples: 6, (6,), (6, 3).
        first_input_shape: Explicit shape of from_class input. Examples: 6, (6,), (6, 3).
        max_elements: Maximum number of second-input elements to plot.
        compact: If True, use smaller subplots and tighter spacing.
        annotate: If True, annotate values for small maps.
    """

    leaf_items = list(partial_derivatives.items(include_nested=True, leaves_only=True))
    matching_tensor = None
    matching_key = None

    for key_path, tensor in leaf_items:
        if not isinstance(key_path, tuple) or len(key_path) < 2:
            continue
        if str(key_path[0]) == from_class and str(key_path[1]) == to_class:
            matching_key = key_path
            matching_tensor = tensor
            break

    if matching_tensor is None:
        available_pairs = sorted({
            (str(k[0]), str(k[1]))
            for k, _ in leaf_items
            if isinstance(k, tuple) and len(k) >= 2
        })
        print(f"No partial derivative found for pair: ({from_class}, {to_class})")
        print("Available pairs:")
        for pair in available_pairs:
            print(pair)
        return

    if not isinstance(matching_tensor, torch.Tensor) or matching_tensor.numel() == 0:
        print(f"Partial derivative tensor for ({from_class}, {to_class}) is empty")
        return

    data = matching_tensor.detach().cpu().numpy()
    if data.ndim < 2:
        print(f"Tensor for ({from_class}, {to_class}) has ndim={data.ndim}; need at least 2")
        return

    tensor_shape = tuple(data.shape)

    def _normalize_required_shape(shape_arg, name):
        if shape_arg is None:
            print(f"{name} is required and must be provided explicitly")
            return None
        if isinstance(shape_arg, int):
            shape_arg = (shape_arg,)
        else:
            shape_arg = tuple(shape_arg)
        if len(shape_arg) == 0:
            print(f"{name} must have at least one dimension")
            return None
        if any((not isinstance(d, int)) or d <= 0 for d in shape_arg):
            print(f"{name} must contain only positive integers, got {shape_arg}")
            return None
        return shape_arg

    first_shape = _normalize_required_shape(first_input_shape, "first_input_shape")
    second_shape = _normalize_required_shape(second_input_shape, "second_input_shape")
    if first_shape is None or second_shape is None:
        return

    expected_shape = tuple(first_shape) + tuple(second_shape)
    if tensor_shape != expected_shape:
        print(
            f"Shape mismatch for ({from_class}, {to_class}): tensor shape is {tensor_shape}, "
            f"but expected first_input_shape + second_input_shape = {expected_shape}."
        )
        return

    second_indices = list(np.ndindex(second_shape))
    if len(second_indices) > max_elements:
        print(f"Showing first {max_elements} of {len(second_indices)} second-input elements")
        second_indices = second_indices[:max_elements]

    plots = []
    for s_idx in second_indices:
        # Keep full first-input shape, fix one element in second input.
        # Example: data[first..., second...] -> slice[first...]
        selector = (slice(None),) * len(first_shape) + s_idx
        viz_data = data[selector]
        plots.append((s_idx, viz_data))

    if len(plots) == 0:
        print(f"No plottable slices for ({from_class}, {to_class})")
        return

    n_plots = len(plots)
    ncols = min(4 if compact else 3, n_plots)
    nrows = int(np.ceil(n_plots / ncols))
    width_per_col = 3.2 if compact else 5.0
    height_per_row = 2.8 if compact else 4.2
    fig, axes = plt.subplots(nrows, ncols, figsize=(width_per_col * ncols, height_per_row * nrows))
    axes = np.array(axes).reshape(nrows, ncols)

    def _format_rough_value(val: float) -> str:
        abs_val = abs(val)
        if abs_val == 0.0:
            return "0"
        if abs_val >= 1e2 or abs_val < 1e-2:
            return f"{val:.1e}"
        return f"{val:.2f}"

    for p_idx, (idx, viz_data) in enumerate(plots):
        row_idx = p_idx // ncols
        col_idx = p_idx % ncols
        ax = axes[row_idx, col_idx]

        # Ensure 2D rendering: keep full first input, but adapt 1D/ND for imshow.
        if viz_data.ndim == 1:
            display_data = viz_data[np.newaxis, :]
        elif viz_data.ndim == 2:
            display_data = viz_data
        else:
            # Flatten higher first-input dims into rows for compact display.
            display_data = viz_data.reshape(int(np.prod(viz_data.shape[:-1])), viz_data.shape[-1])

        abs_max = np.max(np.abs(display_data)) if display_data.size > 0 else 0.0
        if abs_max == 0.0:
            abs_max = 1.0

        im = ax.imshow(display_data, cmap='coolwarm', aspect='auto', vmin=-abs_max, vmax=abs_max)
        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.tick_params(labelsize=6 if compact else 8)

        title = f"d{from_class}/d{to_class}[{idx}]"
        ax.set_title(title, fontsize=8 if compact else 10, weight='bold')
        ax.set_xlabel('From Input - Last Dim', fontsize=6 if compact else 8)
        if viz_data.ndim <= 2:
            ax.set_ylabel('From Input - First Dim', fontsize=6 if compact else 8)
        else:
            ax.set_ylabel('Flattened From Dims', fontsize=6 if compact else 8)
        ax.tick_params(axis='both', which='major', labelsize=6 if compact else 7)
        ax.grid(False)

        if annotate and display_data.shape[0] <= 10 and display_data.shape[1] <= 10:
            for i in range(display_data.shape[0]):
                for j in range(display_data.shape[1]):
                    val = display_data[i, j]
                    text_color = 'white' if abs(val) > abs_max / 2 else 'black'
                    ax.text(j, i, _format_rough_value(val), ha='center', va='center', color=text_color,
                            fontsize=6 if compact else 8, weight='bold')

    for empty_idx in range(n_plots, nrows * ncols):
        row_idx = empty_idx // ncols
        col_idx = empty_idx % ncols
        axes[row_idx, col_idx].axis('off')

    pair_label = f"{from_class} -> {to_class}"
    key_label = " -> ".join([str(k) for k in matching_key]) if isinstance(matching_key, tuple) else str(matching_key)
    plt.suptitle(
        f'Partial Derivatives ({pair_label}) - {component_name}\n'
        f'key={key_label}, first_shape={first_shape}, second_shape={second_shape}',
        fontsize=10 if compact else 12,
        weight='bold'
    )
    plt.tight_layout(pad=0.6 if compact else 1.0)
    plt.show()


def collect_partial_derivative_chain_by_pairs(partial_derivatives, derivative_pairs: List[Tuple[str, str]]):
    """
    Collects a list of partial derivative tensors from TensorDict using (from_class, to_class) pairs.

    Args:
        partial_derivatives: TensorDict with nested partial derivative tensors.
        derivative_pairs: List of tuples like [('likelihood_slots', 'likelihood_free'), ...].

    Returns:
        Tuple[List[torch.Tensor], List[Tuple]]:
            - resolved tensors in the requested order
            - resolved key paths in the requested order
    """
    leaf_items = list(partial_derivatives.items(include_nested=True, leaves_only=True))
    tensors = []
    resolved_keys = []

    for from_class, to_class in derivative_pairs:
        matching_tensor = None
        matching_key = None
        for key_path, tensor in leaf_items:
            if not isinstance(key_path, tuple) or len(key_path) < 2:
                continue
            if str(key_path[0]) == str(from_class) and str(key_path[1]) == str(to_class):
                matching_key = key_path
                matching_tensor = tensor
                break

        if matching_tensor is None:
            raise KeyError(f"Could not find partial derivative for pair ({from_class}, {to_class})")
        if not isinstance(matching_tensor, torch.Tensor):
            raise TypeError(f"Resolved entry for pair ({from_class}, {to_class}) is not a torch.Tensor")

        tensors.append(matching_tensor)
        resolved_keys.append(matching_key)

    return tensors, resolved_keys


def multiply_chain_derivatives(goal_derivative: torch.Tensor,
                               partial_derivative_chain: List[torch.Tensor],
                               keep_batch_dims: int = 1,
                               return_intermediate: bool = False):
    """
    Applies chain-rule tensor multiplication across a sequence of partial derivatives.

    Contract rule per step:
        current shape = (batch..., out_dims)
        partial shape = (out_dims, in_dims)
        result shape  = (batch..., in_dims)

    The number of contracted dims is inferred automatically by matching the largest
    suffix of current out_dims with the prefix of the partial derivative.

    Args:
        goal_derivative: First derivative tensor (e.g. shape (1, 6, 3)).
        partial_derivative_chain: List of partial derivative tensors in chain order.
        keep_batch_dims: Number of leading dims in current tensor treated as batch dims.
        return_intermediate: If True, also return intermediate tensors.

    Returns:
        If return_intermediate=False:
            torch.Tensor final chained derivative.
        If return_intermediate=True:
            Tuple[torch.Tensor, List[torch.Tensor]] where list includes start + each step.
    """
    if not isinstance(goal_derivative, torch.Tensor):
        raise TypeError("goal_derivative must be a torch.Tensor")
    if keep_batch_dims < 0:
        raise ValueError("keep_batch_dims must be >= 0")

    current = goal_derivative
    steps = [current]

    for step_idx, partial in enumerate(partial_derivative_chain):
        if not isinstance(partial, torch.Tensor):
            raise TypeError(f"partial_derivative_chain[{step_idx}] must be a torch.Tensor")

        if current.ndim < keep_batch_dims:
            raise ValueError(
                f"Current tensor at step {step_idx} has ndim={current.ndim}, "
                f"which is smaller than keep_batch_dims={keep_batch_dims}"
            )

        current_out_shape = tuple(current.shape[keep_batch_dims:])
        partial_shape = tuple(partial.shape)
        if len(current_out_shape) == 0:
            raise ValueError(
                f"Current tensor at step {step_idx} has no non-batch dims left to contract. "
                f"Current shape: {tuple(current.shape)}"
            )

        max_contract = min(len(current_out_shape), len(partial_shape))
        contract_dims = None
        for n_contract in range(max_contract, 0, -1):
            if current_out_shape[-n_contract:] == partial_shape[:n_contract]:
                contract_dims = n_contract
                break

        if contract_dims is None:
            raise ValueError(
                f"Could not infer contraction at step {step_idx}. "
                f"Current non-batch shape={current_out_shape}, partial shape={partial_shape}."
            )

        current_contract_axes = list(range(current.ndim - contract_dims, current.ndim))
        partial_contract_axes = list(range(contract_dims))
        current = torch.tensordot(current, partial, dims=(current_contract_axes, partial_contract_axes))
        steps.append(current)

    if return_intermediate:
        return current, steps
    return current


def visualize_chain_derivative_product_external(goal_derivative: torch.Tensor,
                                                partial_derivative_chain: List[torch.Tensor],
                                                component_name: str,
                                                keep_batch_dims: int = 1,
                                                compact: bool = True,
                                                annotate: bool = True,
                                                show_intermediate: bool = False,
                                                step_labels: Union[None, List[str]] = None,
                                                show_partials_in_same_figure: bool = True,
                                                unflatten_border_every: int = 3):
    """
    Multiplies goal derivative with a chain of partial derivatives and visualizes result(s).

    Args:
        goal_derivative: Initial derivative tensor, e.g. (1, 6, 3).
        partial_derivative_chain: Partial derivatives in chain order.
        component_name: Name shown in plot title.
        keep_batch_dims: Number of leading dims treated as batch dims during multiplication.
        compact: If True, smaller figure size.
        annotate: If True, show numeric values for small heatmaps.
        show_intermediate: If True, plots each intermediate chained result.
        step_labels: Optional labels for steps when show_intermediate=True.
        show_partials_in_same_figure: If True, includes goal + all selected partial
            derivative factors in the same figure as the chain product result(s).
        unflatten_border_every: Draws thicker borders every N cells on partial
            derivative panels to make flattened dimensions easier to parse.
    """
    try:
        final_tensor, steps = multiply_chain_derivatives(
            goal_derivative=goal_derivative,
            partial_derivative_chain=partial_derivative_chain,
            keep_batch_dims=keep_batch_dims,
            return_intermediate=True,
        )
    except Exception:
        print("Could not compute chained derivative product:")
        traceback.print_exc()
        return

    def _to_display_data(tensor_data):
        if tensor_data.ndim == 0:
            return tensor_data.reshape(1, 1)
        if tensor_data.ndim == 1:
            return tensor_data[np.newaxis, :]
        if tensor_data.ndim == 2:
            return tensor_data
        return tensor_data.reshape(int(np.prod(tensor_data.shape[:-1])), tensor_data.shape[-1])

    if show_intermediate:
        product_tensors = steps[1:] if len(steps) > 1 else [final_tensor]
    else:
        product_tensors = [final_tensor]

    if show_intermediate and step_labels is not None and len(step_labels) == len(steps):
        product_labels = [f"product:{lbl}" for lbl in step_labels[1:]]
    elif show_intermediate:
        product_labels = [f"product step {i + 1}" for i in range(len(product_tensors))]
    else:
        product_labels = ["product:final"]

    factor_tensors = [goal_derivative] + list(partial_derivative_chain)
    factor_labels = ["factor:goal"] + [f"factor:partial {i + 1}" for i in range(len(partial_derivative_chain))]

    if show_partials_in_same_figure:
        ncols = max(len(factor_tensors), len(product_tensors))
        nrows = 2
    else:
        n_plots = len(product_tensors)
        ncols = min(4 if compact else 3, n_plots)
        nrows = int(np.ceil(n_plots / ncols))

    width_per_col = 3.2 if compact else 5.0
    height_per_row = 2.8 if compact else 4.2

    fig, axes = plt.subplots(nrows, ncols, figsize=(width_per_col * ncols, height_per_row * nrows))
    axes = np.array(axes).reshape(nrows, ncols)

    def _format_value(val: float) -> str:
        abs_val = abs(val)
        if abs_val == 0.0:
            return "0"
        if abs_val >= 1e2 or abs_val < 1e-2:
            return f"{val:.1e}"
        return f"{val:.2f}"

    def _draw_panel(ax, tensor, title_label, draw_unflatten_borders=False):
        data = tensor.detach().cpu().numpy()
        display_data = _to_display_data(data)

        abs_max = np.max(np.abs(display_data)) if display_data.size > 0 else 0.0
        if abs_max == 0.0:
            abs_max = 1.0

        im = ax.imshow(display_data, cmap='coolwarm', aspect='auto', vmin=-abs_max, vmax=abs_max)
        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.tick_params(labelsize=6 if compact else 8)

        if draw_unflatten_borders:
            n_rows, n_cols = display_data.shape

            # Thin cell borders for easier readability.
            for r in range(n_rows + 1):
                ax.axhline(r - 0.5, color='white', linewidth=0.35, alpha=0.4, zorder=3)
            for c in range(n_cols + 1):
                ax.axvline(c - 0.5, color='white', linewidth=0.35, alpha=0.4, zorder=3)

            # Thicker separators every `unflatten_border_every` rows (flattened axis).
            if isinstance(unflatten_border_every, int) and unflatten_border_every > 0:
                for r in range(0, n_rows + 1, unflatten_border_every):
                    ax.axhline(r - 0.5, color='black', linewidth=1.4, alpha=0.75, zorder=4)

        ax.set_title(f"{title_label}\nshape={tuple(tensor.shape)}", fontsize=8 if compact else 10, weight='bold')
        ax.set_xlabel('Last Dimension', fontsize=6 if compact else 8)
        ax.set_ylabel('Flattened Leading Dims', fontsize=6 if compact else 8)
        ax.tick_params(axis='both', which='major', labelsize=6 if compact else 7)
        ax.grid(False)

        if annotate and display_data.shape[0] <= 10 and display_data.shape[1] <= 10:
            for i in range(display_data.shape[0]):
                for j in range(display_data.shape[1]):
                    val = display_data[i, j]
                    text_color = 'white' if abs(val) > abs_max / 2 else 'black'
                    ax.text(j, i, _format_value(val), ha='center', va='center', color=text_color,
                            fontsize=6 if compact else 8, weight='bold')

    if show_partials_in_same_figure:
        for col_idx, (label, tensor) in enumerate(zip(factor_labels, factor_tensors)):
            _draw_panel(
                axes[0, col_idx],
                tensor,
                label,
                draw_unflatten_borders=str(label).startswith("factor:partial"),
            )
        for col_idx in range(len(factor_tensors), ncols):
            axes[0, col_idx].axis('off')

        for col_idx, (label, tensor) in enumerate(zip(product_labels, product_tensors)):
            _draw_panel(axes[1, col_idx], tensor, label)
        for col_idx in range(len(product_tensors), ncols):
            axes[1, col_idx].axis('off')
    else:
        flat_axes = axes.reshape(-1)
        for idx, (label, tensor) in enumerate(zip(product_labels, product_tensors)):
            _draw_panel(flat_axes[idx], tensor, label)
        for idx in range(len(product_tensors), len(flat_axes)):
            flat_axes[idx].axis('off')

    plt.suptitle(
        f'Chain Rule Derivative Product - {component_name}',
        fontsize=10 if compact else 12,
        weight='bold'
    )
    plt.tight_layout(pad=0.6 if compact else 1.0)
    plt.show()