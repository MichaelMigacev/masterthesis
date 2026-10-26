import inspect
from abc import abstractmethod
from typing import Callable, Iterable, Type, Union

import torch
from tensordict import TensorDict

from domip.base_classes.util import collect_args, gather_partial_derivatives, collect_shapes


class Goal:

    name: str
    is_active: bool
    g_func: Callable
    fulfilled_value : Union[float, None]

    def __init__(self, name: str, is_active:bool=True, dtype:Union[torch.dtype, None] = None, device : Union[torch.device, None] = None, mockbuild : bool = False, fulfilled_value : Union[float, None] = None):
        self.device = device
        self.dtype = dtype
        self.name = name
        self.is_active = is_active
        self.fulfilled_value = fulfilled_value
        if not mockbuild:
            self.g_func = self.define_goal_cost_function()

    @abstractmethod
    def define_goal_cost_function(self):
        pass

    def __str__(self):
        return self.name


def evaluate_goals(goals: Iterable[Type[Goal]], output_dict):
    partial_derivatives = TensorDict({}, batch_size=())
    goal_values = TensorDict({}, batch_size=())
    for g in goals:
        if g.is_active:
            args, diff_argnums, input_names_diff = collect_args(output_dict, inspect.getfullargspec(g.g_func).args, None, [g.name])
            partial_jacs, current_cost_value = torch.func.jacfwd(g.g_func, argnums=diff_argnums,
                                                                                    has_aux=True, randomness="same")(*args)
            in_out_shapes = collect_shapes(output_dict, {g.name: current_cost_value})
            partial_derivatives = partial_derivatives.update(gather_partial_derivatives(partial_jacs, input_names_diff, [g.name], in_out_shapes))
            goal_values[g.name] = current_cost_value
    return partial_derivatives, goal_values


def visualize_goal_derivatives(goal_derivatives, component_name):
    """
    Visualizes all goal derivatives as heatmaps.
    
    Args:
        goal_derivatives: TensorDict containing the goal derivatives.
        component_name: Name of the component for display purposes.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    #if len(goal_derivatives) == 0:
    #    return
    
    # Collect all leaf tensors
    tensors = {}
    for key, tensor in goal_derivatives.items(include_nested=True, leaves_only=True):
        if isinstance(tensor, torch.Tensor) and tensor.numel() > 0:
            tensors[str(key)] = tensor
    
    if not tensors:
        return
    
    num_quantities = len(tensors)
    fig, axes = plt.subplots(1, num_quantities, figsize=(6 * num_quantities, 5))
    
    # Handle case where there's only one quantity
    if num_quantities == 1:
        axes = [axes]
    
    for idx, (quantity_name, tensor) in enumerate(tensors.items()):
        # Handle different tensor shapes - visualize the first batch element and first two dimensions if available
        data = tensor.detach().cpu().numpy()
        
        # Extract 2D slice for visualization
        if len(data.shape) >= 2:
            if len(data.shape) > 2:
                # Take first batch element and first 2D slice
                if len(data.shape) == 4:
                    # Assuming shape is (batch, channels, from_dim, to_dim)
                    viz_data = data[0, 0, :, :]
                    print(f"Visualizing 2D slice of tensor for quantity '{quantity_name}' with original shape {data.shape} and visualized shape {viz_data.shape}")
                elif len(data.shape) == 3:
                    # Assuming shape is (batch, from_dim, to_dim)
                    viz_data = data[0, :, :]
                elif len(data.shape) == 5:
                    viz_data = data[0, 0, 0, :, :]  # Take first batch and first channel, keep remaining dimensions
                    print(f"Visualizing 2D tensor for quantity '{quantity_name}' with shape {data.shape}")
                else:
                    viz_data = data
                    print(f"Visualizing 2D tensor for quantity '{quantity_name}' with shape {data.shape}")
                print(f"Vizdata shape: {viz_data.shape}")
            else:
                viz_data = data
            
            ax = axes[idx]
            im = ax.imshow(viz_data, cmap='coolwarm', aspect='equal', 
                           vmin=-np.max(np.abs(viz_data)), vmax=np.max(np.abs(viz_data)))
            cbar = plt.colorbar(im, ax=ax)
            cbar.ax.tick_params(labelsize=8)
            ax.set_title(f'{quantity_name}', fontsize=11, weight='bold')
            ax.set_xlabel('To Dimension', fontsize=8)
            ax.set_ylabel('From Dimension', fontsize=8)
            ax.tick_params(axis='both', which='major', labelsize=7)
            ax.grid(False)
            
            # Add text annotations
            for i in range(viz_data.shape[0]):
                for j in range(viz_data.shape[1]):
                    val = viz_data[i, j]
                    text_color = 'white' if abs(val) > np.max(np.abs(viz_data)) / 2 else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=text_color, fontsize=10, weight='bold')
    
    plt.suptitle(f'Goal Derivatives - {component_name}', fontsize=12, weight='bold')
    plt.tight_layout()
    plt.show()
