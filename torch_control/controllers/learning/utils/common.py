import math
from typing import Dict, Tuple, Union

import numpy as np
import torch
from gym import spaces
from torch.autograd import Function


class SimpleSafeGuard(Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = torch.where(torch.isnan(grad_output), torch.zeros_like(grad_output), grad_output)
        grad_output = torch.where(torch.isinf(grad_output), torch.zeros_like(grad_output), grad_output)
        return grad_output

# Usage:
safeguard = SimpleSafeGuard.apply


def get_obs_shape(
    observation_space: spaces.Space,
) -> Union[Tuple[int, ...], Dict[str, Tuple[int, ...]]]:
    """
    Get the shape of the observation (useful for the buffers).

    :param observation_space:
    :return:
    """
    if isinstance(observation_space, spaces.Box):
        return observation_space.shape
    elif isinstance(observation_space, spaces.Discrete):
        # Observation is an int
        return (1,)
    elif isinstance(observation_space, spaces.MultiDiscrete):
        # Number of discrete features
        return (int(len(observation_space.nvec)),)
    elif isinstance(observation_space, spaces.MultiBinary):
        # Number of binary features
        return observation_space.shape
    elif isinstance(observation_space, spaces.Dict):
        return {key: get_obs_shape(subspace) for (key, subspace) in observation_space.spaces.items()}  # type: ignore[misc]

    else:
        raise NotImplementedError(f"{observation_space} observation space is not supported")
    

def get_flattened_obs_dim(observation_space: spaces.Space) -> int:
    """
    Get the dimension of the observation space when flattened.
    It does not apply to image observation space.

    Used by the ``FlattenExtractor`` to compute the input shape.

    :param observation_space:
    :return:
    """
    # See issue https://github.com/openai/gym/issues/1915
    # it may be a problem for Dict/Tuple spaces too...
    if isinstance(observation_space, spaces.MultiDiscrete):
        return sum(observation_space.nvec)
    else:
        # Use Gym internal method
        return spaces.utils.flatdim(observation_space)
    

def get_action_dim(action_space: spaces.Space) -> int:
    """
    Get the dimension of the action space.

    :param action_space:
    :return:
    """
    if isinstance(action_space, spaces.Box):
        return np.prod(action_space.shape, dtype=np.int32)
    elif isinstance(action_space, spaces.Discrete):
        # Action is an int
        return 1
    elif isinstance(action_space, spaces.MultiDiscrete):
        # Number of discrete actions
        return int(len(action_space.nvec))
    elif isinstance(action_space, spaces.MultiBinary):
        # Number of binary actions
        assert isinstance(
            action_space.n, int
        ), "Multi-dimensional MultiBinary action space is not supported. You can flatten it instead."
        return int(action_space.n)
    else:
        raise NotImplementedError(f"{action_space} action space is not supported")


def get_grad_norm(it):
    """Compute the L2 norm of gradients for a given iterator of parameters."""
    sum_grad = 0
    for x in it:
        if x.grad is not None:
            sum_grad += x.grad.norm().item() ** 2
    return torch.sqrt(torch.tensor(sum_grad))


def get_param_norm(it):
    """Compute the L2 norm of parameters for a given iterator of parameters."""
    sum_param = 0
    for x in it:
        sum_param += x.norm().item() ** 2
    return torch.sqrt(torch.tensor(sum_param))


def gaussian_pdf(x, mean, std):
    """Compute the probability density function of a Gaussian distribution."""
    return (1 / (math.sqrt(2 * math.pi) * std)) * torch.exp(-(x - mean) ** 2 / (2 * std ** 2))


def gaussian2d_pdf(x, mean, std):
    """Compute the probability density function of a 2D Gaussian distribution."""
    return (
        1
        / (2 * math.pi * std[0] * std[1])
        * torch.exp(
            -(
                ((x[0] - mean[0]) / std[0]) ** 2 + ((x[1] - mean[1]) / std[1]) ** 2
            )
            / 2
        )
    )


def huber_loss(err, delta=1.0):
    """Compute the Huber loss."""
    case_a = (abs(err) <= delta).to(torch.float32)
    case_b = (abs(err) > delta).to(torch.float32)
    
    return case_a * 0.5 * err ** 2 + case_b * delta * (abs(err) - 0.5 * delta)


def rescale_actions(action, low, high):
    action = torch.clamp(action, -1, 1)
    d = (high - low) / 2.0
    m = (high + low) / 2.0
    scaled_action = action * d + m
    return scaled_action


class AffineShaper:
    def __init__(self, scale=1, shift=0, min_value=-np.inf, max_value=np.inf):
        self.scale = scale
        self.shift = shift
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, x):
        return torch.clamp(x * self.scale + self.shift, self.min_value, self.max_value)
    

class EpisodeData:
    def __init__(self):
        # Initialize an empty dict to store tensors
        self.data = {}

    def add_step(self, step_data):
        """Add a step of data. step_data should be a dict where keys are data keys 
        and values are PyTorch tensors."""

        for key, value in step_data.items():
            if key not in self.data:
                # This is the first data for this key. 
                # We need to add a new dimension to stack data along this dimension.
                self.data[key] = value.unsqueeze(1).detach()
            else:
                # This key has previously stored data, so we stack along the first dimension.
                self.data[key] = torch.cat((self.data[key], value.unsqueeze(1).detach()), dim=1)

    def get_data(self, indices, reduce='none'):
        """Return the data."""
        if len(self.data) == 0:
            return {}
        
        reduce_fn = {'mean': torch.mean, 'sum': torch.sum, 'none': lambda x: x}[reduce]
        return {key: reduce_fn(val, dim=1)[indices] for key, val in self.data.items()}

    def reset(self, indices):
        """Reset the data for the environments at the given indices."""
        if len(self.data) == 0:
            return
        
        for key in self.data.keys():
            self.data[key][indices] = 0
    
    def __getitem__(self, indices):
        """Return a new dict with data at the given indices. 
        Indices should be a list or a 1-D tensor."""
        indices = torch.tensor(indices)  # Ensure indices is a tensor
        return {key: val[indices] for key, val in self.data.items()}