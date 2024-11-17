from functools import partial
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class MLP(nn.Module):
    def __init__(self,
                 layers: List[int],
                 activation_fn: nn.Module = nn.ELU,
                 use_layer_norm: bool = False,
                 use_spectral_norm: bool = False,
                 activate_last: bool = True):
        super().__init__()
        
        modules = []
        for i in range(len(layers) - 1):
            layer = nn.Linear(layers[i], layers[i+1])

            if use_spectral_norm:
                layer = spectral_norm(layer)
                modules.append(layer)
            elif use_layer_norm:
                modules.append(layer)
                modules.append(nn.LayerNorm(layers[i+1]))
            else:
                modules.append(layer)

            if (i < len(layers) - 2 or activate_last):
                modules.append(activation_fn())

        self.layers = nn.Sequential(*modules)

    def forward(self, x):
        return self.layers(x)
    
    def init_params(self):
        init_fn = lambda m: init(m,
                                 weight_init=nn.init.orthogonal_,
                                 bias_init=lambda x: nn.init.constant_(x, 0),
                                 gain=np.sqrt(2))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init_fn(m)
    

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    
    def forward(self, x):
        return x


class TCN(nn.Module):
    """
    Temporal Convolutional Network
    """
    def __init__(self,
                 channels: List[int],
                 kernels: List[int],
                 strides: List[int],
                 activation_fn: nn.Module = nn.ELU,
                 use_layer_norm: bool = False,
                 use_spectral_norm: bool = False,
                 activate_last: bool = True):
        super().__init__()
        
        modules = []
        for i in range(len(channels) - 1):
            layer = nn.Conv1d(channels[i], channels[i+1], kernels[i], strides[i])
            
            if use_spectral_norm:
                layer = spectral_norm(layer)
                modules.append(layer)
            elif use_layer_norm:
                modules.append(layer)
                modules.append(nn.LayerNorm(channels[i+1]))
            else:
                modules.append(layer)
            
            if (i < len(channels) - 2 or activate_last):
                modules.append(activation_fn())
                
        self.layers = nn.Sequential(*modules)

    def forward(self, x):
        return self.layers(x)
    
    def init_params(self):
        init_fn = lambda m: init(m,
                                 weight_init=nn.init.orthogonal_,
                                 bias_init=lambda x: nn.init.constant_(x, 0),
                                 gain=np.sqrt(2))
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init_fn(m)
               
 