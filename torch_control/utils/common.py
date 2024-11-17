import os
import random

import numpy as np
import torch

from omegaconf import OmegaConf, DictConfig

class HistoryQueue:
    def __init__(self, max_len, num_parallels, data_size, device: str='cuda', dt: float=0.02):
        # check if data_size is iterable
        if not hasattr(data_size, '__iter__'):
            data_size = [int(data_size)]
        self._queue = torch.zeros(
            max_len, num_parallels, *data_size, dtype=torch.float32, device=device)
        self.dt = dt
    
    def update(self, data):
        pop_data = self._queue[0]
        self._queue = torch.cat([self._queue[1:], data.unsqueeze(0)], dim=0)
        return pop_data

    @torch.no_grad()
    def tail(self, tail_len: int, include_diff: bool = False, include_int: bool = False):
        tail_data = [self._queue[-tail_len:]]
        if include_diff:
            tail_data.append(torch.diff(self._queue[-tail_len-1:], dim=0) / self.dt)
        if include_int:
            tail_data.append(torch.cumsum(self._queue[-tail_len:], dim=0))
        return torch.cat(tail_data, dim=-1)
    
    def reset(self, reset_idx=None, init_data=None):
        if reset_idx is None:
            reset_idx = torch.arange(self._queue.shape[1])
        self._queue[:, reset_idx] = 0.
        if init_data is not None:
            # init queue with init_data for all positions in reset_idx
            ndim = init_data.ndim
            self._queue[:, reset_idx] = init_data.unsqueeze(0).repeat(self._queue.shape[0], *[1] * ndim)

    def diff(self):
        return torch.diff(self._queue, dim=0) # (max_len-1, num_parallels, *data_size)

    def integrate(self, axis=0):
        return torch.cumsum(self._queue, dim=axis) # (max_len, num_parallels, *data_size)

    @torch.no_grad()
    def clear_grad(self):
        self._queue = self._queue.clone()

def set_all_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print("[Info] Setting all random seeds to {}".format(seed))


def unflatten_config(config_dict):
    """
    Revised version of unflatten_config to handle cases where the values are dictionaries with 'desc' and 'value' keys.
    The function extracts the 'value' field and ignores the 'desc' field.
    """
    unflattened_config = {}

    for key, value in config_dict.items():
        if key in ["wandb_version", "_wandb"]:  # Skip wandb-specific keys
            continue

        parts = key.split('.')
        d = unflattened_config
        for part in parts[:-1]:
            if part not in d or not isinstance(d[part], dict):
                d[part] = {}
            d = d[part]
        
        # Extract the 'value' field if it exists, otherwise use the original value
        if isinstance(value, (dict, DictConfig)) and 'value' in value:
            d[parts[-1]] = value['value']
        else:
            d[parts[-1]] = value

    return unflattened_config

def load_config_from_wandb(wandb_dir):
    wandb_cfg = OmegaConf.load(os.path.join(wandb_dir, "config.yaml"))
    ckpt_dir = os.path.join(wandb_dir, "checkpoints")
    
    wandb_cfg = unflatten_config(wandb_cfg)
    cfg = OmegaConf.create(wandb_cfg)
    cfg.controller.ckpt_dir = ckpt_dir
    
    return cfg

def nan_alarm(x, trigger_pdb: bool = True):
    if torch.isnan(x).any():
        # print x's name
        print("[Warning] NaN detected in tensor, triggering debugger...")
        if trigger_pdb:
            import pdb; pdb.set_trace()
        exit(1)