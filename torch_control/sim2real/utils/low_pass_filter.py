from typing import Dict, Union

import copy
import torch
import torch.nn as nn

# Reference: https://en.wikipedia.org/wiki/Low-pass_filter

class LowPassFilter:
    def __init__(self, cutoff_freq: float, dt: float):
        rc = 1. / (2. * torch.pi * cutoff_freq)

        self.alpha = dt / (dt + rc)
        self.record = None

    def update(self, input_data: Union[torch.Tensor, Dict[str, torch.Tensor]]):
        if self.record is None:
            self.record = copy.copy(input_data)
        else:
            if isinstance(input_data, torch.Tensor):
                self.record.lerp_(input_data.to(self.record.device), self.alpha)
            elif isinstance(input_data, dict):
                for key, value in self.record.items():
                    self.record[key].lerp_(input_data[key].to(value.device),
                                           self.alpha)
            else:
                raise ValueError
            
        return self.record

    def filter(self, input_data):
        if self.record is None:
            return input_data
        
        if isinstance(input_data, torch.Tensor):
            return (1 - self.alpha) * self.record + self.alpha * input_data.to(self.record.device)
        elif isinstance(input_data, dict):
            filtered_data = {}
            for key, value in self.record.items():
                filtered_data[key] = (1 - self.alpha) * value + self.alpha * input_data[key].to(value.device)
            return filtered_data
        else:
            raise ValueError