import torch
from torch import nn
from torch_control.utils.common import nan_alarm

_STD_MIN_VALUE = 1e-6
_STD_MAX_VALUE = 1e6

class RunningMeanStd(nn.Module):
    def __init__(self, shape, device=torch.device('cpu')):
        super().__init__()
        
        self.register_buffer('count', torch.zeros(1, requires_grad=False, device=device))
        self.register_buffer('mean', torch.zeros(shape, requires_grad=False, device=device, dtype=torch.float32))
        self.register_buffer('std', torch.ones(shape, requires_grad=False, device=device, dtype=torch.float32))
        self.register_buffer('summed_var', torch.zeros(shape, requires_grad=False, device=device, dtype=torch.float32))

    def _validate_batch_shapes(self, batch):
        expected_shape = (batch.shape[0],) + self.mean.shape
        assert batch.shape == expected_shape, f'{batch.shape} != {expected_shape}'

    @torch.no_grad()
    def update(self, batch, validate_shapes=True):
        batch = batch.reshape(-1, *self.mean.shape)
        if validate_shapes:
            self._validate_batch_shapes(batch)

        batch_dims = batch.shape[0]
        self.count = self.count + batch_dims
        diff_to_old_mean = batch - self.mean
        mean_update = torch.sum(diff_to_old_mean, dim=0) / self.count
        self.mean = self.mean + mean_update
        diff_to_new_mean = batch - self.mean
        var_update = diff_to_old_mean * diff_to_new_mean
        var_update = torch.sum(var_update, dim=0)
        self.summed_var = self.summed_var + var_update
        self.summed_var = torch.clamp(self.summed_var, min=0)  # ensure summed_var non-negative
        self.std = torch.sqrt(self.summed_var / self.count)
        self.std = torch.clamp(self.std, _STD_MIN_VALUE, _STD_MAX_VALUE)

    def normalize(self, batch, max_abs_value=None):
        if not torch.is_floating_point(batch):
            return batch
        data = (batch - self.mean) / self.std
        if max_abs_value is not None:
            data = torch.clamp(data, -max_abs_value, +max_abs_value)
        return data

    def denormalize(self, batch):
        if not torch.is_floating_point(batch):
            return batch
        return batch * self.std + self.mean
