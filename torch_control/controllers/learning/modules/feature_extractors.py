import numpy as np
import torch
import torch.nn as nn
from torch_control.controllers.learning.modules.modules import MLP, TCN
from torch_control.controllers.learning.modules.transformers import build_transformer_encoder, get_pos_encoding
            
class MixinExtractor(nn.Module):
    def __init__(self, obs_dict: dict, activation_fn=nn.ELU):
        """
        obs_dict: dict of {name: (length, extractor_type)}
        """
        super().__init__()
        self.suffix = "_extractor"
        self.valid_obs = []
        self.obs_slice_dict = {}
        
        slice_start = 0
        print("Observation Feature Extractors:")
        for name in obs_dict:
            length, kwargs = obs_dict[name]
            obs_slice = slice(slice_start, slice_start + length)
            self.obs_slice_dict[name] = obs_slice
            slice_start += length
            
            if length == 0 or kwargs.net is None or kwargs.net == "none":
                continue

            if kwargs.net == "identity":
                setattr(self, name + self.suffix, 
                        IdentityExtractor(obs_slice))
            elif kwargs.net == "mlp":
                output_dim = kwargs.get('output_dim', None)
                assert output_dim is not None
                setattr(self, name + self.suffix, 
                        MLPExtractor(obs_slice, output_dim=output_dim, activation_fn=activation_fn))
            elif kwargs.net == "simple_tcn":
                unit_dim = kwargs.get('unit_dim', None)
                assert unit_dim is not None
                setattr(self, name + self.suffix, 
                        SimpleTCNExtractor(obs_slice, unit_dim=unit_dim, activation_fn=activation_fn))
            elif kwargs.net == "complex_tcn":
                output_dim = kwargs.get('output_dim', None)
                unit_dim = kwargs.get('unit_dim', None)
                assert output_dim is not None and unit_dim is not None
                setattr(self, name + self.suffix, 
                        ComplexTCNEncoder(obs_slice, unit_dim=unit_dim, output_dim=output_dim, activation_fn=activation_fn))
            elif kwargs.net == "transformer":
                output_dim = kwargs.get('output_dim', None)
                unit_dim = kwargs.get('unit_dim', None)
                transformer_cfg = kwargs.get('transformer_cfg', None)
                assert transformer_cfg is not None
                setattr(self, name + self.suffix, 
                        TransformerExtractor(transformer_cfg, obs_slice, unit_dim=unit_dim, output_dim=output_dim))
            else:
                raise NotImplementedError(f"Extractor type {kwargs.net} not implemented.")
            
            self.valid_obs.append(name)
            print(f"{name}: {kwargs} \n\t{obs_slice} => feature dim {getattr(self, name + self.suffix).feature_dim}")

    @property
    def feature_dim(self):
        return sum([getattr(self, name + self.suffix).feature_dim for name in self.valid_obs])
    
    @property
    def feature_slice_dict(self):
        all_dims = np.array([getattr(self, name + self.suffix).feature_dim for name in self.valid_obs])
        all_cumdims = np.cumsum(all_dims)
        slice_dict = {name: slice(all_cumdims[i-1] if i > 0 else 0, all_cumdims[i]) for i, name in enumerate(self.valid_obs)}
        return slice_dict
        
    def forward(self, obs):
        features = [getattr(self, name + self.suffix)(obs) for name in self.valid_obs]
        return torch.cat(features, dim=-1)
        
    def init_params(self, gain: float = np.sqrt(2)):
        for name in self.valid_obs:
            getattr(self, name + self.suffix).init_params(gain=gain)
    
class IdentityExtractor(nn.Module):
    def __init__(self, obs_slice: slice):
        super().__init__()
        self.obs_slice = obs_slice
        
    @property
    def feature_dim(self):
        return self.obs_slice.stop - self.obs_slice.start
    
    def forward(self, obs):
        return obs[..., self.obs_slice]
    
    def init_params(self, gain: float = np.sqrt(2)):
        pass

class MLPExtractor(nn.Module):
    def __init__(self, 
                 obs_slice: slice,
                 output_dim: int = 16,
                 activation_fn=nn.ELU):
        super().__init__()
        self.obs_slice = obs_slice
        self.output_dim = output_dim
        
        layers = [obs_slice.stop - obs_slice.start, 32, 32, self.output_dim]
        self.mlp = MLP(layers=layers,
                       activation_fn=activation_fn,
                       use_layer_norm=False,
                       use_spectral_norm=False,
                       activate_last=True)
        
    @property
    def feature_dim(self):
        return self.output_dim
        
    def forward(self, obs):
        return self.mlp(obs[..., self.obs_slice])
    
    def init_params(self, gain: float = np.sqrt(2)):
        self.mlp.init_params()
    
class SimpleTCNExtractor(nn.Module):
    def __init__(self, 
                 obs_slice: slice,
                 unit_dim: int = 3,
                 activation_fn=nn.ELU):
        super().__init__()
        self.obs_slice = obs_slice
        self.unit_dim = unit_dim
        seq_len = (self.obs_slice.stop - self.obs_slice.start) // self.unit_dim

        if seq_len >= 20:
            self.tcn = TCN(channels=[self.unit_dim, 32, 16],
                        kernels=[6, 4],
                        strides=[3, 2],
                        activation_fn=activation_fn,
                        activate_last=False)
        elif seq_len >= 5:
            self.tcn = TCN(channels=[self.unit_dim, 8, 8],
                           kernels=[3, 3],
                           strides=[1, 1],
                           activation_fn=activation_fn,
                           activate_last=False)
        else:
            raise ValueError("Sequence with length less than 5 is too short for TCN.")
        
        self.output_dim = np.prod(self.tcn(
            torch.zeros(1, self.unit_dim, seq_len)).shape[-2:], dtype=np.int32)

    @property
    def feature_dim(self):
        return self.output_dim

    def forward(self, obs):
        x = obs[..., self.obs_slice]
        x = x.reshape(x.shape[:-1] + (-1, self.unit_dim))
        x = torch.permute(x, (0, 2, 1)) # (batch, channels, seq_len)
        x = self.tcn(x)
        return x.flatten(start_dim=1)
   
    def init_params(self, gain: float = np.sqrt(2)):
        self.tcn.init_params()


class ComplexTCNEncoder(nn.Module):
    def __init__(self,
                 obs_slice: slice,
                 unit_dim: int = 3,
                 output_dim: int = 32,
                 activation_fn=nn.ELU):
        super().__init__()
        self.obs_slice = obs_slice
        self.unit_dim = unit_dim
        self.output_dim = output_dim
        seq_len = (obs_slice.stop - obs_slice.start) // unit_dim
        
        self.linear_in = MLP([self.unit_dim, 32],
                             activation_fn=activation_fn,
                             activate_last=True)
        
        if seq_len >= 40:
            self.tcn = TCN(channels=[32, 32, 32, 32],
                        kernels=[8, 5, 5],
                        strides=[4, 1, 1],
                        activation_fn=activation_fn,
                        activate_last=True)
        elif seq_len >= 20:
            self.tcn = TCN(channels=[32, 32, 32],
                        kernels=[6, 4],
                        strides=[2, 2],
                        activation_fn=activation_fn,
                        activate_last=True)
        elif seq_len >= 10:
            self.tcn = TCN(channels=[32, 32, 32],
                        kernels=[4, 2],
                        strides=[2, 1],
                        activation_fn=activation_fn,
                        activate_last=True)
        else:
            raise ValueError("Sequence with length less than 7 should use the SimpleTCNExtractor.")
        
        _tcn_dim = np.prod(self.tcn(torch.zeros(1, 32, seq_len)).shape[-2:], dtype=np.int32)
        
        self.linear_out = MLP([_tcn_dim, output_dim],
                              activation_fn=activation_fn,
                              activate_last=True)
        
    @property
    def feature_dim(self):
        return self.output_dim
        
    def forward(self, obs):
        x = obs[..., self.obs_slice]
        x = x.reshape(x.shape[:-1] + (-1, self.unit_dim)) # (batch, seq_len, channels)
        x = self.linear_in(x)
        x = torch.permute(x, (0, 2, 1)) # (batch, channels, seq_len)
        x = self.tcn(x)
        x = x.flatten(start_dim=1)
        x = self.linear_out(x)
        return x

class TransformerExtractor(nn.Module):
    def __init__(self,
                 transformer_cfg,
                 obs_slice: slice,
                 unit_dim: int = 3,
                 output_dim: int = 32,):
        super().__init__()
        self.obs_slice = obs_slice
        self.unit_dim = unit_dim
        self.output_dim = output_dim
        seq_len = (obs_slice.stop - obs_slice.start) // unit_dim
        # seq shape: (batch, seq_len, unit_dim)

        self.in_embed = nn.Linear(self.unit_dim, transformer_cfg.hidden_dim)
        self.out_embed = nn.Linear(transformer_cfg.hidden_dim, self.output_dim)

        self.cls_embed = nn.Embedding(1, transformer_cfg.hidden_dim)
        self.pos_embed = nn.Embedding(seq_len, transformer_cfg.hidden_dim)
        self.cls_embed = nn.Embedding(1, transformer_cfg.hidden_dim)
        self.encoder = build_transformer_encoder(transformer_cfg)

    @property
    def feature_dim(self):
        return self.output_dim

    def forward(self, obs):
        x = obs[..., self.obs_slice]
        x = x.reshape(x.shape[:-1] + (-1, self.unit_dim))
        obs_embed = self.in_embed(x) # (batch, seq_len, embed_dim)
        pos_embed = torch.arange(
            x.shape[1]).to(x.device).unsqueeze(0).repeat(x.shape[0], 1) # (batch, seq_len)
        pos_embed = self.pos_embed(pos_embed.long()) # (batch, seq_len, embed_dim)
        obs_embed = obs_embed + pos_embed # (batch, seq_len, embed_dim)
        cls_embed = self.cls_embed.weight # (1, embed_dim)
        cls_embed = cls_embed.unsqueeze(0).repeat(x.shape[0], 1, 1) # (batch, 1, embed_dim)
        x = torch.cat([cls_embed, obs_embed], dim=1) # (batch, seq_len + 1, embed_dim)
        x = torch.permute(x, (1, 0, 2)) # (seq_len + 1, batch, embed_dim)
        x = self.encoder(x) # (seq_len + 1, batch, embed_dim)
        out = self.out_embed(x[0]) # (batch, embed_dim)

        return out

        
if __name__ == "__main__":
    test_input = torch.randn(10, 20 + 10 * 3 + 4)
    unit_dim = 14
    seq_len = 50
    
    model = ComplexTCNEncoder(slice(0, 0 + seq_len * unit_dim), output_dim=32, unit_dim=unit_dim, short_traj=5)

    print(model.feature_dim)

    