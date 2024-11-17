from functools import partial
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_control.controllers.learning.modules import \
    sb3_distributions as sb3_dist
from torch_control.controllers.learning.modules.modules import MLP, init
from torch_control.utils.common import nan_alarm

activation_dict = {
    'none': lambda x: x,
    'relu': F.relu,
    'elu': F.elu,
    'leakyrelu': F.leaky_relu,
    'tanh': F.tanh,
}

class MLPPolicy(nn.Module):
    def __init__(self, cfg, obs_dim, action_dim):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        assert cfg.output_activation.lower() in activation_dict.keys()
        self.output_activation_fn = activation_dict[cfg.output_activation.lower()]

        self.net = MLP([obs_dim] + cfg.hidden_sizes,
                        activation_fn=getattr(nn, cfg.activation),
                        use_layer_norm=cfg.use_layer_norm,
                        use_spectral_norm=cfg.use_spectral_norm,
                        activate_last=True)
        
        if cfg.independent_std:
            if cfg.squash_action:
                self.action_dist = sb3_dist.SquashedDiagGaussianDistribution(action_dim, transform_logp=cfg.transform_logp)
            else:
                self.action_dist = sb3_dist.DiagGaussianDistribution(action_dim)
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=cfg.hidden_sizes[-1], log_std_init=0.0
            )
        else:
            dist_kwargs = {
                'full_std': True,
                'use_expln': False,
                'learn_features': False,
                'squash_output': cfg.squash_action,
                'transform_logp': cfg.transform_logp,
            }
            self.action_dist = sb3_dist.StateDependentNoiseDistribution(action_dim, **dist_kwargs)
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=cfg.hidden_sizes[-1], latent_sde_dim=cfg.hidden_sizes[-1], log_std_init=cfg.log_std_init,
            )
            
    def init_params(self):
        self.net.init_params()
        init(self.action_net, 
             nn.init.orthogonal_, 
             lambda x: nn.init.constant_(x, 0), 
             gain=0.01)
        
    def reset_noise(self, n_envs: int = 1) -> None:
        """
        Sample new weights for the exploration matrix.

        :param n_envs:
        """
        if not isinstance(self.action_dist, 
                          sb3_dist.StateDependentNoiseDistribution):
            return
        self.action_dist.sample_weights(self.log_std, batch_size=n_envs)
        
    def get_action_dist_from_latent(self, latent: torch.Tensor):
        mean_actions = self.output_activation_fn(self.action_net(latent))
        nan_alarm(mean_actions)
        
        if isinstance(self.action_dist, sb3_dist.DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
        elif isinstance(self.action_dist, sb3_dist.StateDependentNoiseDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std, latent)
        else:
            raise ValueError("Invalid action distribution")

    def forward(self, obs, deterministic: bool = True):
        latent = self.net(obs)
        distribution = self.get_action_dist_from_latent(latent)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, self.action_dim))

        return actions
    
    def evaluate_actions(self, obs, actions):
        latent = self.net(obs)
        
        distribution = self.get_action_dist_from_latent(latent)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        
        if entropy is None:
            entropy = -1.0 * log_prob
        return log_prob, entropy.mean()

    def act(self, obs, deterministic=False):
        latent = self.net(obs)

        distribution = self.get_action_dist_from_latent(latent)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)

        return actions, log_prob
    
class ValueNet(nn.Module):
    def __init__(self, cfg, obs_dim):
        super().__init__()

        self.net = MLP([obs_dim] + cfg.hidden_sizes,
                        activation_fn=getattr(nn, cfg.activation),
                        use_layer_norm=cfg.use_layer_norm,
                        use_spectral_norm=cfg.use_spectral_norm,
                        activate_last=True)
        
        self.value_net = nn.Linear(cfg.hidden_sizes[-1], 1)
        
    def init_params(self):
        self.net.init_params()
        init(self.value_net,
             nn.init.orthogonal_,
             lambda x: nn.init.constant_(x, 0),
             gain=1.0)
        
    def forward(self, obs):
        return torch.squeeze(self.value_net(self.net(obs)), -1)


class ChunkMLPPolicy(MLPPolicy):
    def __init__(self, cfg, 
                 obs_dim, action_dim, 
                 chunk_size, chunk_weight, chunk_weight_order):
        super().__init__(cfg, obs_dim, action_dim)
        
        self.chunk_size = chunk_size
        self.chunk_weight = chunk_weight
        self.chunk_weight_order = chunk_weight_order
        
        self.action_dist = sb3_dist.ChunkDiagGaussianDistribution(
            action_dim, self.chunk_size)
        self.action_net, self.log_std = self.action_dist.proba_distribution_net(
            latent_dim=cfg.hidden_sizes[-1], log_std_init=0.0)
        
    def reset_noise(self, n_envs: int = 1) -> None:
        return 
    
    def get_action_dist_from_latent(self, latent: torch.Tensor):
        mean_actions = self.output_activation_fn(self.action_net(latent))
        
        return self.action_dist.proba_distribution(mean_actions, self.log_std)
    
    def forward(self, obs, deterministic: bool = True):
        """Given an observation at time t, predict the next k actions to take, i.e., \pi(a_t:t+k | o_t)

        Args:
            obs (torch.Tensor): (batch_size, obs_dim)
            deterministic (bool): Whether to sample from the distribution or take the mean action
            
        Returns:
            actions (torch.Tensor): (batch_size, chunk_size, action_dim)
        """
        latent = self.net(obs)
        
        distribution = self.get_action_dist_from_latent(latent)
        actions = distribution.get_actions(deterministic=deterministic) # (batch_size, chunk_size, action_dim)
        return actions
    
    def evaluate_actions(self, obs_chunk, action, chunk_weights: torch.Tensor = None):
        """Given a chunk of observations (o_t-k+1:t) and the recorded final action at time t (a_t), evaluate the log probability of the action and the entropy of the distribution.

        Args:
            obs_chunk (torch.Tensor): (batch_size, chunk_size, obs_dim)
            actions (torch.Tensor): (batch_size, action_dim)
            chunk_weights (torch.Tensor): (batch_size, chunk_size) indicating the weights of each time step
            
        Returns:
            log_prob (torch.Tensor): (batch_size,)
            entropy (torch.Tensor): (batch_size,)
        """
        assert obs_chunk.ndim == 3
        assert obs_chunk.shape[:2] == chunk_weights.shape
        
        obs_chunk = obs_chunk[:, -self.chunk_size:, :]
        
        latent = self.net(obs_chunk) # (batch_size, chunk_size, latent_dim)
        mean_actions = self.output_activation_fn(
            self.action_net(latent)) # (batch_size, chunk_size, chunk_size * action_dim)
        
        mean_actions = mean_actions.reshape(-1, self.chunk_size, self.chunk_size, self.action_dim)
        log_stds = self.log_std.reshape(self.chunk_size, self.action_dim) # time spans from 0 to k-1
        
        chunk_index = torch.arange(self.chunk_size).type_as(obs_chunk)
        aligned_means = mean_actions[:, chunk_index.long(), self.chunk_size-1-chunk_index.long(), :] # (batch_size, chunk_size, action_dim)
        aligned_vars = log_stds.flip(dims=[0]).exp().square() # (chunk_size, action_dim)
        
        if self.chunk_size == 1:
            mu = aligned_means.squeeze(1) # (batch_size, action_dim)
            sigma = aligned_vars.squeeze(0).sqrt() * torch.ones_like(mu) # (batch_size, action_dim)
        else:
            mu = (chunk_weights.unsqueeze(-1) * aligned_means).sum(dim=1) # (batch_size, action_dim)
            sigma = (chunk_weights.square().unsqueeze(-1) * aligned_vars).sum(dim=1).sqrt()
        
        try:
            action_dist = torch.distributions.Normal(mu, sigma)
        except:
            print("mu:", mu.min(dim=1))
            print("sigma:", sigma.max(dim=1))
            import pdb; pdb.set_trace()
        
        log_prob = action_dist.log_prob(action).sum(-1) # (batch_size,)
        entropy = action_dist.entropy().sum(-1)
        
        if entropy is None:
            entropy = -1.0 * log_prob
        
        return log_prob, entropy.mean()
    
    def act(self, obs, deterministic=False):
        """Given an observation at time t, predict the next k actions to take, i.e., \pi(a_t:t+k | o_t)

        Args:
            obs (torch.Tensor): (batch_size, obs_dim)
            deterministic (bool): Whether to sample from the distribution or take the mean action
            
        Returns:
            actions (torch.Tensor): (batch_size, chunk_size, action_dim)
            dist_mean (torch.Tensor): (batch_size, chunk_size, action_dim)
            dist_var (torch.Tensor): (batch_size, chunk_size, action_dim)
        """
        latent = self.net(obs)
        
        distribution = self.get_action_dist_from_latent(latent)
        actions = distribution.get_actions(deterministic=deterministic) # (batch_size, chunk_size, action_dim)
        
        return actions, distribution.mean, distribution.variance
        

class ChunkValueNet(ValueNet):
    def __init__(self, cfg, obs_dim, chunk_size, chunk_weight, chunk_weight_order):
        super().__init__(cfg, obs_dim)
        
        self.chunk_size = chunk_size
        self.chunk_weight = chunk_weight
        self.chunk_weight_order = chunk_weight_order
        
        self.value_net = nn.Linear(cfg.hidden_sizes[-1], chunk_size)
        
    def forward(self, obs):
        """Given an observation at time t, predict the next k values, i.e., V(o_t:t+k)
        
        Args:
            obs (torch.Tensor): (batch_size, obs_dim)
            
        Returns:
            values (torch.Tensor): (batch_size, chunk_size)
        """
        latent = self.net(obs) # (batch_size, chunk_size, latent_dim)
        values = self.value_net(latent) # (batch_size, chunk_size)
        return values
        
    def evaluate_values(self, obs_chunk, chunk_weights: torch.Tensor = None):
        """Given a chunk of observations (o_t-k+1:t), evaluate the value of the chunk, i.e., V(o_t:t+k)

        Args:
            obs_chunk (torch.Tensor): (batch_size, chunk_size, obs_dim)
            chunk_weights (torch.Tensor): (batch_size, chunk_size) indicating the weights of each time step
            
        Returns:
            values (torch.Tensor): (batch_size,)
        """
        
        assert obs_chunk.ndim == 3
        assert obs_chunk.shape[:2] == chunk_weights.shape
                
        obs_chunk = obs_chunk[:, -self.chunk_size:, :]
        
        latent = self.net(obs_chunk) # (batch_size, chunk_size, latent_dim)
        value_chunks = self.value_net(latent) # (batch_size, chunk_size, chunk_size)
        
        chunk_index = torch.arange(self.chunk_size).type_as(obs_chunk)
        aligned_values = value_chunks[:, chunk_index.long(), self.chunk_size-1-chunk_index.long()] # (batch_size, chunk_size)
        
        if self.chunk_size == 1:
            values = aligned_values.squeeze(1)
        else:
            values = (chunk_weights * aligned_values).sum(dim=1)
        
        return values
        
        