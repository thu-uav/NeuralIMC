import abc
from collections import namedtuple
from typing import Generator, NamedTuple, Union

import numpy as np
import torch
import torch.nn as nn
from gym import spaces
from torch_control.controllers.learning.modules.running_mean_std import \
    RunningMeanStd
from torch_control.controllers.learning.utils.common import (get_action_dim,
                                                             get_obs_shape,)
from torch_control.utils.common import nan_alarm


def explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]

    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero

    :param y_pred: the prediction
    :param y_true: the expected value
    :return: explained variance of ypred and y
    """
    assert y_true.ndim == 1 and y_pred.ndim == 1
    var_y = torch.var(y_true)
    return torch.nan if var_y == 0 else 1. - torch.var(y_true - y_pred) / var_y

class BaseBuffer(abc.ABC):
    """
    Base class that represent a buffer (rollout or replay)

    :param buffer_size: Max number of element in the buffer
    :param obs_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device to which the values will be converted
    :param num_envs: Number of parallel environments
    """

    def __init__(self,
                 buffer_size: int,
                 obs_space: spaces.Space,
                 action_space: spaces.Space,
                 device: str = "cpu",
                 num_envs: int = 1,):
        super().__init__()
        self.buffer_size = buffer_size
        self.obs_space = obs_space
        self.action_space = action_space
        self.obs_shape = get_obs_shape(obs_space)
        self.obs_dim = np.prod(self.obs_shape, dtype=np.int32)
        self.action_dim = get_action_dim(action_space)
        self.device = torch.device(device)
        self.num_envs = num_envs

    def size(self) -> int:
        """
        :return: The current size of the buffer
        """
        if self.full:
            return self.buffer_size
        return self.pos

    def reset(self) -> None:
        """
        Reset the buffer.
        """
        self.pos = 0
        self.full = False
        
    @abc.abstractmethod
    def add(self, *args, **kwargs) -> None:
        """
        Add elements to the buffer.
        """
        raise NotImplementedError()
    
    @staticmethod
    def swap_and_flatten(arr: torch.Tensor) -> torch.Tensor:
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (num_envs)
        to convert shape from [n_steps, num_envs, ...] (when ... is the shape of the features)
        to [n_steps * num_envs, ...] (which maintain the order)

        :param arr:
        :return:
        """
        arr = arr.swapdims(0, 1).reshape(-1, *arr.shape[2:])
        return arr

###### On-policy buffer ######
class OnpolicyBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    dones: torch.Tensor
    action_log_probs: torch.Tensor
    advantages: torch.Tensor
    progresses: torch.Tensor = None

class OnpolicyBuffer(BaseBuffer):
    """
    Rollout buffer used in on-policy algorithms like PPO.
    """
    def __init__(self,
                 buffer_size: int,
                 obs_space: spaces.Space,
                 action_space: spaces.Space,
                 device: Union[torch.device, str] = "cpu",
                 num_envs: int = 1,
                 gae_lambda: float = 1,
                 gamma: float = 0.99,
                 normalize_advantages: bool = True,):
        super().__init__(buffer_size, obs_space, action_space, device, num_envs=num_envs)
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.normalize_advantages = normalize_advantages
        self.reset()

    @torch.no_grad()
    def reset(self) -> None:
        self.observations = torch.zeros((self.buffer_size + 1, self.num_envs, *self.obs_shape), dtype=torch.float32).to(self.device)
        self.actions = torch.zeros((self.buffer_size, self.num_envs, self.action_dim), dtype=torch.float32).to(self.device)
        self.values = torch.zeros((self.buffer_size + 1, self.num_envs), dtype=torch.float32).to(self.device)
        self.rewards = torch.zeros((self.buffer_size, self.num_envs), dtype=torch.float32).to(self.device)
        self.returns = torch.zeros((self.buffer_size + 1, self.num_envs), dtype=torch.float32).to(self.device)
        self.dones = torch.zeros((self.buffer_size + 1, self.num_envs), dtype=torch.float32).to(self.device)
        self.action_log_probs = torch.zeros((self.buffer_size, self.num_envs), dtype=torch.float32).to(self.device)
        super().reset()
        
    @torch.no_grad()
    def add(self,
            next_obs: torch.Tensor,
            action: torch.Tensor,
            value: torch.Tensor,
            reward: torch.Tensor,
            done: torch.Tensor,
            action_log_probs: torch.Tensor,
            ) -> None:
        self.observations[self.pos + 1] = next_obs.clone()
        self.actions[self.pos] = action.clone()
        self.values[self.pos] = value.clone()
        self.rewards[self.pos] = reward.clone()
        self.dones[self.pos + 1] = done.clone()
        self.action_log_probs[self.pos] = action_log_probs.clone()
        
        self.pos = self.pos + 1 #) % self.buffer_size

    @torch.no_grad()
    def after_update(self) -> None:
        """Copy last observation to first observation and reset pointers after update is done"""
        self.observations[0] = self.observations[-1]
        self.dones[0] = self.dones[-1]
        # set remaining values to 0
        self.observations[1:].fill_(0)
        self.actions.fill_(0)
        self.values.fill_(0)
        self.rewards.fill_(0)
        self.returns.fill_(0)
        self.dones[1:].fill_(0)
        self.action_log_probs.fill_(0)
        self.pos = 0

    @torch.no_grad()
    def compute_returns_and_advantage(self, last_values: torch.Tensor, value_normalizer: RunningMeanStd = None) -> None:
        self.values[-1] = last_values
        gae = 0
        if value_normalizer is not None:
            values = value_normalizer.denormalize(self.values.unsqueeze(-1)).squeeze(-1)
        else:
            values = self.values

        for step in reversed(range(self.buffer_size)):
            delta = self.rewards[step] + self.gamma * values[step + 1] * (1 - self.dones[step + 1]) - values[step]
            gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[step + 1]) * gae
            self.returns[step] = gae + values[step]

        self.advantages = self.returns[:-1] - values[:-1]
        
        if self.normalize_advantages:
            self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    @torch.no_grad()
    def get_sampler(self, num_minibatch) -> Generator[OnpolicyBufferSamples, None, None]:
        indices = torch.randperm(self.buffer_size * self.num_envs, device=self.device)
        
        obs = self.swap_and_flatten(self.observations[:-1])
        actions = self.swap_and_flatten(self.actions)
        values = self.swap_and_flatten(self.values[:-1])
        returns = self.swap_and_flatten(self.returns[:-1])
        dones = self.swap_and_flatten(self.dones[:-1])
        action_log_probs = self.swap_and_flatten(self.action_log_probs)
        advantages = self.swap_and_flatten(self.advantages)

        minibatch_size = int(self.buffer_size * self.num_envs // num_minibatch)
        sampler = [indices[i * minibatch_size : (i + 1) * minibatch_size] for i in range(num_minibatch)]

        for indices in sampler:
            yield OnpolicyBufferSamples(
                observations=obs[indices],
                actions=actions[indices],
                values=values[indices],
                returns=returns[indices],
                dones=dones[indices],
                action_log_probs=action_log_probs[indices],
                advantages=advantages[indices],
            )

    def get_explained_var(self):
        return explained_variance(self.values.flatten(),
                                  self.returns.flatten())


class ChunkOnpolicyBuffer(BaseBuffer):
    """
    Rollout buffer used in on-policy algorithms like PPO, supporting action/value chunking.
    """
    def __init__(self,
                 buffer_size: int,
                 obs_space: spaces.Space,
                 action_space: spaces.Space,
                 device: Union[torch.device, str] = "cpu",
                 num_envs: int = 1,
                 gae_lambda: float = 1,
                 gamma: float = 0.99,
                 normalize_advantages: bool = True,
                 chunk_size: int = 1,
                 chunk_weight: float = 0.01,
                 chunk_weight_order: int = 0,
                 enable_action_chunk: bool = False,
                 enable_value_chunk: bool = False):
        super().__init__(buffer_size, obs_space, action_space, device, num_envs=num_envs)
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.normalize_advantages = normalize_advantages
        self.chunk_size = chunk_size
        self.chunk_weight = chunk_weight
        self.chunk_weight_order = chunk_weight_order
        self.enable_action_chunk = enable_action_chunk
        self.enable_value_chunk = enable_value_chunk
        self.offset = self.chunk_size - 1
        self.reset()

    @torch.no_grad()
    def reset(self) -> None:
        self.observations = torch.zeros(
            (self.offset+self.buffer_size+1, self.num_envs, *self.obs_shape)).float().to(self.device)
        self.progresses = torch.zeros(
            (self.offset+self.buffer_size+1, self.num_envs)).float().to(self.device)
        
        self.action_chunks = torch.zeros(
            (self.offset+self.buffer_size, self.buffer_size+self.offset, self.num_envs, self.action_dim)).float().to(self.device)
        self.mean_chunks = torch.zeros(
            (self.offset+self.buffer_size, self.buffer_size+self.offset, self.num_envs, self.action_dim)).float().to(self.device)
        self.var_chunks = torch.zeros(
            (self.offset+self.buffer_size, self.buffer_size+self.offset, self.num_envs, self.action_dim)).float().to(self.device)
        self.value_chunks = torch.zeros(
            (self.offset+self.buffer_size+1, self.buffer_size+1+self.offset, self.num_envs)).float().to(self.device)
        
        self.actions = torch.zeros(
            (self.buffer_size, self.num_envs, self.action_dim)).float().to(self.device)
        self.values = torch.zeros(
            (self.buffer_size+1, self.num_envs)).float().to(self.device)
        self.action_log_probs = torch.zeros(
            (self.buffer_size, self.num_envs)).float().to(self.device)
        self.rewards = torch.zeros(
            (self.buffer_size, self.num_envs)).float().to(self.device)
        self.returns = torch.zeros(
            (self.buffer_size+1, self.num_envs), dtype=torch.float32).to(self.device)
        self.dones = torch.zeros(
            (self.buffer_size+1, self.num_envs), dtype=torch.float32).to(self.device)
        
        super().reset()
        
    @torch.no_grad()
    def insert_obs(self, obs: torch.Tensor, idx: int = -1):
        if idx < 0:
            idx = self.pos
        self.observations[self.offset+idx] = obs.clone() # (num_envs, obs_dim)
        
    @torch.no_grad()
    def get_observations(self):
        return self.observations[self.offset:]
        
    @torch.no_grad()
    def add(self,
            next_obs: torch.Tensor,
            action: torch.Tensor,
            value: torch.Tensor,
            reward: torch.Tensor,
            done: torch.Tensor,
            action_log_probs: torch.Tensor,
            progress: torch.Tensor,) -> None:
        self.actions[self.pos] = action.clone()
        self.values[self.pos] = value.clone()
        self.rewards[self.pos] = reward.clone()
        self.action_log_probs[self.pos] = action_log_probs.clone()
        self.dones[self.pos+1] = done.clone()
        
        self.observations[self.offset+self.pos+1] = next_obs.clone()
        self.progresses[self.offset+self.pos+1] = progress.clone()
        
        self.pos = self.pos + 1 #) % self.buffer_size
        
    @torch.no_grad()
    def compute_chunk_weights(self, pos: int = None, progresses: torch.Tensor = None, next_pos: bool = False, chunk_size: int = None) -> torch.Tensor:
        if pos is None:
            pos = self.offset + self.pos
        if progresses is None:
            progresses = self.progresses.clone()
        if pos < 0:
            pos = progresses.shape[0] + pos
        if next_pos:
            progresses[pos] += 1.
        if chunk_size is None:
            chunk_size = self.chunk_size
            
        resets = progresses == 0
        episode_ids = resets.cumsum(dim=0) # (chunk_size-1+buffer_size+1, num_envs)
        curr_episode_ids = episode_ids[pos]
        episode_mask = episode_ids == curr_episode_ids # (chunk_size-1+buffer_size+1, num_envs)
        
        chunk_mask = torch.zeros_like(episode_mask) # (chunk_size-1+buffer_size+1, num_envs)
        chunk_mask[pos-chunk_size+1:pos+1] = 1.
        
        valid_chunk_mask = (episode_mask & chunk_mask) # (chunk_size-1+buffer_size+1, num_envs)
        
        if self.chunk_weight_order == 0:
            weight_logits = valid_chunk_mask.cumsum(dim=0).float() - 1. # (chunk_size-1+buffer_size+1, num_envs)
            exp_weights = torch.exp(-self.chunk_weight * weight_logits)
        else:
            weight_logits = valid_chunk_mask.flip(dims=[0]).cumsum(dim=0).flip(dims=[0]).float() - 1. # (chunk_size-1+buffer_size+1, num_envs)
            exp_weights = torch.exp(-self.chunk_weight * weight_logits)
            
        exp_weights = exp_weights * valid_chunk_mask.float()
        exp_weights = exp_weights / exp_weights.sum(dim=0, keepdim=True) # (chunk_size-1+buffer_size, num_envs)
        
        if torch.all(exp_weights == 0, dim=0).any():
            print("All chunk weights are zero, this is not allowed")
            import pdb; pdb.set_trace()
        
        return exp_weights # (chunk_size-1+buffer_size+1, num_envs)
    
    @torch.no_grad()
    def aggregate_action_chunks(self,
                                action_chunk: torch.Tensor,
                                mean_chunk: torch.Tensor,
                                var_chunk: torch.Tensor,
                                exp_weights: torch.Tensor = None,
                                insert: bool = True,) -> torch.Tensor:
        pos = self.pos
        chunk_size = self.chunk_size if self.enable_action_chunk else 1
        if insert:
            self.action_chunks[self.offset+pos, 
                               pos: pos+chunk_size] = action_chunk.transpose(0, 1)
            self.mean_chunks[self.offset+pos,
                             pos: pos+chunk_size] = mean_chunk.transpose(0, 1)
            self.var_chunks[self.offset+pos,
                            pos: pos+chunk_size] = var_chunk.transpose(0, 1)
            
            actions_for_curr_step = self.action_chunks[:, pos] # (chunk_size-1+buffer_size, num_envs, action_dim)
            means_for_curr_step = self.mean_chunks[:, pos] # (chunk_size-1+buffer_size, num_envs, action_dim)
            vars_for_curr_step = self.var_chunks[:, pos] # (chunk_size-1+buffer_size, num_envs, action_dim)
        else:
            actions_for_curr_step = self.action_chunks[:, pos] # (chunk_size-1+buffer_size, num_envs, action_dim)
            means_for_curr_step = self.mean_chunks[:, pos] # (chunk_size-1+buffer_size, num_envs, action_dim)
            vars_for_curr_step = self.var_chunks[:, pos] # (chunk_size-1+buffer_size, num_envs, action_dim)
            
            actions_for_curr_step[self.offset+pos] = action_chunk[:, 0] # (num_envs, action_dim)
            means_for_curr_step[self.offset+pos] = mean_chunk[:, 0] # (num_envs, action_dim)
            vars_for_curr_step[self.offset+pos] = var_chunk[:, 0] # (num_envs, action_dim)
            
        if exp_weights is None:
            exp_weights = self.compute_chunk_weights(pos=self.offset+pos, chunk_size=chunk_size)
            
        exp_weights = exp_weights[:-1] # (chunk_size-1+buffer_size, num_envs)
            
        agg_actions = (actions_for_curr_step * exp_weights.unsqueeze(-1)).sum(dim=0) # (num_envs, action_dim)
        agg_means = (means_for_curr_step * exp_weights.unsqueeze(-1)).sum(dim=0) # (num_envs, action_dim)
        agg_vars = (vars_for_curr_step * exp_weights.unsqueeze(-1).square()).sum(dim=0) # (num_envs, action_dim)
        
        return agg_actions, agg_means, agg_vars
    
    @torch.no_grad()
    def aggregate_value_chunks(self,
                               value_chunk: torch.Tensor,
                               exp_weights: torch.Tensor = None,
                               insert: bool = True,
                               next_pos: bool = False) -> torch.Tensor:
        pos = self.pos + int(next_pos)
        chunk_size = self.chunk_size if self.enable_value_chunk else 1
        if insert:
            self.value_chunks[self.offset+pos,
                              pos: pos+chunk_size] = value_chunk.transpose(0, 1)
            values_for_curr_step = self.value_chunks[:, pos] # (chunk_size-1+buffer_size+1, num_envs)
        else:
            values_for_curr_step = self.value_chunks[:, pos] # (chunk_size-1+buffer_size+1, num_envs)
            values_for_curr_step[self.offset+pos] = value_chunk[:, 0] # (num_envs)
            
        if exp_weights is None:
            exp_weights = self.compute_chunk_weights(pos=self.offset+pos, next_pos=next_pos, chunk_size=chunk_size)
            
        agg_values = (values_for_curr_step * exp_weights).sum(dim=0) # (num_envs)
        
        return agg_values
        
    @torch.no_grad()
    def aggregate_chunks(self, 
                          action_chunk: torch.Tensor, 
                          mean_chunk: torch.Tensor, 
                          var_chunk: torch.Tensor,
                          value_chunk: torch.Tensor,) -> torch.Tensor:
        if self.enable_action_chunk and self.enable_value_chunk:
            exp_weights = self.compute_chunk_weights() # (chunk_size-1+buffer_size+1, num_envs)
        else:
            exp_weights = None
        
        agg_actions, agg_means, agg_vars = self.aggregate_action_chunks(action_chunk, mean_chunk, var_chunk, exp_weights)
        agg_values = self.aggregate_value_chunks(value_chunk, exp_weights)
        
        action_dist = torch.distributions.Normal(agg_means, agg_vars.sqrt())
        action_log_probs = action_dist.log_prob(agg_actions).sum(dim=-1) # (num_envs)
        
        return agg_actions, action_log_probs, agg_values
    
    @torch.no_grad()
    def compute_returns_and_advantage(self, last_values: torch.Tensor, value_normalizer: RunningMeanStd = None) -> None:
        self.values[-1] = last_values
        gae = 0
        if value_normalizer is not None:
            values = value_normalizer.denormalize(self.values.unsqueeze(-1)).squeeze(-1)
        else:
            values = self.values

        for step in reversed(range(self.buffer_size)):
            delta = self.rewards[step] + self.gamma * values[step + 1] * (1 - self.dones[step + 1]) - values[step]
            gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[step + 1]) * gae
            self.returns[step] = gae + values[step]

        self.advantages = self.returns[:-1] - values[:-1]
        
        if self.normalize_advantages:
            self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)
    
    def get_explained_var(self):
        return explained_variance(self.values.flatten(),
                                  self.returns.flatten())
        
    @torch.no_grad()
    def get_sampler(self, num_minibatch) -> Generator[OnpolicyBufferSamples, None, None]:
        indices = torch.randperm(self.buffer_size * self.num_envs, device=self.device)
        
        obs_chunk = self.observations[:-1].unfold(0, self.chunk_size, 1) # (buffer_size, num_envs, obs_shape, chunk_size)
        obs_chunk = obs_chunk.transpose(-1, -2) # (buffer_size, num_envs, chunk_size, *obs_shape)
        
        progress_chunk = self.progresses[:-1].unfold(0, self.chunk_size, 1) # (buffer_size, num_envs, chunk_size)
        
        obs = self.swap_and_flatten(obs_chunk)
        progress = self.swap_and_flatten(progress_chunk)
        actions = self.swap_and_flatten(self.actions)
        values = self.swap_and_flatten(self.values[:-1])
        returns = self.swap_and_flatten(self.returns[:-1])
        dones = self.swap_and_flatten(self.dones[:-1])
        action_log_probs = self.swap_and_flatten(self.action_log_probs)
        advantages = self.swap_and_flatten(self.advantages)

        minibatch_size = int(self.buffer_size * self.num_envs // num_minibatch)
        sampler = [indices[i * minibatch_size : (i + 1) * minibatch_size] for i in range(num_minibatch)]

        for indices in sampler:
            yield OnpolicyBufferSamples(
                observations=obs[indices],
                actions=actions[indices],
                values=values[indices],
                returns=returns[indices],
                dones=dones[indices],
                action_log_probs=action_log_probs[indices],
                advantages=advantages[indices],
                progresses=progress[indices],
            )
            
    @torch.no_grad()
    def after_update(self) -> None:
        """Copy last observation to first observation and reset pointers after update is done"""
        self.observations[:self.offset+1] = self.observations[-self.offset-1:]
        self.progresses[:self.offset+1] = self.progresses[-self.offset-1:]
        # set remaining values to 0
        self.observations[self.offset+1:] = 0.
        self.progresses[self.offset+1:] = 0.
        
        if self.offset > 0:
            self.action_chunks[:self.offset] = self.action_chunks[-self.offset:]
            self.mean_chunks[:self.offset] = self.mean_chunks[-self.offset:]
            self.var_chunks[:self.offset] = self.var_chunks[-self.offset:]
            self.value_chunks[:self.offset] = self.value_chunks[-self.offset-1:-1]
        # set remaining values to 0
        self.action_chunks[self.offset:] = 0.
        self.mean_chunks[self.offset:] = 0.
        self.var_chunks[self.offset:] = 0.
        self.value_chunks[self.offset:] = 0.
        
        self.actions.fill_(0)
        self.values.fill_(0)
        self.rewards.fill_(0)
        self.returns.fill_(0)
        self.dones.fill_(0)
        self.action_log_probs.fill_(0)
        self.pos = 0
        
        
###### Trajectory buffer ######
class TrajectoryBufferSample(NamedTuple):
    state: torch.Tensor
    action: torch.Tensor
    priority: torch.Tensor = None
    trace_back: torch.Tensor = None

class TrajectoryBuffer:
    def __init__(self,
                 buffer_size: int,
                 episode_length: int,
                 state_dim: int,
                 action_dim: int,
                 num_envs: int,
                 device: Union[torch.device, str] = "cuda",
                 use_priority: bool = True,
                 priority_method: str = 'diff'):
        self.buffer_size = buffer_size
        self.episode_length = episode_length
        self.max_num_episodes = buffer_size // episode_length
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.use_priority = use_priority
        self.priority_method = priority_method
        
        self.reset()
        
    def reset(self) -> None:
        self.pos = 0
        self.full = False

        self.states = torch.zeros(self.max_num_episodes, self.episode_length, self.state_dim).to(self.device)
        self.actions = torch.zeros(self.max_num_episodes, self.episode_length, self.action_dim).to(self.device)
        self.active_mask = torch.zeros(self.max_num_episodes, self.episode_length).to(self.device)
        self.used_times = torch.ones(self.max_num_episodes).to(self.device) * (-1.)
        self.idx_pointer = torch.arange(self.num_envs).to(self.device)

    @property
    def priorities(self):
        return torch.exp(-0.01 * self.used_times)
        
    def get_next_idxes(self, num_idxes: int) -> torch.Tensor:
        if self.full:
            if self.use_priority:
                next_idxes = self.priorities.argsort(descending=False)[:num_idxes]
            else:
                next_idxes = (torch.arange(num_idxes).to(self.device) + self.pos) % self.max_num_episodes
        elif self.pos + num_idxes >= self.max_num_episodes:
            inner_idxes = torch.arange(self.pos, self.max_num_episodes).to(self.device)
            if self.use_priority:
                outer_idxes = self.priorities.argsort(descending=False)[:num_idxes - len(inner_idxes)]
            else:
                outer_idxes = torch.arange(num_idxes - len(inner_idxes)).to(self.device)
            next_idxes = torch.cat([inner_idxes, outer_idxes])
            self.full = True
        elif self.pos + num_idxes < self.max_num_episodes:
            next_idxes = torch.arange(self.pos, self.pos + num_idxes).to(self.device)
        else:
            print("[WARNING] Unexpected case in TrajectoryBuffer.get_next_idxes, need to check the logic")
            import pdb; pdb.set_trace()
        
        self.pos = (self.pos + num_idxes) % self.max_num_episodes

        return next_idxes
        
    def add(self,
            next_state: torch.Tensor,
            action: torch.Tensor,
            progress: torch.Tensor,
            done: torch.Tensor,) -> None:
        
        if done.sum() > 0:
            next_idxes = self.get_next_idxes(done.sum())
            self.idx_pointer[done] = next_idxes
            self.used_times[next_idxes.long()] = 0.
        
        self.states[self.idx_pointer.long(),
                    progress.long()] = next_state.clone()
        self.active_mask[self.idx_pointer.long(),
                         progress.long()] = 1.
    
        self.actions[self.idx_pointer.long()[~done],
                     progress[~done].long() - 1] = action[~done].clone()
        
    def init(self, state: torch.Tensor):
        self.pos += len(self.idx_pointer)
        self.states[self.idx_pointer.long(), 0] = state.clone()
        self.active_mask[self.idx_pointer.long(), 0] = 1.
        self.used_times[self.idx_pointer.long()] = 0.

    def compute_in_episode_priority(self, data):
        """
        data: (..., data_dim, seq_len)
        """
        if self.priority_method == 'diff':
            derivatives = torch.diff(data, dim=-1)
            magnitude = torch.abs(derivatives)
            magnitude = magnitude.sum(dim=-1).sum(dim=-1)
            in_episode_priority = magnitude / magnitude.sum(dim=1, keepdim=True)
        else:
            raise NotImplementedError
        
        return in_episode_priority
        
    def get_chunks(self, 
                   horizon: int, 
                   num_minibatch: int, 
                   minibatch_size: int,) -> torch.Tensor:
        # get valid episodes
        valid_episodes = self.used_times >= 0
        states = self.states[valid_episodes].unfold(1, horizon, 1) # (N, num_chunks, state_dim, horizon)
        actions = self.actions[valid_episodes].unfold(1, horizon, 1) # (N, num_chunks, action_dim, horizon)
        masks = self.active_mask[valid_episodes].unfold(1, horizon, 1) # (N, num_chunks, horizon)
        
        priority = self.priorities[valid_episodes].unsqueeze(1).expand_as(masks[..., 0])
        if self.use_priority:
            trace_back = torch.arange(self.max_num_episodes).to(self.device)[
                valid_episodes].unsqueeze(1).expand_as(masks[..., 0])

        in_episode_priority = self.compute_in_episode_priority(states)

        priority = priority * in_episode_priority

        # flatten the first two dimensions
        states = states.flatten(0, 1)
        actions = actions.flatten(0, 1)
        masks = masks.flatten(0, 1)
        priority = priority.flatten(0, 1)
        if self.use_priority:
            trace_back = trace_back.flatten(0, 1)
        
        # remove inactive chunks
        valid_chunks = masks.all(dim=-1) # (N * num_chunks)
        states = states[valid_chunks]
        actions = actions[valid_chunks]
        masks = masks[valid_chunks]
        priority = priority[valid_chunks]
        if self.use_priority:
            trace_back = trace_back[valid_chunks]
        
        max_num_chunks = states.shape[0]
        if num_minibatch is None or num_minibatch * minibatch_size > max_num_chunks:
            num_minibatch = max_num_chunks // minibatch_size
            print(f"[WARNING] no enough chunks to sample, use less minibatches {num_minibatch}")
            
        num_chunks = num_minibatch * minibatch_size

        # sort priority to pick the most #num_chunks important chunks
        sorted_idxes = priority.argsort(descending=True)

        picked_idxes = sorted_idxes[:num_chunks]
        states = states[picked_idxes]
        actions = actions[picked_idxes]
        priority = priority[picked_idxes]
        if self.use_priority:
            trace_back = trace_back[picked_idxes]
        
        if self.use_priority:
            # add used_times with trace back
            indices, counts = trace_back.unique(return_counts=True)
            self.used_times[indices] += counts.float()
        
        return states, actions, priority
    
    def get_sampler(self, 
                    minibatch_size: int, 
                    horizon: int, 
                    num_minibatch: int = None, 
                    device: str = None):
        all_states, all_actions, all_priority = self.get_chunks(horizon, num_minibatch, minibatch_size)
        
        indices = torch.randperm(all_states.shape[0], device=self.device)
        num_minibatch = all_states.shape[0] // minibatch_size
        sampler = [indices[i * minibatch_size : (i + 1) * minibatch_size] for i in range(num_minibatch)]

        if device is None:
            device = self.device
        
        for indices in sampler:
            yield TrajectoryBufferSample(
                state=all_states[indices].permute(0, 2, 1).to(device), # (minibatch_size, horizon, state_dim)
                action=all_actions[indices].permute(0, 2, 1).to(device), # (minibatch_size, horizon, action_dim)
                priority=all_priority[indices].to(device), # (minibatch_size)
            )
    