from typing import Dict, List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import GPT2Config

from ..base import get_params, TrajectoryModel
from .gpt2.trajectory_gpt2 import GPT2Model


class TransformerDynamics(TrajectoryModel):
    
    """
    This model uses GPT to model (state_1, action_1, state_2, action_2, ...)
    """
    
    def __init__(self,
                 cfg,
                 state_dim: int,
                 action_dim: int,
                 max_ep_len: int = 4096,
                 rot_mode: str = 'quat',
                 device: str = 'cuda'):
        super().__init__(state_dim, action_dim, rot_mode, device)
        self.cfg = cfg
        
        self.hidden_size = cfg.hidden_size
        self.max_context_length = cfg.max_context_length
        self.action_coef = cfg.action_coef
        if self.action_coef is None or self.action_coef < 0:
            self.action_coef = 0
        self.smooth_coef = cfg.smooth_coef
        if self.smooth_coef is None or self.smooth_coef < 0:
            self.smooth_coef = 0
        self.max_ep_len = max_ep_len
        self.action_tanh = cfg.get('action_tanh', True)
        self.abs_time_embed = cfg.get('abs_time_embed', False)
        self.clip_grad_norm = cfg.clip_grad_norm

        self.train_horizon = cfg.get('train_horizon', 40)
        
        config = GPT2Config(vocab_size=1, 
                            n_embd=self.hidden_size, 
                            **(cfg.gpt2_cfg))
        
        self.transformer = GPT2Model(config)
        
        self.embed_timestep = nn.Embedding(self.max_ep_len * 2, self.hidden_size) # *2 is a redundant operation to avoid numerical issues
        self.embed_state = nn.Linear(self.state_dim, self.hidden_size)
        self.embed_action = nn.Linear(self.action_dim, self.hidden_size)
        self.embed_ln = nn.LayerNorm(self.hidden_size)
        
        self.predict_state = nn.Linear(self.hidden_size, self.state_dim)
        self.predict_action = nn.Sequential(
            nn.Linear(self.hidden_size, self.action_dim),
            nn.Tanh() if self.action_tanh else nn.Identity()
        )
        
        modules = [self.transformer, self.embed_timestep, self.embed_state, self.embed_action, self.embed_ln,
                   self.predict_state, self.predict_action]
        params = get_params(modules)
        
        self.opt = torch.optim.Adam(params, lr=cfg.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, mode='min', factor=0.5, patience=4, verbose=True)
        
        self.to(self.device)

    def __repr__(self):
        return "TransformerDynamics(hidden_size={}, state_dim={}, action_dim={}, max_ep_len={}, rot_mode={}, device={})".format(
            self.hidden_size, self.state_dim, self.action_dim, self.max_ep_len, self.tgt_rot, self.device
        )
        
    def full_forward(self, states, actions, timesteps=None, attention_mask=None):
        batch_size, seq_length = states.shape[0], states.shape[1]
        
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long).to(self.device)
        if timesteps is None or not self.abs_time_embed:
            timesteps = torch.arange(seq_length, device=self.device).unsqueeze(0).expand(batch_size, -1).to(self.device)
            
        # embed states and actions
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        time_embeddings = self.embed_timestep(timesteps)
        
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        
        # this makes the sequence look like (s_1, a_1, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = torch.stack(
            (state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 2*seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)
        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 2*seq_length)
        
        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        
        x = transformer_outputs['last_hidden_state']
        
        # reshape x so that the second dimension corresponds to the original
        # states (0), or actions (1); i.e. x[:,0,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 2, self.hidden_size).permute(0, 2, 1, 3) # (batch, 2, seq, hidden)
        
        pred_state = self.predict_state(x[:, 1]) # predict next state given current state and action
        pred_action = self.predict_action(x[:, 0]) # predict action given current state
        
        return pred_state, pred_action
    
    def full_predict(self, states, actions, timesteps, output_terms=['state', 'action']):
        if states.ndim == 3:
            batch_size, seq_length = states.shape[0], states.shape[1]
        else:
            batch_size, seq_length = 1, states.shape[0]
        
        states = states.reshape(batch_size, -1, self.state_dim)
        actions = actions.reshape(batch_size, -1, self.action_dim)
        if timesteps is not None:
            timesteps = timesteps.reshape(batch_size, -1)
        
        if self.max_context_length is not None:
            states = states[:, -self.max_context_length:]
            actions = actions[:, -self.max_context_length:]
            
            if timesteps is not None:
                timesteps = timesteps[:, -self.max_context_length:]
            else:
                timesteps = torch.arange(states.shape[1], device=states.device).unsqueeze(0).expand(batch_size, -1)
            
            # pad all tokens to sequence length
            attention_mask = torch.cat([torch.zeros(self.max_context_length-states.shape[1]), torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).unsqueeze(0).expand(batch_size, -1)
            
            states = torch.cat(
                [torch.zeros((states.shape[0], self.max_context_length-states.shape[1], 
                              self.state_dim), device=states.device), 
                 states],
                dim=1).to(dtype=torch.float32)
            actions = torch.cat(
                [torch.zeros((actions.shape[0], self.max_context_length - actions.shape[1], 
                              self.action_dim), device=actions.device), 
                 actions],
                dim=1).to(dtype=torch.float32)
            if timesteps is not None:
                timesteps = torch.cat(
                    [torch.zeros((timesteps.shape[0], self.max_context_length-timesteps.shape[1]), 
                                device=timesteps.device),
                    timesteps],
                    dim=1).to(dtype=torch.long)
            
        else:
            attention_mask = None
            
        pred_states, pred_actions = self.full_forward(states, actions, timesteps, attention_mask) # (batch, seq, dim)
        
        outputs = {}
        if 'state' in output_terms:
            outputs['state'] = pred_states[:, -1]
        if 'action' in output_terms:
            outputs['action'] = pred_actions[:, -1]
        
        return outputs
    
    def update_step(self, states, actions, timesteps=None):
        assert states.ndim == 3
        
        states = self.preprocess_state(states)
        
        target_states = states[:, 1:].clone()
        target_actions = actions.clone()
        
        pred_states, pred_actions = self.full_forward(states, actions) # (batch, seq, dim)
        
        state_pred_loss = F.mse_loss(pred_states[:, :-1].flatten(0,1), target_states.flatten(0,1), reduction='none')
        action_pred_loss = F.mse_loss(pred_actions.flatten(0,1), target_actions.flatten(0,1), reduction='none')
        state_smooth_loss = F.mse_loss(pred_states[:, 1:].flatten(0,1), pred_states[:, :-1].flatten(0,1), reduction='none')
        action_smooth_loss = F.mse_loss(pred_actions[:, 1:].flatten(0,1), pred_actions[:, :-1].flatten(0,1), reduction='none')

        stds = state_pred_loss.view(-1, state_pred_loss.shape[-1]).std(0)
        weights = stds / stds.sum()
        state_loss = ((state_pred_loss.mean(0) + self.smooth_coef * state_smooth_loss.mean(0)) * weights).sum()

        stds = action_pred_loss.view(-1, action_pred_loss.shape[-1]).std(0)
        weights = stds / stds.sum()
        action_loss = ((action_pred_loss.mean(0) + self.smooth_coef * action_smooth_loss.mean(0)) * weights).sum()

        loss = state_loss + self.action_coef * action_loss

        self.opt.zero_grad()
        loss.mean().backward()
        if self.clip_grad_norm is not None:
            assert len(self.opt.param_groups) == 1
            params = self.opt.param_groups[0]['params']
            grad_norm = torch.nn.utils.clip_grad_norm_(params, self.clip_grad_norm)
        self.opt.step()
        
        return {'loss': loss, 'pred_state_loss': state_pred_loss, 'pred_action_loss': action_pred_loss, 'smooth_state_loss' : state_smooth_loss, 'smooth_action_loss': action_smooth_loss, 'grad_norm': grad_norm}
        
    def unroll(self, x0, us, detach_xt=False):
        assert x0.ndim == 2
        assert us.ndim == 3
        assert us.shape[1] == x0.shape[0]
        
        T, batch_size, _ = us.shape
        
        actions = torch.zeros((batch_size, 0, self.action_dim), device=self.device, dtype=torch.float32)
        states = x0.view(batch_size, 1, self.state_dim)
        timesteps = torch.zeros((batch_size, 1), device=self.device, dtype=torch.long)
        
        for t in range(T):
            ut = us[t]
            
            actions = torch.cat([actions, ut.unsqueeze(1)], dim=1) # (batch, seq, dim)
            
            preds = self.full_predict(states, actions, 
                                 timesteps=timesteps, 
                                 output_terms=['state'])
            
            states = torch.cat([states, preds['state'].unsqueeze(1)], dim=1)
            timesteps = torch.cat([timesteps,
                                   torch.ones((batch_size, 1), device=self.device, dtype=torch.long) * (t+1)], dim=1)
            
            if detach_xt:
                states = states.detach()
            
        pred_xs = states[:, 1:].transpose(0, 1)
        return pred_xs

    def reset_aux_vars(self, idx: Union[int, torch.Tensor]):
        if isinstance(idx, int):
            idx = torch.arange(idx, device=self.device)
            
        n_batch = idx.shape[0]
        
        if self.max_context_length is not None:
            max_len = self.max_context_length
        else:
            max_len = self.max_ep_len
        
        if 'state' not in self.aux_vars:
            self.aux_vars['state'] = torch.zeros((n_batch, max_len, self.state_dim), device=self.device)
            self.aux_vars['action'] = torch.zeros((n_batch, max_len, self.action_dim), device=self.device, dtype=torch.float32)
            self.aux_vars['timestep'] = torch.zeros((n_batch, max_len), device=self.device, dtype=torch.long)
        else:
            self.aux_vars['state'][idx] = 0.
            self.aux_vars['action'][idx] = 0.
            self.aux_vars['timestep'][idx] = 0
            
    def add_xu(self, x, u):
        self.aux_vars['state'] = torch.cat([self.aux_vars['state'].roll(-1, 1)[:,:-1], x.unsqueeze(1)], dim=1)
        self.aux_vars['action'] = torch.cat([self.aux_vars['action'].roll(-1, 1)[:,:-1], u.unsqueeze(1)], dim=1)
        t = self.aux_vars['timestep'][:, -1] + 1
        self.aux_vars['timestep'] = torch.cat([self.aux_vars['timestep'].roll(-1, 1)[:,:-1], t.unsqueeze(1)], dim=1)
    
    @torch.no_grad()
    def predict(self, x, u, output_quat: bool = False):
        x = self.preprocess_state(x)
        
        self.add_xu(x, u)
        
        states, actions, timesteps = self.aux_vars['state'], self.aux_vars['action'], self.aux_vars['timestep']
        
        preds = self.full_predict(states, actions, timesteps, output_terms=['state', 'action'])
        
        xtp1 = preds['state']
        
        xtp1 = self.postprocess_state(xtp1, output_quat)
        
        return xtp1
        
        