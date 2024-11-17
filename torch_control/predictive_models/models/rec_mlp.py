from functools import partial
from typing import Dict, List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch_control.utils.common import nan_alarm
from .base import get_params, TrajectoryModel

def build_mlp(input_dim, output_dim, hidden_dim, hidden_depth, sn=False, sn_first=False, sn_last=False, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        if sn:
            if sn_first:
                mods = [spectral_norm(nn.Linear(input_dim, hidden_dim)), nn.ReLU(inplace=True)]
            else:
                mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
            for i in range(hidden_depth - 1):
                mods += [spectral_norm(nn.Linear(hidden_dim, hidden_dim)), nn.ReLU(inplace=True)]
            if sn_last:
                mods.append(spectral_norm(nn.Linear(hidden_dim, output_dim)))
            else:
                mods.append(nn.Linear(hidden_dim, output_dim))
        else:
            mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
            for i in range(hidden_depth - 1):
                mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
            mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data, gain=np.sqrt(2))
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param.data, gain=np.sqrt(2))
            elif "bias" in name:
                param.data.fill_(0)
    elif isinstance(m, nn.GRU):
        for name, param in m.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param.data, gain=np.sqrt(2))
            elif "bias" in name:
                param.data.fill_(0)

class RecMLPDynamics(TrajectoryModel):
    def __init__(self,
                 cfg,
                 state_dim: int,
                 action_dim: int,
                 rot_mode: str = 'quat',
                 device: str = 'cuda'):
        super().__init__(state_dim, action_dim, rot_mode, device)

        self.cfg = cfg

        self.smooth_coef = cfg.smooth_coef
        if self.smooth_coef is None or self.smooth_coef < 0:
            self.smooth_coef = 0
        
        self.residual = cfg.residual
        self.detach_xt = cfg.detach_xt
        self.clip_grad_norm = cfg.clip_grad_norm
        
        self.rec_type = cfg.rec_type
        self.rec_num_layers = cfg.rec_num_layers
        self.rec_latent_dim = cfg.rec_latent_dim

        self.train_horizon = cfg.get('train_horizon', 4)
        
        self.xu_enc = build_mlp(self.state_dim + action_dim, cfg.rec_latent_dim, 
                                cfg.xu_enc_hidden_dim, cfg.xu_enc_hidden_depth, 
                                sn=cfg.sn, sn_first=True, sn_last=True)
        self.x_dec = build_mlp(cfg.rec_latent_dim, self.state_dim,
                               cfg.x_dec_hidden_dim, cfg.x_dec_hidden_depth,
                               sn=cfg.sn, sn_first=True, sn_last=False)
        
        modules = [self.xu_enc, self.x_dec]
        
        if cfg.rec_num_layers > 0:
            if cfg.rec_type == 'lstm':
                self.rec = nn.LSTM(cfg.rec_latent_dim, cfg.rec_latent_dim, cfg.rec_num_layers)
            elif cfg.rec_type == 'gru':
                self.rec = nn.GRU(cfg.rec_latent_dim, cfg.rec_latent_dim, cfg.rec_num_layers)
            else:
                raise ValueError(f'Unknown rec type {cfg.rec_type}')
            self.rec.flatten_parameters()
            
            modules.append(self.rec)
            
        params = get_params(modules)
        self.opt = torch.optim.Adam(params, lr=cfg.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, mode='min', factor=0.5, patience=4, verbose=True)
        
        self.to(self.device)

    def __repr__(self):
        return "RecMLPDynamics(rec_type={}, rec_num_layers={}, rec_latent_dim={}, residual={}, detach_xt={}, smooth_coef={}, clip_grad_norm={})".format(
            self.rec_type, self.rec_num_layers, self.rec_latent_dim, self.residual, self.detach_xt, self.smooth_coef, self.clip_grad_norm
        )
        
    def init_hidden_state(self, n_batch: int):
        if self.rec_type == 'lstm':
            h = torch.zeros(self.rec_num_layers, n_batch, self.rec_latent_dim, device=self.device)
            c = torch.zeros_like(h)
            h = (h, c)
        elif self.rec_type == 'gru':
            h = torch.zeros(self.rec_num_layers, n_batch, self.rec_latent_dim, device=self.device)
        else:
            raise ValueError(f'Unknown rec type {self.rec_type}')
        
        return h
    
    def update_step(self, state, action):
        """update step for training the model

        Args:
            state (torch.Tensor): (batch_size, seq_len, state_dim)
            action (torch.Tensor): (batch_size, seq_len, action_dim)

        Returns:
            pred_loss (torch.Tensor): (batch_size, seq_len, state_dim)
            grad_norm (float): gradient norm
        """
        assert state.ndim == 3
        
        state = self.preprocess_state(state)
        nan_alarm(state)
        
        # transform to (seq, batch, dim)
        state, action = state.permute(1, 0, 2), action.permute(1, 0, 2)
        target_state = state[1:].clone()
        # predict next states
        pred_state = self.unroll(state[0], action[:-1], detach_xt=self.detach_xt)
        assert pred_state.shape == target_state.shape
        nan_alarm(pred_state)
        
        pred_loss = F.mse_loss(pred_state.flatten(0,1), 
                               target_state.flatten(0,1), 
                               reduction='none')

        smooth_loss = F.mse_loss(pred_state[1:].flatten(0,1), 
                                pred_state[:-1].flatten(0,1), 
                                reduction='none')

        stds = pred_loss.view(-1, pred_loss.shape[-1]).std(0)
        weights = stds / stds.sum()

        # loss = (pred_loss * weights).mean() + 0.1 * (smooth_loss * weights).mean()
        loss = ((pred_loss.mean(0) + self.smooth_coef * smooth_loss.mean(0)) * weights).sum()
        
        self.opt.zero_grad()
        loss.backward()
        if self.clip_grad_norm is not None:
            assert len(self.opt.param_groups) == 1
            params = self.opt.param_groups[0]['params']
            grad_norm = torch.nn.utils.clip_grad_norm_(params, self.clip_grad_norm)
        self.opt.step()
        
        nan_alarm(pred_loss)
        
        return {'loss': loss, 'pred_state_loss': pred_loss, 'smooth_state_loss': smooth_loss, 'grad_norm': grad_norm}
    
    def unroll(self, x0, us, detach_xt=False):
        assert x0.ndim == 2
        assert us.ndim == 3
        assert us.shape[1] == x0.shape[0]
        
        T, batch_size, _ = us.shape
        
        if self.rec_num_layers > 0:
            h = self.init_hidden_state(batch_size)
            
        pred_xs = []
        xt = x0
        for t in range(T):
            ut = us[t]
            
            if detach_xt:
                xt = xt.detach()
                
            xu_t = torch.cat([xt, ut], dim=1)
            xu_emb = self.xu_enc(xu_t).unsqueeze(0)
            if self.rec_num_layers > 0:
                xtp1_emb, h = self.rec(xu_emb, h)
            else:
                xtp1_emb = xu_emb
            if self.residual:
                xtp1 = xt + self.x_dec(xtp1_emb.squeeze(0))
            else:
                xtp1 = self.x_dec(xtp1_emb.squeeze(0))
            pred_xs.append(xtp1)
            xt = xtp1
            
        pred_xs = torch.stack(pred_xs, dim=0)
        
        return pred_xs
    
    @torch.no_grad()
    def forward(self, x, u):
        return self.predict(x, u)
    
    def reset_aux_vars(self, idx: Union[int, torch.Tensor]):
        if isinstance(idx, int):
            idx = torch.arange(idx, device=self.device)

        n_batch = idx.shape[0]
        if self.rec_num_layers > 0:
            if self.rec_type == 'lstm':
                h = torch.zeros(self.rec_num_layers, n_batch, self.rec_latent_dim, device=self.device)
                c = torch.zeros_like(h)
                if 'h' not in self.aux_vars:
                    self.aux_vars['h'] = h
                    self.aux_vars['c'] = c
                else:
                    self.aux_vars['h'][:, idx] = 0.
                    self.aux_vars['c'][:, idx] = 0.
            elif self.rec_type == 'gru':
                h = torch.zeros(self.rec_num_layers, n_batch, self.rec_latent_dim, device=self.device)
                if 'h' not in self.aux_vars:
                    self.aux_vars['h'] = h
                else:
                    self.aux_vars['h'][:, idx] = 0.
            else:
                raise ValueError(f'Unknown rec type {self.rec_type}')
    
    @torch.no_grad()
    def predict(self, x, u, output_quat: bool = False):
        x = self.preprocess_state(x)
        nan_alarm(x)
        
        xu_t = torch.cat([x, u], dim=-1) # (batch, state_dim + action_dim)
        xu_emb = self.xu_enc(xu_t).unsqueeze(0) # (1, batch, rec_latent_dim)
        if self.rec_num_layers > 0:
            if self.rec_type == 'lstm':
                xtp1_emb, (h, c) = self.rec(xu_emb, (self.aux_vars['h'], self.aux_vars['c']))
                self.aux_vars['h'], self.aux_vars['c'] = h, c
            elif self.rec_type == 'gru':
                xtp1_emb, h = self.rec(xu_emb, self.aux_vars['h'])
                self.aux_vars['h'] = h
            else:
                raise ValueError(f'Unknown rec type {self.rec_type}')
        else:
            xtp1_emb = xu_emb

        nan_alarm(xtp1_emb)
            
        if self.residual:
            xtp1 = x + self.x_dec(xtp1_emb.squeeze(0)) # (batch, state_dim)
        else:
            xtp1 = self.x_dec(xtp1_emb.squeeze(0))

        nan_alarm(xtp1)
            
        xtp1 = self.postprocess_state(xtp1, output_quat)
        nan_alarm(xtp1)
            
        return xtp1
        
