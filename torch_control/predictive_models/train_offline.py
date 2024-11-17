from typing import Dict

import argparse
import copy
import os
import random
import re
import sys
import tqdm
sys.path.append("..")

import hydra
import numpy as np
import time
import torch
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from torch_control.controllers.base import Controller
from torch_control.controllers.learning.modules import TrajectoryBuffer
from torch_control.dynamics.quadrotor.quadrotor_state import QuadrotorState
from torch_control.sim2real.utils.low_pass_filter import LowPassFilter
from torch_control.tasks.base import GymEnv
from torch_control.utils.visualizer import Visualizer
from torch_control.utils.rot_utils import quat2euler, matrix2euler
from torch_control.utils.wandb_utils import init_wandb
from torch_control.utils.common import set_all_seed, load_config_from_wandb
from torch_control.predictive_models import get_predictive_model

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../configs")

def shorten_string(s):
    # Splitting the string into arguments
    args = re.split(r',(?![^\[\]]*\])', s)

    # Keeping the most important part of each argument and handling list structures correctly
    shortened_args = []
    for arg in args:
        # Extract the key name from the last segment before '=' and retain the entire value
        key = arg.split('=')[0].split('.')[-1]
        value = arg.split('=')[1]
        shortened_arg = f"{key}={value}"
        shortened_args.append(shortened_arg)

    # Joining the arguments back into a string
    return ','.join(shortened_args)

def load_data(paths):
    all_data = {}
    for path in paths:
        data = np.load(path + '/data.npz')
        for key in data.keys():
            if key not in all_data:
                all_data[key] = data[key][:8000]
            else:
                all_data[key] = np.concatenate([all_data[key], data[key][:8000]], axis=0)

    return all_data

@torch.no_grad()
def eval(predictive_model, eval_data_path, eval_horizon, save_fig=False, save_dir=None, epoch_i=None, data_root=None):
    predictive_model.eval()
    
    data = np.load(f"{data_root}/data/{eval_data_path}/data.npz")
    gt_states, gt_actions = data['states'][-2000:], data['actions'][-2000:]
    
    batch_size, seq_len, state_dim = gt_states.shape
    predictive_model.reset_aux_vars(batch_size)
    
    pred_states = []
    
    def n2t(x):
        return torch.from_numpy(x).to(predictive_model.device)
        
    xt = n2t(gt_states[:, 0])
    pred_states.append(xt.cpu().numpy())
    for t in range(seq_len-1):
        if t % eval_horizon == 0:
            xt = n2t(gt_states[:, t])
            
        ut = n2t(gt_actions[:, t])
        
        xt = predictive_model.predict(xt, ut)
        pred_states.append(xt.cpu().numpy())
        
    predictive_model.train()
    
    pred_states = np.stack(pred_states, axis=1)
    
    mse = np.mean((gt_states - pred_states)**2)
    pos_ref, vel_ref, mat_ref, angvel_ref = np.split(gt_states, [3, 6, 15], axis=-1)
    pos_est, vel_est, mat_est, angvel_est = np.split(pred_states, [3, 6, 15], axis=-1)
    
    batch_shape = mat_ref.shape[:-1]
    
    ang_ref = matrix2euler(mat_ref.reshape(-1, 3, 3)).reshape(*batch_shape, 3).detach().cpu().numpy()
    ang_est = matrix2euler(mat_est.reshape(-1, 3, 3)).reshape(*batch_shape, 3).detach().cpu().numpy()
    
    ang_ref, ang_est, angvel_ref, angvel_est = ang_ref * 180 / np.pi, ang_est * 180 / np.pi, angvel_ref, angvel_est
    
    pos_mse = np.mean((pos_ref - pos_est)**2)
    vel_mse = np.mean((vel_ref - vel_est)**2)
    ang_mse = np.mean((ang_ref - ang_est)**2)
    angvel_mse = np.mean((angvel_ref - angvel_est)**2)
    
    print("--------------------")
    print(f"[Eval] {eval_data_path} @ Horizon {eval_horizon}")
    print(f"MSE: {mse}, Pos MSE: {pos_mse}, Vel MSE: {vel_mse}, Att MSE: {ang_mse}, AngVel MSE: {angvel_mse}")
    
    if save_fig:
        import matplotlib.pyplot as plt
        FIG_PATH = f"{save_dir}/{eval_data_path}-{eval_horizon}/Epoch{epoch_i}"
        if not os.path.exists(FIG_PATH):
            os.makedirs(FIG_PATH)
        for img_i in [0, 5, 10]:
            # plot position x, y, z
            fig, ax = plt.subplots(3, 1, figsize=(5, 5))
            for i in range(3):
                ax[i].plot(pos_ref[img_i, :, i], label='gt')
                ax[i].plot(pos_est[img_i, :, i], label='pred')
                ax[i].set_title(f"Position {i}")
                ax[i].legend()
            fig.tight_layout()
            fig.savefig(f"{FIG_PATH}/pos_{img_i}.png")
            plt.close()
            # plot velocity x, y, z
            fig, ax = plt.subplots(3, 1, figsize=(5, 5))
            for i in range(3):
                ax[i].plot(vel_ref[img_i, :, i], label='gt')
                ax[i].plot(vel_est[img_i, :, i], label='pred')
                ax[i].set_title(f"Velocity {i}")
                ax[i].legend()
            fig.tight_layout()
            fig.savefig(f"{FIG_PATH}/vel_{img_i}.png")
            plt.close()
            # plot attitude roll, pitch, yaw
            fig, ax = plt.subplots(3, 1, figsize=(5, 5))
            for i in range(3):
                ax[i].plot(ang_ref[img_i, :, i], label='gt')
                ax[i].plot(ang_est[img_i, :, i], label='pred')
                ax[i].set_title(f"Attitude {i}")
                ax[i].legend()
            fig.tight_layout()
            fig.savefig(f"{FIG_PATH}/ang_{img_i}.png")
            plt.close()
            # plot angular velocity x, y, z
            fig, ax = plt.subplots(3, 1, figsize=(5, 5))
            for i in range(3):
                ax[i].plot(angvel_ref[img_i, :, i], label='gt')
                ax[i].plot(angvel_est[img_i, :, i], label='pred')
                ax[i].set_title(f"Angular Velocity {i}")
                ax[i].legend()
            fig.tight_layout()
            fig.savefig(f"{FIG_PATH}/angvel_{img_i}.png")
            plt.close()
            
    return {'mse': mse, 'pos_mse': pos_mse, 'vel_mse': vel_mse, 'ang_mse': ang_mse, 'angvel_mse': angvel_mse}

@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="offline")
def main(cfg):
    overrided_args = HydraConfig.get().job.override_dirname

    if cfg.task.name == "track" and cfg.controller.algo_name == "ppo":
        cfg.controller.feature_extractor.task_obs.net = "simple_tcn"
    if cfg.quadrotor.name == "train_random":
        cfg.eval.quadrotor = ["eval_random"]
    if cfg.task.name == "track":
        cfg.wandb.run_name += f"_{cfg.task.trajectory.type}"
    if overrided_args:
        cfg.wandb.run_name += "-" + shorten_string(overrided_args)
    if cfg.exp_name is not None:
        cfg.wandb.run_name = cfg.exp_name

    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    set_all_seed(cfg.seed)
    
    model_cfg = cfg.predictive_model
    offline_cfg = copy.deepcopy(cfg.offline_train)

    model_cfg.transformer.max_context_length = min(model_cfg.transformer.max_context_length, 
                                                   offline_cfg.train.horizon)
    
    wandb_run = init_wandb(cfg)
    data_root = offline_cfg.data_root

    in_buffer_device = offline_cfg.in_buffer_device
    save_dir = offline_cfg.save_dir
    if save_dir is None:
        save_dir = f"outputs_offline/{cfg.wandb.run_name}/"
    ckpt_dir = os.path.join(save_dir, "checkpoints")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
        
    print("Save dir: ", save_dir)
    
    # dump model_cfg and offline_cfg
    with open(f"{ckpt_dir}/model_cfg.yaml", 'w') as f:
        OmegaConf.save(model_cfg, f)
    with open(f"{ckpt_dir}/offline_cfg.yaml", 'w') as f:
        OmegaConf.save(offline_cfg, f)

    train_tasks = [offline_cfg.train.task] if isinstance(offline_cfg.train.task, str) else offline_cfg.train.task
    train_models = [offline_cfg.train.model] if isinstance(offline_cfg.train.model, str) else offline_cfg.train.model
    # train_horizon = int(offline_cfg.train.horizon)
    train_horizon = int(model_cfg.get(model_cfg.arch).train_horizon)
    
    eval_tasks = [offline_cfg.eval.task] if isinstance(offline_cfg.eval.task, str) else offline_cfg.eval.task
    if eval_tasks is None:
        eval_tasks = train_tasks
    eval_models = [offline_cfg.eval.model] if isinstance(offline_cfg.eval.model, str) else offline_cfg.eval.model
    if eval_models is None:
        eval_models = train_models
    eval_horizons = [offline_cfg.eval.horizon] if isinstance(offline_cfg.eval.horizon, int) else offline_cfg.eval.horizon
    if eval_horizons is None:
        eval_horizons = [train_horizon]
    
    for task in train_tasks:
        for model in train_models:
            train_sets = [f"{data_root}/data/{model}_{task}_{offline_cfg.ctrl}"]

    # load dataset
    dataset = load_data(train_sets)

    traj_buffer = TrajectoryBuffer(buffer_size=np.prod(dataset['states'].shape[:2]),
                                   episode_length=dataset['states'].shape[1],
                                   state_dim=dataset['states'].shape[2],
                                   action_dim=dataset['actions'].shape[2],
                                   num_envs=256, # will not be used
                                   device=in_buffer_device,
                                   use_priority=False, # will not be used
                                   )

    traj_buffer.max_num_episodes = dataset['states'].shape[0]
    traj_buffer.episode_length = dataset['states'].shape[1]
    traj_buffer.buffer_size = traj_buffer.max_num_episodes * traj_buffer.episode_length
    traj_buffer.states = torch.from_numpy(dataset['states']).to(in_buffer_device)
    traj_buffer.actions = torch.from_numpy(dataset['actions']).to(in_buffer_device)
    traj_buffer.active_mask[:] = True
    traj_buffer.used_times[:] = 0
    
    _, max_ep_len, state_dim = traj_buffer.states.shape
    action_dim = traj_buffer.actions.shape[-1]
    
    # create predictive model
    predictive_model = get_predictive_model(model_arch=model_cfg.arch,
                                            cfg=model_cfg.get(model_cfg.arch),
                                            state_dim=traj_buffer.states.shape[-1],
                                            action_dim=traj_buffer.actions.shape[-1],
                                            max_ep_len=max_ep_len,
                                            rot_mode=model_cfg.rot_mode,
                                            device=cfg.device,)

    print("[PredictiveModel] # of parameters: ", sum(p.numel() for p in predictive_model.parameters()))

    wandb.log({'model_params': sum(p.numel() for p in predictive_model.parameters())})
    
    best_loss = np.inf
    
    # train and eval
    for epoch in range(offline_cfg.max_epoches):
        print(f"------ Epoch {epoch} -----")
        sampler = traj_buffer.get_sampler(minibatch_size=offline_cfg.minibatch_size, 
                                          horizon=train_horizon+1,
                                          num_minibatch=offline_cfg.num_minibatch,
                                          device=cfg.device)

        infos = {
            'train/pred_state_loss': [],
            'train/grad_norm': []
        }
        
        for batch in sampler:
            train_info = predictive_model.update_step(batch.state, batch.action)
            
            infos['train/pred_state_loss'].append(train_info['pred_state_loss'].mean().item())
            infos['train/grad_norm'].append(train_info['grad_norm'].item())
            for state_i in range(train_info['pred_state_loss'].shape[-1]):
                info_name = f'train/pred_state_loss/{state_i}'
                if info_name not in infos:
                    infos[info_name] = []
                infos[info_name].append(train_info['pred_state_loss'][..., state_i].mean().item())
            if 'train/pred_action_loss' in train_info:
                if 'train/pred_action_loss' not in infos:
                    infos['train/pred_action_loss'] = []
                infos['train/pred_action_loss'].append(train_info['pred_action_loss'].mean().item())
                for action_i in range(train_info['pred_action_loss'].shape[-1]):
                    info_name = f'train/pred_action_loss/{action_i}'
                    if info_name not in infos:
                        infos[info_name] = []
                    infos[info_name].append(train_info['pred_action_loss'][..., action_i].mean().item())

        predictive_model.scheduler.step(np.mean(infos['train/pred_state_loss']))
                    
        best_loss = min(best_loss, np.mean(infos['train/pred_state_loss']))
        is_best = best_loss == np.mean(infos['train/pred_state_loss'])

        print("IS BEST: ", is_best, "BEST LOSS: ", best_loss)
        for key in infos:
            wandb.log({key: np.mean(infos[key])})
            print(f"{key}: \t{np.mean(infos[key])}")

        if epoch % offline_cfg.eval_interval == 0 or epoch == offline_cfg.max_epoches-1:
            for eval_task in eval_tasks:
                for eval_model in eval_models:
                    for eval_horizon in eval_horizons:
                        eval_data_path = f"{eval_model}_{eval_task}_{offline_cfg.ctrl}"
                        eval_info = eval(predictive_model, eval_data_path, eval_horizon, 
                                         save_dir=save_dir, save_fig=True,
                                         epoch_i=epoch, data_root=data_root)
                        for key in eval_info:
                            wandb.log({f'eval/{eval_model}-{eval_task}/horizon_{eval_horizon}/{key}': eval_info[key]})
        
        # save model
        if is_best:
            torch.save(predictive_model.state_dict(), f"{ckpt_dir}/best_model.pth")
            print(f"Saved best model to {save_dir}/best_model.pth")
        
        if epoch % offline_cfg.save_interval == 0:
            torch.save(predictive_model.state_dict(), f"{ckpt_dir}/model_{epoch}.pth")
            
    os.system(f"kill -9 {os.getpid()}")
    wandb_run.finish()

if __name__ == "__main__":
    main()
            
            
    
    