from typing import Dict

import argparse
import os
import random
import re
import sys
sys.path.append("..")

import hydra
import numpy as np
import time
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from torch_control.controllers.base import Controller
from torch_control.dynamics.quadrotor.quadrotor_state import QuadrotorState
from torch_control.sim2real.utils.low_pass_filter import LowPassFilter
from torch_control.tasks.base import GymEnv
from torch_control.utils.visualizer import Visualizer
from torch_control.utils.rot_utils import quat2euler
from torch_control.utils.timer import TimeReporter

from deploy.helper import DeployHelper

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../../torch_control/configs")

def set_all_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print("[Info] Setting all random seeds to {}".format(seed))
    
def get_init_odom():
    odom_dict = {'position': torch.tensor([0., 0., 1.]),
                 'orientation': torch.tensor([1., 0., 0., 0.]),
                 'linear_velocity': torch.tensor([0., 0., 0.]),
                 'angular_velocity': torch.tensor([0., 0., 0.])}
    for key, default_value in odom_dict.items():
        print(f"[Init] Please input the init {key}:")
        print(f"\t(Note: separated by space, or press enter to use default value {default_value.tolist()})")
        x_in = input()
        if x_in:
            odom_dict[key] = torch.tensor([float(x) for x in x_in.split(' ')])
    return odom_dict

def state2dict(state: QuadrotorState):
    return {'position': state.pos.squeeze(0),
            'orientation': state.quat.squeeze(0),
            'linear_velocity': state.vel.squeeze(0),
            'angular_velocity': state.ang_vel.squeeze(0),
            }
    
def ref2dict(ref: Dict):
    if 'orientation' in ref:
        angle = quat2euler(ref['orientation']).squeeze(0)
    elif 'angle' in ref:
        angle = ref['angle'].squeeze(0)
    elif 'yaw' in ref:
        angle = torch.zeros_like(ref['position'])
        angle[..., 2] = ref['yaw']
        angle = angle.squeeze(0)
    else:
        angle = torch.zeros_like(ref['position']).squeeze(0)
        
    if 'velocity' in ref:
        vel = ref['velocity'].squeeze(0)
    else:
        vel = torch.zeros_like(ref['position']).squeeze(0)
        
    pos = ref['position'].squeeze(0)
        
    return {'position': pos,
            'angle': angle,
            'linear_velocity': vel,}

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

@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="sim2real")
def main(cfg):
    overrided_args = HydraConfig.get().job.override_dirname
    if overrided_args:
        overrided_args = shorten_string(overrided_args)
    
    if cfg.task.name == "track" and cfg.controller.name == "ppo":
        cfg.controller.feature_extractor.task_obs.net = 'simple_tcn'
    if cfg.quadrotor.name == "train_random":
        cfg.eval.quadrotor = ["eval_random"]
        
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    
    set_all_seed(cfg.seed)
    cfg.task.max_time = 20
        
    env = GymEnv.REGISTRY[cfg.task.name](cfg, 1)
    eval_env = GymEnv.REGISTRY[cfg.task.name](cfg, 1)
    
    controller = Controller.REGISTRY[cfg.controller.name](
        cfg.controller, env, eval_env, device=cfg.device)
    
    helper = DeployHelper(ros_version=-1)
    
    if cfg.render:
        vis = Visualizer(ros_version=-1)
    
    controller.setup_params()
    
    init_task_fn, preprocess_fn, controller_fn, postprocess_fn \
        = controller.export_deploy_funcs()
    
    print("[Deploy] Mask body-rate:", cfg.mask_bodyrate)
    print("[Deploy] Obs low-pass filter:", cfg.obs_lpf_cutoff, "Hz")
    print("[Deploy] Cmd Low-pass filter:", cfg.cmd_lpf_cutoff, "Hz")
    
    init_odom = get_init_odom()
    
    tgt_pos, tgt_quat = helper.run()
    
    target_ref = {'position': torch.tensor(tgt_pos),
                  'orientation': torch.tensor(tgt_quat)}
        
    init_task_fn(init_odom, target_ref)
    
    start = input("Press enter to start the task...")
    
    state = eval_env.robots.state
    odom_dict = state2dict(state)
    
    for name, param in controller.actor_extractor.named_parameters():
        print(f"[actor_extractor] {name}: {param.shape}")
        
    for name, param in controller.actor.named_parameters():
        print(f"[actor] {name}: {param.shape}")
    
    timer = TimeReporter()
    timer.add_timer('whole')
    timer.add_timer('pre')
    timer.add_timer('call')
    timer.add_timer('post')
    
    for _ in range(1000):
        
        timer.start_timer('whole')
        
        timer.start_timer('pre')
        obs_tensor, _ = preprocess_fn(odom_dict, eval_env.time.item())
        timer.end_timer('pre')
        
        timer.start_timer('call')
        control_output = controller_fn(obs_tensor)
        timer.end_timer('call')
        
        timer.start_timer('post')
        cmd_tensor = postprocess_fn(control_output)
        timer.end_timer('post')
        
        timer.end_timer('whole')
        
    timer.report()
        
        
if __name__ == "__main__":
    main()
    