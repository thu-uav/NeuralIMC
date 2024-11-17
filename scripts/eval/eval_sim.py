from typing import Dict

import argparse
import copy
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
from torch_control.utils.common import set_all_seed, load_config_from_wandb

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../../torch_control/configs")

def get_init_odom():
    odom_dict = {'position': torch.tensor([0., 0., 0.]),
                 'orientation': torch.tensor([1., 0., 0., 0.]),
                 'linear_velocity': torch.tensor([0., 0., 0.]),
                 'angular_velocity': torch.tensor([0., 0., 0.])}
    # for key, default_value in odom_dict.items():
    #     print(f"[Init] Please input the init {key}:")
    #     print(f"\t(Note: separated by space, or press enter to use default value {default_value.tolist()})")
    #     x_in = input()
    #     if x_in:
    #         odom_dict[key] = torch.tensor([float(x) for x in x_in.split(' ')])
    return odom_dict

def state2dict(state: QuadrotorState):
    return {'position': state.pos.squeeze(0),
            'orientation': state.quat.squeeze(0),
            'linear_velocity': state.vel.squeeze(0),
            'angular_velocity': state.ang_vel.squeeze(0),
            }

def ref2dict(ref: Dict):
    if 'orientation' in ref:
        angle = quat2euler(ref['orientation'])
    elif 'angle' in ref:
        angle = ref['angle']
    elif 'yaw' in ref:
        angle = torch.zeros_like(ref['position'])
        angle[..., 2] = ref['yaw']
        angle = angle
    else:
        angle = torch.zeros_like(ref['position'])
        
    if 'linear_velocity' in ref:
        vel = ref['linear_velocity']
    else:
        vel = torch.zeros_like(ref['position'])
        
    pos = ref['position']
        
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

def euler2quat(ang):
    """
    Convert Euler angles (roll, pitch, yaw) to quaternions (qw, qx, qy, qz)
    """
    roll, pitch, yaw = [x * np.pi / 180 for x in ang]

    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return [qw, qx, qy, qz]

@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="sim2real")
def main(cfg):
    overrided_args = HydraConfig.get().job.override_dirname
    if overrided_args:
        overrided_args = shorten_string(overrided_args)
        
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    
    if cfg.load_from is not None:
        assert os.path.exists(cfg.load_from), f"Checkpoint {cfg.load_from} does not exist!"
        wandb_cfg = load_config_from_wandb(cfg.load_from)
        seed = cfg.seed
        cfg = OmegaConf.merge(cfg, wandb_cfg)
        cfg.seed = seed
        
    if cfg.task.name == "track" and cfg.controller.name == "ppo":
        cfg.controller.feature_extractor.task_obs.net = 'simple_tcn'
    if cfg.quadrotor.name == "train_random":
        cfg.eval.quadrotor = ["eval_random"]

    set_all_seed(1) #cfg.seed)
    # cfg.controller.actor.feature_extractor = cfg.controller.feature_extractor
    
    # cfg.task.max_time = 20
    cfg.wandb.mode = "disabled"
    # wandb_run = init_wandb(cfg)
    # cfg.obs.extrinsics = True
    cfg.wind.use_l1ac = True
        
    env = GymEnv.REGISTRY[cfg.task.name](cfg, 1)
    eval_env = GymEnv.REGISTRY[cfg.task.name](cfg, 1)
    
    controller = Controller.REGISTRY[cfg.controller.algo_name](
        cfg.controller, env, eval_env, device=cfg.device)
    controller.setup_params()
    
    # helper = DeployHelper(ros_version=-1)
    
    if cfg.render:
        vis = Visualizer(ros_version=-1)
    
    init_task_fn, ctrl_fn = controller.export_deploy_funcs()
    
    print("[Deploy] Mask body-rate:", cfg.mask_bodyrate)
    
    init_odom = get_init_odom()
    
    # tgt_pos, tgt_quat = helper.run()
    tgt_pos = [0., 0., 0.]
    tgt_quat = euler2quat([0., 0., 0.])
    
    target_ref = {'position': torch.tensor(tgt_pos),
                  'orientation': torch.tensor(tgt_quat)}
        
    init_task_fn(init_odom, target_ref)
    
    mask_bodyrate = cfg.mask_bodyrate
        
    state_list = []
    ref_list = []
    action_list = []
    time_list = []
    data_list = {}
    
    # start = input("Press enter to start the task...")
    
    state = eval_env.robots.state
    t0 = time.time()
    # t = 0.
    
    if cfg.render:
        vis.render(state.pos, state.quat, mode='quat')
    
    while True:
        start_time = time.time()
        odom_dict = state2dict(state)
        
        state_list.append(odom_dict)
        time_list.append(eval_env.time.item())
        
        cmd_tensor, logging_data, logging_info = ctrl_fn(odom_dict, eval_env.time.item())
        ref_list.append(ref2dict(logging_info['ref']))
        for key, value in logging_data.items():
            if key not in data_list:
                data_list[key] = []
            data_list[key].append(value)
        
        if mask_bodyrate:
            cmd_tensor[..., 1:] = 0.
        
        action_list.append(cmd_tensor)
        
        # dt = np.random.uniform(0.02, 0.05)
        dt = 0.02
        state, done = eval_env.eval_step(cmd_tensor.unsqueeze(0), dt)
        step_time = time.time()
        
        if cfg.render:
            vis.render(state.pos, state.quat, mode='quat')

        if done.any():
            break
        
        elapsed_time = time.time() - start_time
        
        # if elapsed_time < 1./eval_env.robots.control_freq:
        #     time.sleep(1./eval_env.robots.control_freq - elapsed_time)
        # if elapsed_time > 1./eval_env.robots.control_freq:
        #     print("[Warning] Control loop is slower than expected!")
        #     print(f"[Progress] Elapsed time: {elapsed_time}, expect_time = {1./eval_env.robots.control_freq}")

    # plot the trajectory
    sim_pos = np.array([s['position'].tolist() for s in state_list])
    sim_ang = np.array([quat2euler(s['orientation']).tolist() for s in state_list])
    sim_vel = np.array([s['linear_velocity'].tolist() for s in state_list])
    sim_angvel = np.array([s['angular_velocity'].tolist() for s in state_list])
    
    ref_pos = np.array([r['position'].tolist() for r in ref_list])
    ref_ang = np.array([r['angle'].tolist() for r in ref_list])
    ref_vel = np.array([r['linear_velocity'].tolist() for r in ref_list])
    
    sim_cmd = np.array([a.tolist() for a in action_list])
    sim_t = np.array(time_list)
    
    # compute pos error
    pos_error = np.linalg.norm(sim_pos - ref_pos, axis=-1)
    ang_error = np.linalg.norm(sim_ang - ref_ang, axis=-1)
    vel_error = np.linalg.norm(sim_vel - ref_vel, axis=-1)
    
    print(f"[Error.mean] pos_error = {pos_error.mean()}, ang_error = {ang_error.mean()}, vel_error = {vel_error.mean()}")
    print(f"[Error.std] pos_error = {pos_error.std()}, ang_error = {ang_error.std()}, vel_error = {vel_error.std()}")
    print(f"[Control.std] thrust_std = {sim_cmd[:, 0].std(axis=0)}, bodyrate_std = {sim_cmd[:, 1:].std(axis=0)}")
    print(f"[Final Error] pos_error = {pos_error[-1]}, ang_error = {ang_error[-1]}, vel_error = {vel_error[-1]}")
    
    # time_str = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    LOG_PATH = f"./data/{cfg.wandb.run_name}"
    if not os.path.exists("./data"):
        os.makedirs("./data")
    print(f"LOG DIR: {LOG_PATH}")
    
    with open("./data/0_log.txt", "a") as f:
        f.writelines(f"""
        {cfg.wandb.run_name}
            mean pos_error = {pos_error.mean()}, ang_error = {ang_error.mean()}, vel_error = {vel_error.mean()}, 
            std pos_error = {pos_error.std()}, ang_error = {ang_error.std()}, vel_error = {vel_error.std()}
            final pos_error = {pos_error[-1]}, final_ang_error = {ang_error[-1]}, final_vel_error = {vel_error[-1]}
            control thrust_std = {sim_cmd[:, 0].std(axis=0)}, bodyrate_std = {sim_cmd[:, 1:].std(axis=0)}
        """)
    # np.savez(LOG_PATH + "_save" + ".npz", 
    #          sim_pos=sim_pos, sim_ang=sim_ang, sim_vel=sim_vel, sim_angvel=sim_angvel, 
    #          ref_pos=ref_pos, ref_ang=ref_ang, ref_vel=ref_vel, sim_cmd=sim_cmd, sim_t=sim_t)
    # np.savez(LOG_PATH + ".npz", **{
    #          key: np.array(value) for key, value in data_list.items()})
    
    import matplotlib.pyplot as plt
    plt.subplots(4, 4, figsize=(16, 16))
    # pos (x, y, z)
    for i in range(3):
        plt.subplot(4, 4, i+1)
        plt.plot(sim_t, sim_pos[:, i], label='real')
        plt.plot(sim_t, ref_pos[:, i], label='ref')
        plt.legend()
        plt.title(f"pos_{i}")
    # ang (roll, pitch, yaw)
    for i in range(3):
        plt.subplot(4, 4, i+5)
        plt.plot(sim_t, sim_ang[:, i] * 180 / np.pi, label='real')
        plt.plot(sim_t, ref_ang[:, i] * 180 / np.pi, label='ref')
        plt.legend()
        plt.title(f"ang_{i}")
    # vel (x, y, z)
    for i in range(3):
        plt.subplot(4, 4, i+9)
        plt.plot(sim_t, sim_vel[:, i], label='real')
        plt.plot(sim_t, ref_vel[:, i], label='ref')
        plt.legend()
        plt.title(f"vel_{i}")
    # angvel (x, y, z)
    for i in range(3):
        plt.subplot(4, 4, i+13)
        plt.plot(sim_t, sim_angvel[:, i] * 180 / np.pi, label='real')
        plt.legend()
        plt.title(f"angvel_{i}")
    # action (thrust, roll, pitch, yaw)
    for i in range(4):
        plt.subplot(4, 4, (i + 1) * 4)
        plt.plot(sim_t, sim_cmd[:, i], label='real')
        plt.legend()
        plt.title(f"cmd_{i}")
        
    img_path = LOG_PATH + ".png"
    plt.tight_layout()
    plt.savefig(img_path)
    
    print("Saved to {}".format(img_path))
    
if __name__ == "__main__":
    main()
    
