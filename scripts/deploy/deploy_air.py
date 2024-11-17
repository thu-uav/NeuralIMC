import argparse
import os

import hydra
import rospy
import torch_control
import yaml
from geometry_msgs.msg import Pose
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from torch_control.controllers.base import Controller
from torch_control.sim2real.air.controller_node import ControllerNode
from torch_control.tasks.base import GymEnv
from torch_control.utils.common import set_all_seed, load_config_from_wandb

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../../torch_control/configs")

@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="sim2real")
def launch(cfg):
    rospy.init_node('torchctrl')
    
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    if cfg.load_from is not None:
        assert os.path.exists(cfg.load_from), f"Checkpoint {cfg.load_from} does not exist!"
        wandb_cfg = load_config_from_wandb(cfg.load_from)
        seed = cfg.seed
        cfg = OmegaConf.merge(cfg, wandb_cfg)
        cfg.seed = seed

    if cfg.task.name == "track" and cfg.controller.algo_name == "ppo":
        cfg.controller.feature_extractor.task_obs.net = 'simple_tcn'

    cfg.device = 'cpu'
    cfg.wind.use_l1ac = True
    cfg.quadrotor.mass = 1.0
    cfg.random_dynamics = False
    cfg.random_init = False

    set_all_seed(cfg.seed)
    print("Set all seeds to", cfg.seed)
    
    if cfg.check_l1ac:
        cfg.wind.enable = True
        cfg.controller.obs.extrinsics = True
        cfg.wind.use_l1ac = True
        # cfg.controller.feature_extractor.ext_obs.net = None

    env = GymEnv.REGISTRY[cfg.task.name](cfg, 1)
    eval_env = GymEnv.REGISTRY[cfg.task.name](cfg, 1)
    
    controller = Controller.REGISTRY[cfg.controller.algo_name](
        cfg.controller, env, eval_env, device=cfg.device)
    
    controller.setup_params()

    print("[Deploy] Mask body-rate:", cfg.mask_bodyrate)

    if cfg.use_mocap:
        odom_topic = "/air/autopilot/state_estimate"
    else:
        odom_topic = "/air/t265/odom/sample"
    cmd_topic = "/air/autopilot/control_command_input"
    cmd_feedback_topic = "/air/control_command"
    start_topic = "/torchctrl/start"
    target_ref_topic = "/torchctrl/target_ref"
    logging_topic_prefix = "/torchctrl/logging"

    ctrl_node = ControllerNode(odom_topic=odom_topic, 
                               cmd_topic=cmd_topic,
                               cmd_feedback_topic=cmd_feedback_topic,
                               logging_topic_prefix=logging_topic_prefix,
                               start_topic=start_topic,
                               target_ref_topic=target_ref_topic,
                               check_l1ac=cfg.check_l1ac,
                               verbose=True)

    ctrl_node.setup_controller(env.robots.control_freq, 
                               *controller.export_deploy_funcs(),
                               mask_bodyrate=cfg.mask_bodyrate,
                               run_name=cfg.wandb.run_name)

    rospy.spin()
    
if __name__ == "__main__":
    launch()
    
