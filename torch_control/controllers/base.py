import abc
from typing import Any, Callable, Dict, Tuple, Type

import numpy as np
import torch
from torch_control.utils.rot_utils import quat2euler
from torch_control.utils.timer import TimeReporter
from torch_control.utils.visualizer import Visualizer
from torch_control.tasks.base import GymEnv
from torch_control.tasks.setpoint import SetPoint
from torch_control.tasks.track import Track

try:
    import wandb
except:
    print("[WARNING] wandb not installed, will not log to wandb")


class Controller(abc.ABC):

    REGISTRY: Dict[str, Type["Controller"]] = {}

    def __init__(self, cfg, envs, eval_envs_dict, device: str = "cuda"):
        self.cfg = cfg
        self.envs = envs
        
        if isinstance(eval_envs_dict, dict):
            self.eval_envs_dict = eval_envs_dict
        else:
            self.eval_envs_dict = {"train": eval_envs_dict}
        self.eval_counts = 0
        if hasattr(self, "eval_envs_dict"):
            self.min_eval_errors = {env_name: np.inf for env_name in self.eval_envs_dict.keys()}
        self.device = device

        self.timer = TimeReporter()
        self.timer.add_timer("tune_params")
        self.timer.add_timer("eval")
        self.timer.add_timer("inference")

        if cfg is not None:
            self.save_dir = cfg.get("save_dir", None)

    @abc.abstractclassmethod
    def setup_params(self, *args, **kwargs) -> None:
        """
        Setup the parameters of the controller. 
        For NN controllers, this will include the network architecture and optimizer. 
        For classic controllers, this will include the gains.
        """
        raise NotImplementedError
    
    def tune_params(self, only_eval: bool = False, render: bool = False, plot: bool = False, *args, **kwargs) -> None:
        """
        A wrapper for _tune_params, which includes timing and final evaluation.
        """
        if not only_eval:
            self.timer.start_timer("tune_params")
            self._tune_params(*args, **kwargs)
            self.timer.end_timer("tune_params")

        eval_info = self.eval(render=render, plot=plot)
        eval_info["time/tune_params"] = self.timer.get_time("tune_params")
        self.log(eval_info, step=0, print_context="Final Evaluation", log_context="final_eval")
        if not only_eval:
            self.save_params("final")

    @abc.abstractclassmethod
    def _tune_params(self, *args, **kwargs) -> None:
        """
        Internal function for tuning the parameters of the controller.
        For NN controllers, this will include training the network with specific hyperparameters.
        For classic controllers, this will include tuning the gains.
        """
        raise NotImplementedError
    
    @abc.abstractclassmethod
    def save_params(self, *args, **kwargs) -> None:
        """
        Save the parameters of the controller.
        """
        raise NotImplementedError

    @abc.abstractclassmethod
    def __call__(self, obs: torch.Tensor, reference: Any = None) -> torch.Tensor:
        """
        Return the control action given the current obs and reference.
        """
        raise NotImplementedError
    
    @abc.abstractclassmethod
    def export_deploy_funcs(self, *args, **kwargs) -> Tuple[Callable, Callable, Callable]:
        """
        Export the functions for deployment.
        :return preprocess_fn: function that preprocesses the state and reference
        :return controller_fn: function that runs the controller
        :return postprocess_fn: function that postprocesses the control output
        """
        raise NotImplementedError
    
    @abc.abstractclassmethod
    def set_train(self) -> None:
        """
        Set the controller to train mode.
        """
        raise NotImplementedError
    
    @abc.abstractclassmethod
    def set_eval(self) -> None:
        """
        Set the controller to eval mode.
        """
        raise NotImplementedError

    def eval(self, render: bool = False, plot: bool = False) -> Dict[str, Any]:
        """
        Evaluate the controller on the evaluation environments.
        """
        self.set_eval()
        eval_info = {}
        for eval_name, eval_envs in self.eval_envs_dict.items():
            print("========================================")
            print(f"Evaluating on {eval_name} environments")
            print("----------------------------------------")
            eval_info.update(self.eval_once(eval_envs, eval_name, render=render, plot=plot))
            eval_error = eval_info[f"{eval_name}/metrics/error/error_distance(m)"]
            if eval_error < self.min_eval_errors[eval_name]:
                self.min_eval_errors[eval_name] = eval_error
                eval_info[f"best_error_distance(m)/{eval_name}"] = self.min_eval_errors[eval_name]
                self.save_params(f"best_{eval_name}")
        return eval_info

    @torch.no_grad()
    def eval_once(self, eval_envs, eval_name, render: bool = False, plot: bool = False) -> Dict[str, Any]:
        """
        Evaluate the controller on the evaluation environment.
        """
        eval_envs.clear_grad()

        done = torch.zeros(eval_envs.num_envs, dtype=torch.bool).to(self.device)

        eval_returns = torch.zeros(eval_envs.num_envs).to(self.device)
        eval_lengths = torch.zeros(eval_envs.num_envs).to(self.device)
        eval_metrics = {m: torch.zeros(eval_envs.num_envs).to(self.device) 
                        for m in self.envs.eval_metrics}
        eval_success = torch.zeros(eval_envs.num_envs).to(self.device)

        plot_data = {'time': [],
                     'done': [],
                    'cmd': [],
                    'pos': [],
                    'vel': [],
                    'quat': [],
                    'target_pos': [],
                    'target_vel': [],
                    'target_quat': [],
        }

        self.timer.start_timer("eval")
        obs = eval_envs.reset()
        
        self.init_aux_vars(eval_envs.num_envs)
        
        if render:
            vis = Visualizer(num_drones=9)
            render_idx = torch.randint(0, eval_envs.num_envs, (9,))
            # TODO: add api to get state and goal state
            render_pos = eval_envs.robots.state.pos[render_idx]
            render_quat = eval_envs.robots.state.quat[render_idx]
            vis.set_state(render_pos.cpu(), render_quat.cpu())
            if isinstance(eval_envs, SetPoint):
                vis.set_goal_state(
                    eval_envs.target_pos[render_idx].cpu(), 
                    quat2euler(eval_envs.target_quat[render_idx].cpu()))
            else:
                target = eval_envs.get_task_reference()
                vis.set_goal_state(
                    target['position'][render_idx].cpu(), 
                    quat2euler(target['orientation'][render_idx].cpu()))

        while not done.all():
            self.timer.start_timer("inference")

            plot_data['time'].append(eval_envs.time.cpu().numpy())
            plot_data['done'].append(done.cpu().numpy())
            plot_data['pos'].append(eval_envs.robots.state.pos.cpu().numpy())
            plot_data['vel'].append(eval_envs.robots.state.vel.cpu().numpy())
            plot_data['quat'].append(eval_envs.robots.state.quat.cpu().numpy())

            if isinstance(eval_envs, SetPoint):
                plot_data['target_pos'].append(eval_envs.target_pos.cpu().numpy())
                plot_data['target_vel'].append(torch.zeros_like(eval_envs.target_pos).cpu().numpy())
                plot_data['target_quat'].append(eval_envs.target_quat.cpu().numpy())
            else:
                target = eval_envs.get_task_reference()
                plot_data['target_pos'].append(target['position'].cpu().numpy())
                plot_data['target_vel'].append(target['linear_velocity'].cpu().numpy())
                plot_data['target_quat'].append(target['orientation'].cpu().numpy())

            if self.cfg.algo_name in ['pid', 'mppi', 'mppi_resrl']:
                assert isinstance(eval_envs, Track), "Only Track task is supported for pid and mppi controller"
                state = eval_envs.state.clone()
                ref = eval_envs.get_task_reference()
                time = eval_envs.time
                # reference_fn = eval_envs.get_task_reference
                def reference_fn(time):
                    ref = eval_envs.get_task_reference(time)
                    pos = ref.get('position')
                    vel = ref.get('linear_velocity', torch.zeros_like(pos))
                    quat = ref.get('orientation', torch.zeros(pos.shape[:-1] + (4,)))
                    ref_vec = torch.cat([pos, vel, quat], dim=-1)
                    return ref_vec

                env_obs = obs.clone()
                obs = {'state': state, 'reference': ref, 'time': time, 'reference_fn': reference_fn, 'env_obs': env_obs}

            action = self.__call__(obs)
            self.timer.end_timer("inference")
            plot_data['cmd'].append(action.cpu().numpy())
            obs, reward, done_step, info = eval_envs.step(action)
            done_step = done_step.bool()
            
            # only update evaluation results for envs that are first done
            just_done = done_step & ~done
            eval_returns[just_done] = info["episode_return"][just_done]
            eval_lengths[just_done] = info["episode_length"][just_done]
            for m in eval_metrics.keys():
                eval_metrics[m][just_done] = info["episode_" + m][just_done]
            eval_success[just_done] = (info["episode_length"][just_done] >= eval_envs.max_episode_length).float()
                
            # update done mask
            done |= done_step
            
            if render:
                render_pos = eval_envs.robots.state.pos[render_idx]
                render_quat = eval_envs.robots.state.quat[render_idx]
                vis.set_state(render_pos.cpu(), render_quat.cpu())

        self.timer.end_timer("eval")

        eval_info = {
            f"{eval_name}/metrics/return": eval_returns.mean().item(),
            f"{eval_name}/metrics_std/return": eval_returns.std().item(),
            f"{eval_name}/metrics/length": eval_lengths.mean().item(),
            f"{eval_name}/metrics_std/length": eval_lengths.std().item(),
            f"{eval_name}/metrics/success": eval_success.mean().item(),
            f"{eval_name}/metrics_std/success": eval_success.std().item(),
            f"{eval_name}/time/eval": self.timer.get_time("eval", mean=True),
            f"{eval_name}/time/inference": self.timer.get_time("inference", mean=True),
            **{f'{eval_name}/metrics/{m.split("_")[0]}/{m}': eval_metrics[m].mean().item() for m in eval_metrics},
            **{f'{eval_name}/metrics_std/{m.split("_")[0]}/{m}': eval_metrics[m].std().item() for m in eval_metrics},
        }

        if plot:
            plot_eval_data(data_dict=plot_data, save_dir=self.save_dir)

        return eval_info
    
    def log(self, info, step: int, print_context: str, log_context: str):
        """
        Log the information to the logger.
        """
        print("========================================")
        print(print_context)
        print("----------------------------------------")
        f = open(f"{self.save_dir}/output_log.txt", "a")
        for k, v in info.items():
            print(f"{k}: {v:.6f}")
            f.write(f"{k}: {v:.6f}\n")
            try:
                wandb.log({log_context + "/" + k: v}, step=step)
            except:
                # print("[WARNING] wandb not installed, will not log to wandb")
                pass
        f.close()

def register_controller(name, cls):
    if name in Controller.REGISTRY:
        raise ValueError(f"Cannot register duplicate controller ({name})")
    else:
        print(f"[Info] Registering controller: {name}")
    Controller.REGISTRY[name] = cls
    Controller.REGISTRY[name.lower()] = cls
    
def get_controller(name):
    if name not in Controller.REGISTRY:
        raise ValueError(f"Cannot find controller ({name})")
    return Controller.REGISTRY[name]

def plot_eval_data(data_dict, save_dir):
    done = np.array(data_dict['done'])
    time = np.array(data_dict['time'])
    cmd = np.array(data_dict['cmd'])
    pos = np.array(data_dict['pos'])
    vel = np.array(data_dict['vel'])
    quat = np.array(data_dict['quat'])
    target_pos = np.array(data_dict['target_pos'])
    target_vel = np.array(data_dict['target_vel'])
    target_quat = np.array(data_dict['target_quat'])

    import matplotlib.pyplot as plt
    from torch_control.utils import rot_utils as ru

    # plot env with max pos_error
    # pos_error = np.mean(np.linalg.norm(pos - target_pos, axis=-1)[~done], axis=0)
    pos_error = np.linalg.norm(pos - target_pos, axis=-1) * (~done).astype(np.float32)
    pos_error = np.mean(pos_error, axis=0)
    top_5 = np.argsort(pos_error)[-5:]

    for plot_id in top_5:
        error = pos_error[plot_id]
        valid_steps = ~done[:, plot_id]
        fig, axs = plt.subplots(4, 4, figsize=(10, 10))

        # pos: first row (x, y, z)
        for i in range(3):
            axs[0, i].plot(time[valid_steps, plot_id], pos[valid_steps, plot_id, i], label='pos')
            axs[0, i].plot(time[valid_steps, plot_id], target_pos[valid_steps, plot_id, i], label='target pos')
            axs[0, i].set_title(['x', 'y', 'z'][i])
            axs[0, i].legend()
        # vel: second row (vx, vy, vz)
        for i in range(3):
            axs[1, i].plot(time[valid_steps, plot_id], vel[valid_steps, plot_id, i], label='vel')
            axs[1, i].plot(time[valid_steps, plot_id], target_vel[valid_steps, plot_id, i], label='target vel')
            axs[1, i].set_title(['vx', 'vy', 'vz'][i])
            axs[1, i].legend()
        # ang: third row (roll, pitch, yaw)
        ang = ru.np_quat2euler(quat)
        target_ang = ru.np_quat2euler(target_quat)
        for i in range(3):
            axs[2, i].plot(time[valid_steps, plot_id], ang[valid_steps, plot_id, i], label='ang')
            axs[2, i].plot(time[valid_steps, plot_id], target_ang[valid_steps, plot_id, i], label='target ang')
            axs[2, i].set_title(['roll', 'pitch', 'yaw'][i])
            axs[2, i].legend()
        # cmd: fourth row (thrust, roll, pitch, yaw)
        for i in range(4):
            axs[3, i].plot(time[valid_steps, plot_id], cmd[valid_steps, plot_id, i], label='cmd')
            axs[3, i].set_title(['thrust', 'roll', 'pitch', 'yaw'][i])
            axs[3, i].legend()

        plt.tight_layout()
        plt.suptitle(f"Drone {plot_id}")
        plt.savefig(f"{save_dir}/drone_{plot_id}_{error:.3f}.png")


