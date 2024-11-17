import math
from typing import Dict, List, Tuple

import torch
import torch.distributions as D
from torch_control.dynamics.quadrotor.quadrotor_state import QuadrotorState
from torch_control.utils import integrators
from torch_control.utils import math_utils as mu
from torch_control.utils import rot_utils as ru
from torch_control.controllers.l1ac import L1AC_batch

def to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    else:
        return torch.as_tensor(x)

class Quadrotor:
    intrinsics: List[str] = [
        'mass', 'arm_length', 'l2w_ratio', 'l2h_ratio', 'center_of_mass', # rigid-body parameters
        'motor_rot_dirs', 'motor_spread_angles', # drone configuration
        'motor_kf', 'motor_kappa', 'motor_tc', 'max_thrust2weight', # motor parameters
        'Kp'] # control parameters

    def __init__(self, 
                 cfg, 
                 num_instances: int = 1, 
                 device: str = 'cuda', 
                 is_sysid: bool = False, 
                 wind_config: Dict = {},
                 linear_std: float = 0.,
                 angular_std: float = 0.
                 ):
        self.N = num_instances
        self.device = device
        self.is_sysid = is_sysid
        self.use_simple_dynamics = cfg.use_simple_dynamics
        self.wind_config = wind_config
        self.linear_std = linear_std
        self.angular_std = angular_std
        
        # when using simple dynamics, control mode must be 'ctbr'
        if self.use_simple_dynamics:
            assert cfg.control_mode == 'ctbr', "Simple dynamics only supports CTBR control"
        
        self.cur_cmd = torch.zeros((self.N, 4)).to(self.device)
        self.wind_vec = torch.zeros((self.N, 3)).to(self.device)
        self.wind_l1ac = L1AC_batch(self.N, self.wind_config.l1ac, self.device)
        
        self.load_params(cfg)

        self.state = QuadrotorState(num_instances).to(self.device)

        self.setup_intrinsics()
        
        print(f"[Quadrotor] simple dynamics: {self.use_simple_dynamics}")
        print(f"[Quadrotor] {self}")
        print(f"[Wind] {self.wind_config}")

    def reset(self, 
              idx: torch.Tensor = None, 
              state_randomization: Dict[str, D.Distribution] = None,
              dynamics_randomization: Dict[str, D.Distribution] = None,
              state_vec: torch.Tensor = None):
        """
        Reset the state of the quadrotor.
        :param idx: indices of the quadrotors to reset, default to all
        :param state_randomization: randomization of the initial state
        :param dynamics_randomization: randomization of the intrinsic parameters
        """
        if idx is None:
            idx = torch.arange(self.N)
            
        if self.wind_config.get('enable', False):
            if self.wind_config.get('randomize', False):
                min_vec = torch.tensor(list(self.wind_config.get(
                    'min', [-2., -2., -2.]))).to(self.device)
                max_vec = torch.tensor(list(self.wind_config.get(
                    'max', [2., 2., 2.]))).to(self.device)
                # sample wind vector
                vec_dist = D.Uniform(min_vec, max_vec)
                self.wind_vec[idx] = vec_dist.sample((len(idx),)).float().to(self.device)
            else:
                default_vec = list(self.wind_config.get('default', [0., 0., 0.]))
                self.wind_vec[idx] = torch.stack([
                    default_vec[0] * torch.ones_like(idx),
                    default_vec[1] * torch.ones_like(idx),
                    default_vec[2] * torch.ones_like(idx)], dim=-1).to(self.device)
                
            self.wind_l1ac.reset_adaptation(idx)
            
        self.cur_cmd[idx] = torch.zeros((len(idx), 4)).to(self.device)
        self.cur_cmd[..., 0] = self.g # set the collective thrust to hover

        self.state.reset(idx, state_randomization, state_vec=state_vec)
        self.randomize_dynamics(idx, dynamics_randomization)

        # set the initial motor angular velocities for hovering
        self.state.motor_angvel[idx] = self.hover_motor_angvel[idx]

    def set_state(self, attr, value):
        self.state.set(attr, value)
    
    def get_obs_intrinsics(self, keys: List[str] = None):
        if keys is None:
            keys = self.intrinsics
        
        privileged_obs = [getattr(self, item).flatten(1) for item in keys]
        
        if len(privileged_obs) == 0:
            return None
        else:
            return torch.cat(privileged_obs, dim=-1)
        
    def get_obs_extrinsics(self, 
                           state: QuadrotorState, 
                           cmd: torch.Tensor):
        vel_t = state.vel
        if self.control_mode == 'ctbr':
            thrust_t = ru.rotate_axis(2, state.quat, mode='quat') * cmd[..., 0:1]
        else:
            thrust_t = ru.rotate_axis(2, state.quat, mode='quat') * cmd.sum(dim=-1, keepdim=True)
        self.l1ac_wind_vec = self.wind_l1ac.adaptation_fn(
            vel_t, thrust_t, dt=self.control_dt, mass=self.mass)
        
        if not self.wind_config.get('enable', False):
            return torch.zeros_like(self.wind_vec)
        elif self.wind_config.get('use_l1ac', False):
            return self.l1ac_wind_vec
        else:
            return self.wind_vec
    
    def clear_grad(self):
        self.state = self.state.clone()
        self.wind_l1ac.clear_grad()
        all_attrs = [attr for attr in dir(self) if isinstance(getattr(self, attr), torch.Tensor)]
        for attr in all_attrs:
            setattr(self, attr, getattr(self, attr).clone())

    def step(self, 
             cmd: torch.Tensor, 
             dt: float = None, 
             state: QuadrotorState = None, 
             update: bool = True) -> QuadrotorState:
        """
        Step the dynamics forward by dt seconds.
        :param cmd: command to the quadrotor, shape (N, 4), where N is the number of quadrotors
        :param dt: time step, default to self.sim_dt
        :param state: current state of quadrotors, default to self.state
        :param update: whether to update the state of quadrotors, default to True
        """
        if dt is None:
            dt = self.control_dt

        if state is None:
            state = self.state

        max_dt = self.sim_dt
        remain_ctl_dt = dt
        total_dt = 0.
        
        while remain_ctl_dt > 0:
            sim_dt = min(remain_ctl_dt, max_dt)
            if self.use_simple_dynamics:
                state = self._simple_sim_step(cmd, sim_dt, state)
            else:
                state = self._sim_step(cmd, sim_dt, state)
            remain_ctl_dt -= sim_dt
            total_dt += sim_dt

        if update:
            self.state = state

        return state, total_dt
    
    ###################### Rigid-Body Dynamics ######################
    def _simple_sim_step(self,
                         cmd: torch.Tensor,
                         dt: float = None,
                         state: QuadrotorState = None) -> Tuple[QuadrotorState, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Step the dynamics forward by dt seconds.
        """
        if dt is None:
            dt = self.sim_dt

        if state is None:
            state = self.state
            
        assert cmd.shape[0] == len(state), "cmd and state must have the same batch size"
        assert cmd.shape[1] == 4, "cmd must have 4 elements"
        
        g_vec = torch.tensor([0., 0., -self.g]).to(self.device)
        
        self.cur_cmd = self.cur_cmd + self.motor_tc * (cmd - self.cur_cmd)
        
        cmd_acc_z, cmd_ang_vel = self.cur_cmd.split([1, 3], dim=-1)
        
        prop_acc = cmd_acc_z * torch.tensor([0., 0., 1.]).to(self.device)
        acc = ru.rotate_vector(prop_acc, state.quat, 'quat') + g_vec # in world frame
        
        if self.wind_config.get('enable', False) and self.wind_config.get('random_walk', False):
            self.wind_vec = self.wind_vec + torch.randn(self.wind_vec.shape).to(self.device) * dt
            
        acc = acc + self.wind_vec
        
        state.pos = state.pos + integrators.euler_integral(dt, state.vel, acc) # in world frame
        state.vel = state.vel + acc * dt # in world frame
        if self.linear_std is not None and self.linear_std > 0:
            state.vel = state.vel + torch.normal(0., self.linear_std, size=state.vel.shape).to(self.device) # in world frame
        state.acc = acc # in world frame
        
        dang = integrators.euler_integral(dt, cmd_ang_vel, 0) # in body frame
        if self.angular_std is not None and self.angular_std > 0:
            dang = dang + torch.normal(0., self.angular_std, size=dang.shape).to(self.device) # in body frame
        state.quat = integrators.so3_quat_integral(state.quat, dang, dang_in_body=True) # in world frame
        assert torch.isnan(state.quat).any() == False, "NaN in state.quat"
        state.quat = state.quat / torch.norm(state.quat, dim=-1, keepdim=True) # in world frame
        state.ang_vel = cmd_ang_vel # in body frame
        
        return state
    ################################################################

    ###################### Quadrotor Dynamics ######################
    def _sim_step(self, 
                  cmd: torch.Tensor, 
                  dt: float = None, 
                  state: QuadrotorState = None) -> Tuple[QuadrotorState, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Step the dynamics forward by dt seconds.
        """
        if dt is None:
            dt = self.sim_dt

        if state is None:
            state = self.state

        assert cmd.shape[0] == len(state), "cmd and state must have the same batch size"
        assert cmd.shape[1] == 4, "cmd must have 4 elements"

        if self.control_mode == "ctbr":
            # Collective (mass-normalized) Thrust & BodyRate control, in [m/s^2]+[rad/s]*3
            desired_motor_thrusts = self.CTBR_control(cmd, state)
        elif self.control_mode == 'srt':
            # Single-Rotor (mass-normalized) Thrusts, in [m/s^2]*4
            desired_motor_thrusts = cmd * self.mass
        else:
            raise NotImplementedError('Unknown control mode: {}'.format(
                self.control_mode))
            
        desired_motor_thrusts = self.clamp_motor_thrusts(desired_motor_thrusts)
        motor_thrusts, motor_angvel, thrust_and_torques = \
            self.run_motors(desired_motor_thrusts, dt, state)

        # update state
        state.motor_angvel = motor_angvel
        next_state = self.update_sim(thrust_and_torques, dt, state)

        return next_state

    def CTBR_control(self, cmd: torch.Tensor, state: QuadrotorState = None) -> torch.Tensor:
        """
        Computes the desired thrust and torques based on CTBR (Collective Thrust & BodyRate) commands, all of which are in body frame.
        This implementation is based on the control strategy outlined in the paper:
        "Thrust Mixing, Saturation, and Body-Rate Control for Accurate Aggressive Quadrotor Flight" by Faessler et al.
        """
        if state is None:
            state = self.state
        
        cmd_acc_z, cmd_ang_vel = cmd.split([1, 3], dim=-1)
        
        target_thrust = cmd_acc_z * self.mass
        target_thrust = torch.clamp(target_thrust, 
                                    torch.zeros_like(target_thrust), 
                                    self.max_collective_thrust)
        
        rate_error = cmd_ang_vel - state.ang_vel
        target_torques = mu.bmv(self.inertia_matrix, self.Kp) * rate_error + \
            state.ang_vel.cross(mu.bmv(self.inertia_matrix, state.ang_vel), dim=-1)
        
        thrust_and_torques = torch.cat([target_thrust, target_torques], dim=-1) # (N, 4)
        
        return self.inv_mix(thrust_and_torques) # (N, 4)
    
    def run_motors(self, 
                   desired_motor_thrusts: torch.Tensor, 
                   dt: float = None, 
                   state: QuadrotorState = None):
        """
        Simulate the motors by converting desired thrusts to motor angular velocities.
        """
        if dt is None:
            dt = self.sim_dt

        if state is None:
            state = self.state

        desired_motor_angvels = self.motor_thrust2angvel(desired_motor_thrusts)
        desired_motor_angvels = self.clamp_motor_angvel(desired_motor_angvels)

        if self.motor_first_order:
            # first-order low-pass filter
            motor_angvel = state.motor_angvel + self.motor_tc * (desired_motor_angvels - state.motor_angvel)
        else:
            motor_angvel = desired_motor_angvels

        motor_angvel = self.clamp_motor_angvel(motor_angvel)
        motor_thrusts = self.motor_angvel2thrust(motor_angvel)
        motor_thrusts = self.clamp_motor_thrusts(motor_thrusts)
        
        thrust_and_torques = self.mix(motor_thrusts) # in body frame

        return motor_thrusts, motor_angvel, thrust_and_torques

    def update_sim(self, 
                   thrust_and_torques: torch.Tensor,
                   dt: float = None, 
                   state: QuadrotorState = None) -> QuadrotorState:
        """
        Compute the next state given motor angular velocities and thrust and torques.
        """
        if dt is None:
            dt = self.sim_dt

        if state is None:
            state = self.state

        body_thrust, body_torque = thrust_and_torques.split([1, 3], dim=-1) # in body frame

        body_force = body_thrust * torch.tensor([0., 0., 1.]).to(self.device) # in body frame
        force = ru.rotate_vector(body_force, state.quat, 'quat') # in world frame
        gravity = torch.tensor([0., 0., -self.g]).to(self.device) * self.mass # in world frame
        total_force = force + gravity #+ drag_force # in world frame
        acc = total_force / self.mass # in world frame
        
        if self.wind_config.get('enable', False) and self.wind_config.get('random_walk', False):
            self.wind_vec = self.wind_vec + torch.randn(self.wind_vec.shape).to(self.device) * dt
        
        acc = acc + self.wind_vec
        
        com_torque = mu.bmv(self.com_thrust_torque, body_force) # in body frame
        total_torque = body_torque - com_torque # in body frame
        
        state.acc = acc
        state.body_torque = total_torque

        if self.integrator == 'rk4':
            next_state = integrators.rk4_step(self._dynamics, state, dt)
        elif self.integrator == 'euler':
            next_state = integrators.euler_step(self._dynamics, state, dt)
        else:
            raise ValueError('Invalid integrator: {}'.format(self.integrator))
        
        next_state.quat = next_state.quat / torch.norm(next_state.quat, dim=-1, keepdim=True)

        return next_state

    def _dynamics(self, state: QuadrotorState) -> QuadrotorState:
        d_state = QuadrotorState(state.N).to(self.device)

        quat_vel = torch.stack([torch.zeros_like(state.ang_vel[:, 0]),
                                state.ang_vel[:, 0],
                                state.ang_vel[:, 1],
                                state.ang_vel[:, 2],
                                ], dim=-1)

        d_state.pos = state.vel # in world frame
        d_state.quat = 0.5 * mu.bmv(ru.quat_right(quat_vel), state.quat)
        d_state.vel = state.acc
        d_state.ang_vel = mu.bmv(self.inv_inertia_matrix,
            state.body_torque - 
            state.ang_vel.cross(mu.bmv(self.inertia_matrix, state.ang_vel), dim=-1))
        
        return d_state
    ############################################################################
        
    def inv_mix(self, thrust_and_torques: torch.Tensor) -> torch.Tensor:
        """
        Convert desired thrust and torque to single-motor thrusts
        """
        return mu.bmv(self.inv_mixer, thrust_and_torques)

    def mix(self, motor_thrusts: torch.Tensor) -> torch.Tensor:
        """
        Mix single-rotor thrusts to body thrust and torques.
        """
        return mu.bmv(self.mixer, motor_thrusts)
    
    def clamp_motor_thrusts(self, motor_thrusts):
        return motor_thrusts.clip(torch.zeros_like(self.motor_max_thrust), 
                                  self.motor_max_thrust)
    
    def clamp_motor_angvel(self, motor_angvel):
        return motor_angvel.clip(torch.zeros_like(self.motor_max_angvel), 
                                 self.motor_max_angvel)
        
    def load_params(self, cfg):
        """
        Load parameters from a config.
        """
        self.name = cfg.get('name', 'you_know_who')
        
        # common parameters
        self.g = cfg.get('g', 9.81)
        self.sim_dt = cfg.get('dt', 2.5e-3)
        self.integrator = cfg.get('integrator', 'rk4')

        # rigid-body parameters
        self._mass = cfg.get('mass', 1.0)
        self._arm_length = cfg.get('arm_length', 0.125)
        self._l2w_ratio = cfg.get('l2w_ratio', 1.0)
        self._l2h_ratio = cfg.get('l2h_ratio', 1.0)
        self._center_of_mass = cfg.get('center_of_mass', [0., 0., 0.])
        self._drag_coeff = cfg.get('drag_coeff', 0.2)

        # motor parameters
        self._motor_rot_dirs = cfg.get('motor_rot_dirs', [1., -1., 1., -1.]) # 1 for CW, -1 for CCW
        self._motor_spread_angles = cfg.get('motor_spread_angles', [0.25*torch.pi, 0.75*torch.pi, 1.25*torch.pi, 1.75*torch.pi])
        self._motor_kf = cfg.get('motor_kf', 1.28192e-08)
        self._motor_kappa = cfg.get('motor_kappa', 0.016)
        self._motor_tc = cfg.get('motor_tc', 0.025)
        self._max_thrust2weight = cfg.get('max_thrust2weight', 2.)
        self.motor_first_order = cfg.get('motor_first_order', False)
        
        # control parameters
        self.control_mode = cfg.get('control_mode', 'ctbr')
        self.control_freq = cfg.get('control_freq', 40)
        self.control_dt = 1. / self.control_freq
        self._Kp = cfg.get('Kp', [1.75, 1.75, 1.75])
        
        if self.is_sysid:
            for item in self.intrinsics:
                setattr(self, '_' + item, torch.as_tensor(
                        getattr(self, '_' + item)).float().to(self.device).requires_grad_())

    def setup_intrinsics(self):
        """
        Update intrinsic parameters of the quadrotor.
        """
        # rigid-body parameters
        self.mass = to_tensor(self._mass).expand(self.N, 1).float().to(self.device)
        self.arm_length = to_tensor(self._arm_length).expand(self.N, 1).float().to(self.device)
        self.l2w_ratio = to_tensor(self._l2w_ratio).expand(self.N, 1).float().to(self.device)
        self.l2h_ratio = to_tensor(self._l2h_ratio).expand(self.N, 1).float().to(self.device)
        self.center_of_mass = to_tensor(self._center_of_mass).expand(self.N, 3).float().to(self.device)
        self.drag_coeff = to_tensor(self._drag_coeff).expand(self.N, 1).float().to(self.device)

        # motor parameters
        self.motor_rot_dirs = to_tensor(self._motor_rot_dirs).expand(self.N, 4).float().to(self.device)
        self.motor_spread_angles = to_tensor(self._motor_spread_angles).expand(self.N, 4).float().to(self.device)
        self.motor_kf = to_tensor(self._motor_kf).expand(self.N, 1).float().to(self.device)
        self.motor_kappa = to_tensor(self._motor_kappa).expand(self.N, 1).float().to(self.device)
        self.motor_tc = to_tensor(self._motor_tc).expand(self.N, 1).float().to(self.device)
        self.max_thrust2weight = to_tensor(self._max_thrust2weight).expand(self.N, 1).float().to(self.device)

        # control parameters
        self.Kp = to_tensor(self._Kp).expand(self.N, 3).float().to(self.device)
        
        self.update_intrinsic_utils()

    def randomize_dynamics(self, 
                           idx: torch.Tensor = None, 
                           randomization: Dict[str, D.Distribution] = None,):
        if idx is None:
            idx = torch.arange(self.N)
        
        if randomization is None:
            randomization = {}
        
        for item in randomization.keys():
            assert item in self.intrinsics, f"Unknown intrinsic: {item}"

            param_dist = randomization[item]
            shape = getattr(self, item)[idx].shape
            getattr(self, item)[idx] = param_dist.sample(
                (len(idx),)).float().to(self.device).reshape(shape)
            
        self.update_intrinsic_utils()

    def update_intrinsic_utils(self):
        """
        Compute the inertia matrix, its inverse, and the mixer matrix, as well as the mapping functions.
        """
        # compute inertia matrix and its inverse
        self.inertia_matrix = torch.diag_embed(
            self.mass / 12. * self.arm_length.square() * torch.cat([
                self.l2w_ratio.square() + self.l2h_ratio.square(),
                self.l2w_ratio.square() + self.l2h_ratio.square(),
                self.l2w_ratio.square() + self.l2w_ratio.square()], dim=-1)
            )
        self.inv_inertia_matrix = torch.inverse(self.inertia_matrix)
        
        # compute com_thrust_torque
        rigidbody_size = torch.cat([
            self.arm_length * self.l2w_ratio,
            self.arm_length * self.l2w_ratio,
            self.arm_length * self.l2h_ratio], dim=-1) # (N, 3)
        self.com_thrust_torque = mu.skew_matrix(self.center_of_mass * rigidbody_size)

        # compute mixer matrix
        self.mixer = torch.stack([
            torch.ones_like(self.motor_spread_angles),
            torch.sin(self.motor_spread_angles) * self.arm_length,
            -1. * torch.cos(self.motor_spread_angles) * self.arm_length,
            self.motor_kappa * self.motor_rot_dirs
        ], dim=-2)
        self.inv_mixer = torch.inverse(self.mixer)
        
        # generate the mapping functions
        self.motor_angvel2thrust = lambda x: self.motor_kf * x ** 2
        self.motor_thrust2angvel = lambda y: torch.sqrt(y / self.motor_kf)

        # compute the limits of the motor
        self.motor_max_angvel = torch.sqrt(self.max_thrust2weight * self.mass * self.g / 4. / self.motor_kf)
        self.motor_max_thrust = self.motor_angvel2thrust(self.motor_max_angvel)
        self.max_collective_thrust = self.motor_max_thrust * 4.

        self.hover_motor_angvel = self.motor_thrust2angvel(self.mass * self.g / 4.)
        self.hover_motor_angvel = self.hover_motor_angvel.expand(self.N, 4).float().to(self.device)

    def parameters(self):
        """
        Return all parameters of the quadrotor.
        """
        return [getattr(self, '_' + item) for item in self.intrinsics]
    
    def named_parameters(self):
        """
        Return all parameters of the quadrotor with names.
        """
        return [(item, getattr(self, '_' + item)) for item in self.intrinsics]
    
    def set_parameters(self, params: Dict[str, torch.Tensor]):
        """
        Set all parameters of the quadrotor.
        """
        for name, param in params.items():
            assert name in self.intrinsics, f"Unknown intrinsic: {name}"
            assert param.shape == getattr(self, '_' + name).shape, f"Shape mismatch: {name}"
            setattr(self, '_' + name, param)
            
        self.setup_intrinsics()
    
    def __repr__(self) -> str:
        """
        Print all tunable intrinsic parameters of the quadrotor.
        """
        print_str = 'Quadrotor Intrinsic Parameters:'
        for item in self.intrinsics:
            print_str += f'\n{item}: {getattr(self, "_" + item)}'
        return print_str

    def export_yaml(self, yaml_path):
        """
        Export the quadrotor parameters to a yaml file. (Inverse of load_params)
        """
        f = open(yaml_path, 'w')
        f.write('name: {}\n'.format(self.name))
        f.write('g: {}\n'.format(self.g))
        f.write('dt: {}\n'.format(self.sim_dt))
        f.write('integrator: {}\n'.format(self.integrator))
        f.write('control_mode: {}\n'.format(self.control_mode))
        f.write('control_freq: {}\n'.format(self.control_freq))
        for item in self.intrinsics:
            value = getattr(self, '_' + item)
            if isinstance(value, torch.Tensor):
                value = value.tolist()
            f.write('{}: {}\n'.format(item, value))
