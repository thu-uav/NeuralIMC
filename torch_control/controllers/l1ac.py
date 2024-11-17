import numpy as np
import scipy
import torch

from scipy import linalg
 
"""
Note:
Although somehow confusing, `thrust_t` here represents the acceleration applied by the propellers, 
i.e., it's the mass-normalized thrust in the world frame.
And finally, we want the estimated disturbance `d_hat` to be in the world frame, also mass-normalized.
"""

class L1AC:
    def __init__(self, l1ac_cfg, device: str):
        self.device = device
        self.g = 9.81
        
        self.pseudo_adapt = l1ac_cfg.get('pseudo_adapt', False)
        
        self.mass = l1ac_cfg.get('mass', 0.03)
        self.A_vx = l1ac_cfg.get('A_vx', -6)
        self.A_vy = l1ac_cfg.get('A_vy', -6)
        self.A_vz = l1ac_cfg.get('A_vz', -10)
        self.wc_f = l1ac_cfg.get('wc_f', 0.4)
        
        self.As = np.diag([self.A_vx, self.A_vy, self.A_vz]) # (3, 3)
        
        self.vel_hat = None
        self.d_hat = None
        
        if self.pseudo_adapt:
            self.adaptation_fn = self._pseudo_adaptation
        else:
            self.adaptation_fn = self._l1_adaptation

    def clear_grad(self):
        if self.vel_hat is not None:
            self.vel_hat = self.vel_hat.clone()
        if self.d_hat is not None:
            self.d_hat = self.d_hat.clone()
            
    def _pseudo_adaptation(self, vel_t, thrust_t, dt):
        return torch.zeros_like(thrust_t)
    
    def _l1_adaptation(self, vel_t, thrust_t, dt = 0.02):
        if self.vel_hat is not None:
            mu_s = scipy.linalg.expm(self.As * dt)
            m_phi_inv_mu_s = torch.tensor(-1. * np.matmul(np.linalg.inv(np.linalg.inv(self.As) * (mu_s - np.identity(3))), mu_s), dtype=torch.float32, device=self.device)
            
            As = torch.tensor(self.As, dtype=torch.float32, device=self.device).unsqueeze(0).expand(thrust_t.shape[:-1] + (3, 3))
            m_phi_inv_mu = m_phi_inv_mu_s.unsqueeze(0).expand(thrust_t.shape[:-1] + (3, 3))
            
            z_tilde = self.vel_hat - vel_t # (, 3)
            d_hat = torch.bmm(m_phi_inv_mu, z_tilde.unsqueeze(-1)).squeeze(-1)
            
            # low-pass filter
            self.d_hat = (dt * self.wc_f) * d_hat + (1. - dt * self.wc_f) * self.d_hat
            
            g_vec = torch.tensor([0, 0, -self.g], dtype=torch.float32, device=self.device)
            g_vec = g_vec.unsqueeze(0).expand_as(thrust_t)
            
            dv_hat = g_vec + thrust_t
            
            d_vel = dv_hat + self.d_hat / self.mass + torch.bmm(As, z_tilde.unsqueeze(-1)).squeeze(-1)
            
            self.vel_hat = self.vel_hat + d_vel * dt
        else:
            self.vel_hat = vel_t
            self.d_hat = torch.zeros_like(thrust_t)
        
        return self.d_hat / self.mass
    
class L1AC_batch:
    def __init__(self, num_envs, l1ac_cfg, device: str):
        self.device = device
        self.g = 9.81
        self.N = num_envs
        
        self.pseudo_adapt = l1ac_cfg.get('pseudo_adapt', False)
        
        self.mass = l1ac_cfg.get('mass', 0.03)
        self.A_vx = l1ac_cfg.get('A_vx', -6)
        self.A_vy = l1ac_cfg.get('A_vy', -6)
        self.A_vz = l1ac_cfg.get('A_vz', -10)
        self.wc_f = l1ac_cfg.get('wc_f', 0.4)
        
        self.As = torch.from_numpy(np.diag([self.A_vx, self.A_vy, self.A_vz])
                                   ).float().to(self.device).unsqueeze(0).repeat(self.N, 1, 1)
        
        self.vel_hat = torch.zeros((self.N, 3)).float().to(self.device)
        self.d_hat = torch.zeros((self.N, 3)).float().to(self.device)
        self.d_hat_t = torch.zeros((self.N, 3)).float().to(self.device)
        self.init_label = torch.empty(self.N).fill_(False).to(self.device)
        
        if self.pseudo_adapt:
            self.adaptation_fn = self._pseudo_adaptation
        else:
            self.adaptation_fn = self._l1_adaptation

    def clear_grad(self):
        self.As = self.As.clone()
        self.vel_hat = self.vel_hat.clone()
        self.d_hat = self.d_hat.clone()
        self.d_hat_t = self.d_hat_t.clone()
        self.init_label = self.init_label.clone()
            
    def reset_adaptation(self, idx: torch.Tensor = None):
        if idx is None:
            idx = torch.arange(self.N, dtype=torch.long, device=self.device)
            
        self.vel_hat[idx] = 0.
        self.d_hat[idx] = 0.
        self.d_hat_t[idx] = 0.
        self.init_label[idx] = False
            
    def _pseudo_adaptation(self, vel_t, thrust_t, dt):
        return torch.zeros_like(thrust_t)
    
    def _l1_adaptation(self, vel_t, thrust_t, dt = 0.02, mass = None):
        if mass is None:
            mass = self.mass
        
        try:
            mu_s = torch.linalg.matrix_exp(self.As * dt) # (N, 3, 3)
        except:
            mu_s = torch.from_numpy(linalg.expm(self.As.cpu().numpy()[0] * dt)
                    ).to(self.As.device).unsqueeze(0)
        m_phi_inv_mu_s = -1. * torch.bmm(torch.inverse(
            torch.inverse(self.As) * (mu_s - torch.eye(3).to(mu_s.device))), mu_s) # (N, 3, 3)
        
        z_tilde = self.vel_hat - vel_t # (N, 3)
        d_hat = torch.bmm(m_phi_inv_mu_s, z_tilde.unsqueeze(-1)).squeeze(-1) # (N, 3)
        
        # low-pass filter
        self.d_hat = (dt * self.wc_f) * d_hat + (1. - dt * self.wc_f) * self.d_hat
        
        g_vec = torch.tensor([0, 0, -self.g], dtype=torch.float32, device=self.device)
        
        dv_hat = g_vec + thrust_t
        d_vel = dv_hat + self.d_hat / mass + torch.bmm(self.As, z_tilde.unsqueeze(-1)).squeeze(-1)
        
        self.vel_hat = self.vel_hat + d_vel * dt
        
        self.vel_hat = vel_t * (1. - self.init_label.unsqueeze(-1).float()) \
                  + self.vel_hat * self.init_label.unsqueeze(-1).float()
        d_hat = self.d_hat * self.init_label.unsqueeze(-1).float()
        
        self.init_label[:] = True
        
        return self.d_hat / mass
    
    
if __name__ == "__main__":
    # a unit test to figure out the difference between the two implementations
    # _As = [-6, -6, -10]
    # dt = 0.02
    # alpha = 0.99
    # mass = 1.0

    # # vector version
    # v_As = np.array(_As)
    # v_phi = 1 / v_As  * (np.exp(v_As * dt) - 1)
    # v_scale = - 1 / v_phi * np.exp(v_As * dt)

    # # matrix version
    # m_As = np.diag(_As)
    # m_phi = np.linalg.inv(m_As) * (scipy.linalg.expm(m_As * dt) - np.identity(3))
    # m_scale = -1 * np.linalg.inv(m_phi) * scipy.linalg.expm(m_As * dt)

    # print("phi:\n", v_As, "\n", m_phi, "\n", np.abs(np.diag(v_phi) - m_phi))
    # print("sclae:\n", v_scale, "\n", m_scale, "\n", np.abs(np.diag(v_scale) - m_scale))

    # print(torch.from_numpy(m_As).unsqueeze(0).diagflat().shape)
    
    l1ac_fast = L1AC_batch(1, {}, 'cpu')
    l1ac_norm = L1AC_batch(1, {}, 'cpu')
    l1ac_old = L1AC_batch(1, {}, 'cpu')
    
    l1ac_fast.reset_adaptation()
    l1ac_norm.reset_adaptation()
    l1ac_old.reset_adaptation()
    
    for i in range(100):
    
        vel_t = torch.rand(1, 3)
        thrust_t = torch.rand(1, 3)
        
        out1 = l1ac_fast._l1_adaptation_fast(vel_t, thrust_t, 0.02, 1.0)
        out2 = l1ac_norm._l1_adaptation(vel_t, thrust_t, 0.02, 1.0)
        out3 = l1ac_old._l1_adaptation_old(vel_t, thrust_t, 1.0, 0.02)
        
        print(out1)
        print(out2)
        print(out3)
        print('------')
