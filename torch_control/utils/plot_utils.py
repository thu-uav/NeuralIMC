import os

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

def plot_3d_traj(pos, vel, ang, traj_len, save_dir: str = None, return_array: bool = False):
    """
    Plot the trajectory of the drone in 3D.
    """
    N = pos.shape[1]
    
    # plot N trajectories into subplots, forming near a square matrix
    length = int(np.ceil(np.sqrt(N)))
    fig, axs = plt.subplots(length, length, figsize=(2*length, 2*length), subplot_kw={'projection': '3d'})
    axs = axs.flatten()
    for i in range(N):
        ax = axs[i]
        max_t = int(traj_len[i])
        ax.plot(pos[:max_t, i, 0], pos[:max_t, i, 1], pos[:max_t, i, 2])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title('Trajectory of drone {}'.format(i))
        
    returns = []
    fig.tight_layout()
        
    if return_array:
        returns.append(fig_to_array(fig))
    else:
        if save_dir is not None:
            fig.savefig(os.path.join(save_dir, 'traj.png'))
    
    plt.close(fig)
        
    # plot pos/vel/ang in N*3 subplots
    fig, axs = plt.subplots(3, N, figsize=(2*N, 6))
    for i in range(N):
        axs[0, i].plot(pos[:, i, 0])
        axs[0, i].plot(pos[:, i, 1])
        axs[0, i].plot(pos[:, i, 2])
        axs[0, i].set_title('Position of drone {}'.format(i))
        axs[0, i].legend(['x', 'y', 'z'])
        axs[1, i].plot(vel[:, i, 0])
        axs[1, i].plot(vel[:, i, 1])
        axs[1, i].plot(vel[:, i, 2])
        axs[1, i].set_title('Velocity of drone {}'.format(i))
        axs[1, i].legend(['x', 'y', 'z'])
        axs[2, i].plot(ang[:, i, 0])
        axs[2, i].plot(ang[:, i, 1])
        axs[2, i].plot(ang[:, i, 2])
        axs[2, i].set_title('Angle of drone {}'.format(i))
        axs[2, i].legend(['x', 'y', 'z'])
        
    fig.tight_layout()
    
    if return_array:
        returns.append(fig_to_array(fig))
    else:
        if save_dir is not None:
            fig.savefig(os.path.join(save_dir, 'pos_vel_ang.png'))
    
    plt.close(fig)
        
    return returns
    
def fig_to_array(fig):
    """
    Convert a matplotlib figure to a numpy array.
    """
    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    ncols, nrows = fig.canvas.get_width_height()
    return np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 3)