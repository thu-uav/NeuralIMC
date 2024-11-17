import os
import argparse
import math
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
from torch_control.utils import rot_utils as ru

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_name', type=str, required=True)
    parser.add_argument('--is_cf', action='store_true', default=False)
    return parser.parse_args()

def rad2deg(ang):
    return ang * 180. / np.pi

def quat2euler(qw, qx, qy, qz):
    # Convert quaternion to Euler angles
    # roll
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # pitch
    sinp = 2.0 * (qw * qy - qz * qx)
    if (abs(sinp) >= 1.0):
        pitch = math.copysign(math.pi() / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    # yaw
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz *qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return pd.Series([rad2deg(roll), rad2deg(pitch), rad2deg(yaw)])

def wrap_pi(input_angle):
    # Normalize angle to the range [-180, 180]
    while input_angle >= 180:
        input_angle = input_angle - 360
    while input_angle < -180:
        input_angle = input_angle + 360
    return input_angle

def estimate_omega(att, time):
    # Estimate angular velocity from attitude
    # att: N, 4, np.ndarray
    # time: N, np.ndarray
    dt = np.diff(time)
    euler = ru.np_quat2euler(att)
    d_euler = np.diff(euler, axis=0)
    omg = d_euler / dt[..., None]
    return rad2deg(omg)

def estimate_acc(att, time, vel, extra_acc):
    g_vec = np.array([0, 0, -9.81])
    vel_b = ru.np_inv_rotate_vector(vel, att, mode='quat')
    acc_b = np.diff(vel_b, axis=0) / np.diff(time)[:, None]
    prop_acc = acc_b - g_vec
    if extra_acc is not None:
        prop_acc -= extra_acc[1:]
        
    return prop_acc
        
def load_and_prepare_data(log_name):
    data_path = os.path.join(os.path.dirname(__file__), log_name)
    data = np.load(data_path)
    
    extra_data = {}

    # Process data into DataFrames and return
    time = pd.DataFrame(data['time'], columns=['time']) # N, 1
    dt = time.diff().fillna(0)[1:] # N-1, 1
    pos_ref = pd.DataFrame(data['pos_ref'], columns=['pos_ref_x', 'pos_ref_y', 'pos_ref_z']) # N, 3
    pos_est = pd.DataFrame(data['pos_est'], columns=['pos_est_x', 'pos_est_y', 'pos_est_z']) # N, 3
    vel_ref = pd.DataFrame(data['vel_ref'], columns=['vel_ref_x', 'vel_ref_y', 'vel_ref_z']) # N, 3
    vel_est = pd.DataFrame(data['vel_est'], columns=['vel_est_x', 'vel_est_y', 'vel_est_z']) # N, 4
    att_ref = pd.DataFrame(data['att_ref'], columns=['att_ref_w', 'att_ref_x', 'att_ref_y', 'att_ref_z']) # N, 4
    euler_ref = att_ref.apply(lambda x: quat2euler(x[0], x[1], x[2], x[3]), axis=1) # N, 3
    euler_ref.columns = ['att_ref_x', 'att_ref_y', 'att_ref_z']
    att_est = pd.DataFrame(data['att_est'], columns=['att_ref_w', 'att_est_x', 'att_est_y', 'att_est_z']) # N, 4
    euler_est = att_est.apply(lambda x: quat2euler(x[0], x[1], x[2], x[3]), axis=1) # N, 3
    euler_est.columns = ['att_est_x', 'att_est_y', 'att_est_z']
    omg_ref = pd.DataFrame(data['omg_ref'], columns=['omg_ref_x', 'omg_ref_y', 'omg_ref_z']) # N, 3
    omg_ref = omg_ref.apply(lambda x: rad2deg(x))
    omg_est = pd.DataFrame(data['omg_est'], columns=['omg_est_x', 'omg_est_y', 'omg_est_z']) # N, 3
    omg_est = omg_est.apply(lambda x: rad2deg(x))
    thr_ref = pd.DataFrame(data['thr_ref'], columns=['thr_ref']) # N, 1
    if 'thr_est' in data:
        thr_est = pd.DataFrame(data['thr_est'], columns=['thr_est']) # N, 1
    else:
        thr_est = pd.DataFrame(np.zeros((len(data['time']), 1)), columns=['thr_est']) # N, 1
    
    # acc_ref_np = np.zeros((len(data['time']), 3))
    # acc_ref_np[..., 2] = data['thr_ref'][:, 0]
    # acc_ref = pd.DataFrame(acc_ref_np, columns=['acc_ref_x', 'acc_ref_y', 'acc_ref_z']) # N, 3
    
    # omg_compute = pd.DataFrame(estimate_omega(data['att_est'], data['time']), 
    #                            columns=['omg_compute_x', 'omg_compute_y', 'omg_compute_z'])
    # acc_est = pd.DataFrame(estimate_acc(data['att_est'], data['time'], data['vel_est'], data.get('l1ac', None)), 
    #                        columns=['acc_est_x', 'acc_est_y', 'acc_est_z'])
    
    columns = [pos_ref, pos_est, vel_ref, vel_est, euler_ref, euler_est, 
               omg_ref, omg_est, thr_ref, thr_est] #, omg_compute, acc_ref, acc_est, thr_ref]
    
    if 'l1ac' in data:
        l1ac_est = pd.DataFrame(data['l1ac'][1:], columns=['l1ac_est_x', 'l1ac_est_y', 'l1ac_est_z']) # N, 3
        columns.append(l1ac_est)
        
    if 'latent' in data:
        extra_data['latent'] = data['latent']
    
    if 'obs' in data:
        extra_data['obs'] = data['obs']
    
    df_log = pd.concat(columns, axis=1)
    df_log = df_log.iloc[:len(data['time']), :]
    df_log.index = data['time'] - data['time'][0]
    df_log.index.name = 'time'

    return df_log, extra_data

def plot_error_statistics(df_log, out_path, plot_type, title, yaxis_titles, 
                          flip_flag=False,
                          colors=('rgba(80,26,80,0.8)', 'rgba(26,80,80,0.8)', 'rgba(80,80,26,0.8)')):
    """
    A unified function to plot error statistics for position, velocity, or any other types.

    :param df_log: The DataFrame containing the log data.
    :param out_path: Output path for the HTML plot file.
    :param plot_type: A string indicating the type of plot ('pos', 'vel', 'att', 'omg').
    :param title: The title of the plot.
    :param yaxis_titles: A tuple of titles for the Y-axes.
    :param colors: A tuple of colors for the reference and estimated data plots.
    """
    fig = make_subplots(rows=3, cols=1, x_title='time [s]')
    ref_prefix = f"{plot_type}_ref_"
    est_prefix = f"{plot_type}_est_"
    compute_prefix = f"{plot_type}_compute_"

    for i, axis in enumerate(['x', 'y', 'z'], start=1):
        if flip_flag and axis in ['y', 'z'] and plot_type == 'omg':
            flag = -1
        else:
            flag = 1
        fig.add_trace(go.Scatter(
            x=df_log.index,
            y=df_log[f"{ref_prefix}{axis}"] * flag,
            mode="lines+markers",
            name=f"ref {plot_type} {axis}",
            marker=dict(color=colors[0]),
            text=df_log[f"{ref_prefix}{axis}"]
        ), row=i, col=1)

        fig.add_trace(go.Scatter(
            x=df_log.index,
            y=df_log[f'{est_prefix}{axis}'],
            mode="lines+markers",
            name=f"est {plot_type} {axis}",
            marker=dict(color=colors[1]),
            text=df_log[f"{est_prefix}{axis}"]
        ), row=i, col=1)
        
        # if f"{compute_prefix}{axis}" in df_log.columns:
        #     fig.add_trace(go.Scatter(
        #         x=df_log.index,
        #         y=df_log[f'{compute_prefix}{axis}'],
        #         mode="lines+markers",
        #         name=f"compute {plot_type} {axis}",
        #         marker=dict(color=colors[2]),
        #         text=df_log[f"{compute_prefix}{axis}"]
        #     ), row=i, col=1)

        fig['layout'][f'yaxis{i}'].update({'title': yaxis_titles[i-1]})

    fig.update_layout(height=1000, title_text=title)
    plot_file_name = f"/{plot_type.capitalize()}_Error_Statistics"
    fig.write_html(out_path + plot_file_name + ".html")
    fig.update_layout(width=1500, height=1000, autosize=True, title_text=title)
    pio.write_image(fig, out_path + plot_file_name + ".png")

    # Error Box Plot
    fig = go.Figure()
    errors = []
    for axis in ['x', 'y', 'z']:
        if flip_flag and axis in ['y', 'z'] and plot_type == 'omg':
            flag = -1
        else:
            flag = 1
        error = df_log[f"{ref_prefix}{axis}"] * flag - df_log[f"{est_prefix}{axis}"]
        if plot_type == 'att':
            error = error.apply(wrap_pi)
        errors.append(error)
        fig.add_trace(go.Box(y=error, name=f"{axis.upper()}"))
    err_norm = np.linalg.norm(np.stack(errors, axis=-1), axis=-1)

    title_text = f"""{plot_type.capitalize()} Error Box Graph
        x error[{np.around(np.mean(errors[0]), 4)}±{np.around(np.std(errors[0]), 4)}]
        y error[{np.around(np.mean(errors[1]), 4)}±{np.around(np.std(errors[1]), 4)}]
        z error[{np.around(np.mean(errors[2]), 4)}±{np.around(np.std(errors[2]), 4)}]
        norm[{np.around(np.mean(err_norm), 4)}±{np.around(np.std(err_norm), 4)}]"""

    if plot_type == 'pos':
        yaxis_title_text = 'Position [m]'
    elif plot_type == 'vel':
        yaxis_title_text = 'Velocity [m/s]'
    elif plot_type == 'att':
        yaxis_title_text = 'Attitude [deg]'
    else:
        yaxis_title_text = 'Angular Velocity [deg/s]'
    fig.update_layout(title_text=title_text, yaxis_title_text=yaxis_title_text,)
    plot_file_name_box = f"/{plot_type.capitalize()}_Error_Box_Graph"
    fig.write_html(out_path + plot_file_name_box + ".html")
    fig.update_layout(width=1500, height=500, autosize=True, title_text=title_text, yaxis_title_text=yaxis_title_text,)
    pio.write_image(fig, out_path + plot_file_name_box + ".png")

def plot_thrust(df_log, out_path):
    fig = make_subplots(rows=1, cols=1, x_title='time [s]')
    fig.append_trace(go.Scatter(x = df_log.index,
                                y = df_log.thr_ref,
                                mode = "lines+markers",
                                name = "thrust ref",
                                marker = dict(color='rgba(80,26,80,0.8)')
                                ), row=1, col=1)
    fig.append_trace(go.Scatter(x = df_log.index,
                                y = df_log.thr_est,
                                mode = "lines+markers",
                                name = "thrust est",
                                marker = dict(color='rgba(26,80,80,0.8)')
                                ), row=1, col=1)
    fig.update_layout(height=1000, title_text="Thrust Command")
    fig['layout']['yaxis'].update({'title': 'Thrust'})
    fig.write_html(out_path + "/Thrust.html")
    fig.update_layout(width=1500, height=500, autosize=True, title_text="Thrust Command")
    pio.write_image(fig, out_path + "/Thrust.png")
    
def plot_acc(df_log, out_path, plot_type, title, yaxis_titles, 
             colors=('rgba(80,26,80,0.8)', 'rgba(26,80,80,0.8)', 'rgba(80,80,26,0.8)')):
    fig = make_subplots(rows=3, cols=1, x_title='time [s]')
    ref_prefix = "acc_ref_"
    est_prefix = "acc_est_"
    l1ac_prefix = "l1ac_est_"
    plot_type = "acc"

    for i, axis in enumerate(['x', 'y', 'z'], start=1):
        fig.add_trace(go.Scatter(
            x=df_log.index,
            y=df_log[f"{ref_prefix}{axis}"],
            mode="lines+markers",
            name=f"ref {plot_type} {axis}",
            marker=dict(color=colors[0]),
            text=df_log[f"{ref_prefix}{axis}"]
        ), row=i, col=1)

        fig.add_trace(go.Scatter(
            x=df_log.index,
            y=df_log[f'{est_prefix}{axis}'],
            mode="lines+markers",
            name=f"est {plot_type} {axis}",
            marker=dict(color=colors[1]),
            text=df_log[f"{est_prefix}{axis}"]
        ), row=i, col=1)
        
        # if f"{l1ac_prefix}{axis}" in df_log.columns:
        #     fig.add_trace(go.Scatter(
        #         x=df_log.index,
        #         y=df_log[f'{l1ac_prefix}{axis}'],
        #         mode="lines+markers",
        #         name=f"l1ac {plot_type} {axis}",
        #         marker=dict(color=colors[2]),
        #         text=df_log[f"{l1ac_prefix}{axis}"]
        #     ), row=i, col=1)

        fig['layout'][f'yaxis{i}'].update({'title': yaxis_titles[i-1]})

    fig.update_layout(height=1000, title_text=title)
    plot_file_name = f"/{plot_type.capitalize()}_Error_Statistics"
    fig.write_html(out_path + plot_file_name + ".html")
    fig.update_layout(width=1500, height=1000, autosize=True, title_text=title)
    pio.write_image(fig, out_path + plot_file_name + ".png")

def plot_trajectory(df_log, out_path):
    fig = go.Figure()
    trace0 = go.Scatter3d(x=df_log.pos_ref_x,
                        y=df_log.pos_ref_y,
                        z=df_log.pos_ref_z,
                        mode='lines',
                        name='command',
                        marker=dict(color='rgba(80,26,80,0.8)'))
    trace1 = go.Scatter3d(x = df_log.pos_est_x,
                        y = df_log.pos_est_y,
                        z = df_log.pos_est_z,
                        mode = "lines",
                        name = "estimate",
                        marker = dict(color='rgba(26,80,80,0.8)'))
    fig.add_trace(trace0)
    fig.add_trace(trace1)
    fig.update_layout(
        title='Trajectory',
        scene=dict(
            xaxis_title='X[m]',
            xaxis_dtick=0.1,
            yaxis_title='Y[m]',
            yaxis_dtick=0.1,
            zaxis_title='Z[m]',
            zaxis_dtick=0.1,
            aspectratio=dict(x=1, y=1, z=0.5),
        )
    )
    fig.write_html(out_path + "/Trajectory.html")
    fig.update_layout(width=1500, height=1500, autosize=True)
    pio.write_image(fig, out_path + "/Trajectory.png")

def main():
    args = parse_arguments()
    log_data, extra_data = load_and_prepare_data(args.log_name)
    
    out_path = os.path.join(os.path.dirname(__file__), args.log_name.split('.npz')[0])
    os.makedirs(out_path, exist_ok=True)

    plot_error_statistics(log_data, out_path, 
                          'pos', 'Position Error Statistics', 
                          ('Forward [m]', 'Left [m]', 'Up [m]'))
    plot_error_statistics(log_data, out_path, 
                          'vel', 'Velocity Error Statistics', 
                          ('Forward [m/s]', 'Left [m/s]', 'Up [m/s]'))
    plot_error_statistics(log_data, out_path, 
                          'att', 'Attitude Error Statistics', 
                          ('Roll [deg]', 'Pitch [deg]', 'Yaw [deg]'))
    plot_error_statistics(log_data, out_path,
                          'omg', 'Angular Velocity Error Statistics', 
                          ('Roll [rad/s]', 'Pitch [rad/s]', 'Yaw [deg/s]'), flip_flag=args.is_cf)
    plot_thrust(log_data, out_path)
    # plot_acc(log_data, out_path,
    #         'acc', 'Acceleration Error Statistics', 
    #         ('Forward [m/s2]', 'Left [m/s2]', 'Up [m/s2]'))
    plot_trajectory(log_data, out_path)
    
    for name, value in extra_data.items():
        # plot array data
        plt.figure(figsize=(20, 20))
        plt.imshow(value, cmap='viridis', aspect='auto')
        plt.savefig(out_path + f'/{name}.png')

if __name__ == "__main__":
    main()
