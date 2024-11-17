import os
import argparse
import math
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots

parser = argparse.ArgumentParser()
parser.add_argument('--log_name', type=str)
args = parser.parse_args()
LOG_NAME = args.log_name

def quat2Euler(qw, qx, qy, qz):
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

    return pd.Series([roll, pitch, yaw])

def wrapPi(input):
    while input >= np.pi:
        input = input - 2 * np.pi
    while input < -np.pi:
        input = input + 2 * np.pi
    return input

out_path = os.path.join(os.path.dirname(__file__), '../data', LOG_NAME.split('.')[0])
os.makedirs(out_path, exist_ok=True)

data = np.load(os.path.join(os.path.dirname(__file__), 'data', LOG_NAME))
pos_ref = pd.DataFrame(data['pos_ref'], columns=['pos_ref_x', 'pos_ref_y', 'pos_ref_z'])
pos_est = pd.DataFrame(data['pos_est'], columns=['pos_est_x', 'pos_est_y', 'pos_est_z'])
vel_ref = pd.DataFrame(data['vel_ref'], columns=['vel_ref_x', 'vel_ref_y', 'vel_ref_z'])
vel_est = pd.DataFrame(data['vel_est'], columns=['vel_est_x', 'vel_est_y', 'vel_est_z'])
att_ref = pd.DataFrame(data['att_ref'], columns=['att_ref_w', 'att_ref_x', 'att_ref_y', 'att_ref_z'])
euler_ref = att_ref.apply(lambda x: quat2Euler(x[0], x[1], x[2], x[3]), axis=1)
euler_ref.columns = ['att_ref_x', 'att_ref_y', 'att_ref_z']
att_est = pd.DataFrame(data['att_est'], columns=['att_ref_w', 'att_est_x', 'att_est_y', 'att_est_z'])
euler_est = att_est.apply(lambda x: quat2Euler(x[0], x[1], x[2], x[3]), axis=1)
euler_est.columns = ['att_est_x', 'att_est_y', 'att_est_z']
omg_ref = pd.DataFrame(data['omg_ref'], columns=['omg_ref_x', 'omg_ref_y', 'omg_ref_z'])
omg_est = pd.DataFrame(data['omg_est'], columns=['omg_est_x', 'omg_est_y', 'omg_est_z'])
thr_ref = pd.DataFrame(data['thr_ref'], columns=['thr_ref'])

df_log = pd.concat([pos_ref, pos_est, vel_ref, vel_est, euler_ref, euler_est, omg_ref, omg_est, thr_ref], axis=1)
df_log = df_log.iloc[:len(data['time']), :]
df_log.index = data['time'] - data['time'][0]
df_log.index.name = 'time'

# position
fig = make_subplots(rows=3, cols=1, x_title='time [s]')
trace0 = go.Scatter(
 x = df_log.index,
 y = df_log.pos_ref_x,
 mode = "lines+markers",
 name = "ref pos x",
 marker = dict(color='rgba(80,26,80,0.8)'),
 text = df_log.pos_ref_x)
trace1 = go.Scatter(
 x = df_log.index,
 y = df_log.pos_est_x,
 mode = "lines+markers",
 name = "est pos x",
 marker = dict(color='rgba(26,80,80,0.8)'),
 text = df_log.pos_est_x)
trace2 = go.Scatter(
 x = df_log.index,
 y = df_log.pos_ref_y,
 mode = "lines+markers",
 name = "ref pos y",
 marker = dict(color='rgba(80,26,80,0.8)'),
 text = df_log.pos_ref_y)
trace3 = go.Scatter(
 x = df_log.index,
 y = df_log.pos_est_y,
 mode = "lines+markers",
 name = "est pos y",
 marker = dict(color='rgba(26,80,80,0.8)'),
 text = df_log.pos_est_y)
trace4 = go.Scatter(
 x = df_log.index,
 y = df_log.pos_ref_z,
 mode = "lines+markers",
 name = "ref pos z",
 marker = dict(color='rgba(80,26,80,0.8)'),
 text = df_log.pos_ref_z)
trace5 = go.Scatter(
 x = df_log.index,
 y = df_log.pos_est_z,
 mode = "lines+markers",
 name = "est pos z",
 marker = dict(color='rgba(26,80,80,0.8)'),
 text = df_log.pos_est_z)
fig.append_trace(trace0, row=1, col=1)
fig.append_trace(trace1, row=1, col=1)
fig.append_trace(trace2, row=2, col=1)
fig.append_trace(trace3, row=2, col=1)
fig.append_trace(trace4, row=3, col=1)
fig.append_trace(trace5, row=3, col=1)
fig.update_layout(height=1000, title_text="Position Error Statistics")
fig['layout']['yaxis'].update({'title': 'Forward [m]'})
fig['layout']['yaxis2'].update({'title': 'Left [m]'})
fig['layout']['yaxis3'].update({'title': 'Up [m]'})
fig.write_html(out_path + "/1_Position_Error_Statistics.html")
fig = go.Figure()
error0 = df_log.pos_ref_x - df_log.pos_est_x
fig.add_trace(go.Box(y=error0, name="Forward(x)"))
error1 = df_log.pos_ref_y - df_log.pos_est_y
fig.add_trace(go.Box(y=error1, name="Left(y)"))
error2 = df_log.pos_ref_z - df_log.pos_est_z
fig.add_trace(go.Box(y=error2, name="Up(z)"))
err_norm = np.sqrt((np.square(error0) + np.square(error1) + np.square(error2)).astype(np.float64))
fig.update_layout(
    title_text=f'''Position Error Box Graph
    x error[{np.around(np.mean(error0), 4)}±{np.around(np.std(error0), 4)}]
    y error[{np.around(np.mean(error1), 4)}±{np.around(np.std(error1), 4)}]
    z error[{np.around(np.mean(error2), 4)}±{np.around(np.std(error2), 4)}]
    norm[{np.around(np.mean(err_norm), 4)}±{np.around(np.std(err_norm), 4)}]''',
    yaxis_title_text="Position [m]")
fig.write_html(out_path + "/2_Position_Error_Box_Graph.html")

# velocity
fig = make_subplots(rows=3, cols=1, x_title='time [s]')
trace0 = go.Scatter(
 x = df_log.index,
 y = df_log.vel_ref_x,
 mode = "lines+markers",
 name = "ref vel x",
 marker = dict(color='rgba(80,26,80,0.8)'))
trace1 = go.Scatter(
 x = df_log.index,
 y = df_log.vel_est_x,
 mode = "lines+markers",
 name = "est vel x",
 marker = dict(color='rgba(26,80,80,0.8)'))
trace2 = go.Scatter(
 x = df_log.index,
 y = df_log.vel_ref_y,
 mode = "lines+markers",
 name = "ref vel y",
 marker = dict(color='rgba(80,26,80,0.8)'))
trace3 = go.Scatter(
 x = df_log.index,
 y = df_log.vel_est_y,
 mode = "lines+markers",
 name = "est vel y",
 marker = dict(color='rgba(26,80,80,0.8)'))
trace4 = go.Scatter(
 x = df_log.index,
 y = df_log.vel_ref_z,
 mode = "lines+markers",
 name = "ref vel z",
 marker = dict(color='rgba(80,26,80,0.8)'))
trace5 = go.Scatter(
 x = df_log.index,
 y = df_log.vel_est_z,
 mode = "lines+markers",
 name = "est vel z",
 marker = dict(color='rgba(26,80,80,0.8)'))
fig.append_trace(trace0, row=1, col=1)
fig.append_trace(trace1, row=1, col=1)
fig.append_trace(trace2, row=2, col=1)
fig.append_trace(trace3, row=2, col=1)
fig.append_trace(trace4, row=3, col=1)
fig.append_trace(trace5, row=3, col=1)
fig.update_layout(height=1000, title_text="Velocity Error Statistics")
fig['layout']['yaxis'].update({'title': 'Forward [m/s]'})
fig['layout']['yaxis2'].update({'title': 'Left [m/s]'})
fig['layout']['yaxis3'].update({'title': 'Up [m/s]'})
fig.write_html(out_path + "/3_Velocity_Error_Statistics.html")
fig = go.Figure()
error0 = df_log.vel_ref_x - df_log.vel_est_x
fig.add_trace(go.Box(y=error0, name="Forward(x)"))
error1 = df_log.vel_ref_y - df_log.vel_est_y
fig.add_trace(go.Box(y=error1, name="Left(y)"))
error2 = df_log.vel_ref_z - df_log.vel_est_z
fig.add_trace(go.Box(y=error2, name="Up(z)"))
err_norm = np.sqrt((np.square(error0) + np.square(error1) + np.square(error2)).astype(np.float64))
fig.update_layout(
    title_text=f'''Velocity Error Box Graph
    x error[{np.around(np.mean(error0), 4)}±{np.around(np.std(error0), 4)}]
    y error[{np.around(np.mean(error1), 4)}±{np.around(np.std(error1), 4)}]
    z error[{np.around(np.mean(error2), 4)}±{np.around(np.std(error2), 4)}]
    norm[{np.around(np.mean(err_norm), 4)}±{np.around(np.std(err_norm), 4)}]''',
    yaxis_title_text="Velocity [m/s]")
fig.write_html(out_path + "/4_Velocity_Error_Box_Graph.html")

# attitude
fig = make_subplots(rows=3, cols=1, x_title='time [s]')
trace0 = go.Scatter(
 x = df_log.index,
 y = df_log.att_ref_x * 57.3,
 mode = "lines+markers",
 name = "ref att x",
 marker = dict(color='rgba(80,26,80,0.8)'))
trace1 = go.Scatter(
 x = df_log.index,
 y = df_log.att_est_x * 57.3,
 mode = "lines+markers",
 name = "est att x",
 marker = dict(color='rgba(26,80,80,0.8)'))
trace2 = go.Scatter(
 x = df_log.index,
 y = df_log.att_ref_y * 57.3,
 mode = "lines+markers",
 name = "ref att y",
 marker = dict(color='rgba(80,26,80,0.8)'))
trace3 = go.Scatter(
 x = df_log.index,
 y = df_log.att_est_y * 57.3,
 mode = "lines+markers",
 name = "est att y",
 marker = dict(color='rgba(26,80,80,0.8)'))
trace4 = go.Scatter(
 x = df_log.index,
 y = df_log.att_ref_z * 57.3,
 mode = "lines+markers",
 name = "ref att z",
 marker = dict(color='rgba(80,26,80,0.8)'))
trace5 = go.Scatter(
 x = df_log.index,
 y = df_log.att_est_z * 57.3,
 mode = "lines+markers",
 name = "est att z",
 marker = dict(color='rgba(26,80,80,0.8)'))
fig.append_trace(trace0, row=1, col=1)
fig.append_trace(trace1, row=1, col=1)
fig.append_trace(trace2, row=2, col=1)
fig.append_trace(trace3, row=2, col=1)
fig.append_trace(trace4, row=3, col=1)
fig.append_trace(trace5, row=3, col=1)
fig.update_layout(height=1000, title_text="Attitude Error Statistics")
fig['layout']['yaxis'].update({'title': 'Roll [deg]'})
fig['layout']['yaxis2'].update({'title': 'Pitch [deg]'})
fig['layout']['yaxis3'].update({'title': 'Yaw [deg]'})
fig.write_html(out_path + "/5_Attitude_Error_Statistics.html")
fig = go.Figure()
error0 = (df_log.att_ref_x - df_log.att_est_x).apply(lambda x: wrapPi(x))
fig.add_trace(go.Box(y=error0 * 57.3, name="Roll"))
error1 = (df_log.att_ref_y - df_log.att_est_y).apply(lambda x: wrapPi(x))
fig.add_trace(go.Box(y=error1 * 57.3, name="Pitch"))
error2 = (df_log.att_ref_z - df_log.att_est_z).apply(lambda x: wrapPi(x))
fig.add_trace(go.Box(y=error2 * 57.3, name="Yaw"))
err_norm = np.sqrt((np.square(error0) + np.square(error1) + np.square(error2)).astype(np.float64))
fig.update_layout(
    title_text=f'''Attitude Error Box Graph
    roll error[{np.around(np.mean(error0), 4)}±{np.around(np.std(error0), 4)}]
    pitch error[{np.around(np.mean(error1), 4)}±{np.around(np.std(error1), 4)}]
    yaw error[{np.around(np.mean(error2), 4)}±{np.around(np.std(error2), 4)}]
    norm[{np.around(np.mean(err_norm), 4)}±{np.around(np.std(err_norm), 4)}]''',
    yaxis_title_text="Attitude [deg]")
fig.write_html(out_path + "/6_Attitude_Error_Box_Graph.html")

# angular rate
fig = make_subplots(rows=3, cols=1, x_title='time [s]')
trace0 = go.Scatter(
 x = df_log.index,
 y = df_log.omg_ref_x * 57.3,
 mode = "lines+markers",
 name = "ref omg x",
 marker = dict(color='rgba(80,26,80,0.8)'))
trace1 = go.Scatter(
 x = df_log.index,
 y = df_log.omg_est_x * 57.3,
 mode = "lines+markers",
 name = "est omg x",
 marker = dict(color='rgba(26,80,80,0.8)'))
trace2 = go.Scatter(
 x = df_log.index,
 y = -df_log.omg_ref_y * 57.3,
 mode = "lines+markers",
 name = "ref omg y",
 marker = dict(color='rgba(80,26,80,0.8)'))
trace3 = go.Scatter(
 x = df_log.index,
 y = df_log.omg_est_y * 57.3,
 mode = "lines+markers",
 name = "est omg y",
 marker = dict(color='rgba(26,80,80,0.8)'))
trace4 = go.Scatter(
 x = df_log.index,
 y = -df_log.omg_ref_z * 57.3,
 mode = "lines+markers",
 name = "ref omg z",
 marker = dict(color='rgba(80,26,80,0.8)'))
trace5 = go.Scatter(
 x = df_log.index,
 y = df_log.omg_est_z * 57.3,
 mode = "lines+markers",
 name = "est omg z",
 marker = dict(color='rgba(26,80,80,0.8)'))
fig.append_trace(trace0, row=1, col=1)
fig.append_trace(trace1, row=1, col=1)
fig.append_trace(trace2, row=2, col=1)
fig.append_trace(trace3, row=2, col=1)
fig.append_trace(trace4, row=3, col=1)
fig.append_trace(trace5, row=3, col=1)
fig.update_layout(height=1000, title_text="Angular Rate Error Statistics")
fig['layout']['yaxis'].update({'title': 'Roll Rate [deg/s]'})
fig['layout']['yaxis2'].update({'title': 'Pitch Rate [deg/s]'})
fig['layout']['yaxis3'].update({'title': 'Yaw Rare [deg/s]'})
fig.write_html(out_path + "/7_Angular_Rate_Error_Statistics.html")
fig = go.Figure()
error0 = df_log.omg_ref_x - df_log.omg_est_x
fig.add_trace(go.Box(y=error0 * 57.3, name="Roll Rate(p)"))
error1 = -df_log.omg_ref_y - df_log.omg_est_y
fig.add_trace(go.Box(y=error1 * 57.3, name="Pitch Rate(q)"))
error2 = -df_log.omg_ref_z - df_log.omg_est_z
fig.add_trace(go.Box(y=error2 * 57.3, name="Yaw Rate(r)"))
err_norm = np.sqrt((np.square(error0) + np.square(error1) + np.square(error2)).astype(np.float64))
fig.update_layout(
    title_text=f'''Angular Rate Error Box Graph
    p error[{np.around(np.mean(error0), 4)}±{np.around(np.std(error0), 4)}]
    q error[{np.around(np.mean(error1), 4)}±{np.around(np.std(error1), 4)}]
    r error[{np.around(np.mean(error2), 4)}±{np.around(np.std(error2), 4)}]
    norm[{np.around(np.mean(err_norm), 4)}±{np.around(np.std(err_norm), 4)}]''',
    yaxis_title_text="Angular Rate [deg/s]")
fig.write_html(out_path + "/8_Angular_Rate_Error_Box_Graph.html")

# thrust ref
fig = make_subplots(rows=1, cols=1, x_title='time [s]')
trace0 = go.Scatter(
 x = df_log.index,
 y = df_log.thr_ref,
 mode = "lines+markers",
 name = "thrust ref",
 marker = dict(color='rgba(80,26,80,0.8)'))
fig.append_trace(trace0, row=1, col=1)
fig.update_layout(height=1000, title_text="Thrust Command")
fig['layout']['yaxis'].update({'title': 'Thrust'})
fig.write_html(out_path + "/9_Thrust.html")

# trajectory
fig = go.Figure()
trace0 = go.Scatter3d(
 x=df_log.pos_ref_x,
 y=df_log.pos_ref_y,
 z=df_log.pos_ref_z,
 mode='lines',
 name='command',
 marker=dict(color='rgba(80,26,80,0.8)'))
trace1 = go.Scatter3d(
 x = df_log.pos_est_x,
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
fig.write_html(out_path + "/10_Trajectory.html")