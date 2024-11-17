from typing import Dict

try:
    import rospy
except:
    print("rospy not found, use rclpy instead")
    import rclpy
import torch
from geometry_msgs.msg import Pose
from nav_msgs.msg import Odometry
try:
    from quadrotor_msgs.msg import ControlCommand
except:
    print("quadrotor_msgs not found, import a meaningless pkg as ControlCommand for holding the type")
    import abc as ControlCommand # ugly operation


ENUM_CTRL_MODE = {
    'NONE': 0,
    'ATTITUDE': 1,
    'BODY_RATES': 2,
    'ANGULAR_ACCELERATIONS': 3,
    'ROTOR_THRUSTS': 4
}

def parse_odom(odom_msg: Odometry):
    """
    Parse Odometry message into a dictionary of tensors
    """
    odom_dict = {}
    odom_dict['position'] = torch.tensor([odom_msg.pose.pose.position.x,
                                          odom_msg.pose.pose.position.y,
                                          odom_msg.pose.pose.position.z])
    odom_dict['orientation'] = torch.tensor([odom_msg.pose.pose.orientation.w,
                                             odom_msg.pose.pose.orientation.x,
                                             odom_msg.pose.pose.orientation.y,
                                             odom_msg.pose.pose.orientation.z])
    odom_dict['linear_velocity'] = torch.tensor([odom_msg.twist.twist.linear.x,
                                                 odom_msg.twist.twist.linear.y,
                                                 odom_msg.twist.twist.linear.z])
    odom_dict['angular_velocity'] = torch.tensor([odom_msg.twist.twist.angular.x,
                                                  odom_msg.twist.twist.angular.y,
                                                  odom_msg.twist.twist.angular.z])
    return odom_dict

def parse_target_ref(target_ref_msg: Pose):
    """
    Parse target reference message into a dictionary of tensors
    """
    target_ref_dict = {}
    target_ref_dict['position'] = torch.tensor([target_ref_msg.position.x,
                                                target_ref_msg.position.y,
                                                target_ref_msg.position.z])
    target_ref_dict['orientation'] = torch.tensor([target_ref_msg.orientation.w,
                                                   target_ref_msg.orientation.x,
                                                   target_ref_msg.orientation.y,
                                                   target_ref_msg.orientation.z])
    return target_ref_dict

def generate_cmd_msg(cmd: torch.Tensor, cmd_mode: str = 'BODY_RATES'):
    """
    Convert ControlCommand message to a dictionary of tensors
    :param ctbr: mass-normalized collective thrust and body rate
    :param ctrl_mode: control mode for ControlCommand 
        (0: NONE, 1: ATTITUDE, 2: BODY_RATES, 3: ANGULAR_ACCELERATIONS, 4: ROTOR_THRUSTS)
    """
    
    assert cmd.shape == (4,), "cmd should be a 4-element tensor"
    assert cmd_mode == 'BODY_RATES', "Only BODY_RATES mode is supported for now"
    
    msg = ControlCommand()
    msg.armed = True
    msg.control_mode = ENUM_CTRL_MODE[cmd_mode]
    # TODO: set a min thrust threshold, refer to the autopilot
    msg.collective_thrust = cmd[0].item()
    msg.bodyrates.x = cmd[1].item()
    msg.bodyrates.y = cmd[2].item()
    msg.bodyrates.z = cmd[3].item()
    
    return msg

def generate_odom_msg(odom_dict: Dict, timestamp: float = None, frame_id: str = 'unnamed'):
    """
    Convert dictionary of tensors to Odometry message
    """
    msg = Odometry()
    if timestamp is None:
        msg.header.stamp = rospy.Time.now()
    else:
        assert isinstance(timestamp, float), "timestamp should be a float"
        msg.header.stamp = rospy.Time.from_sec(timestamp)
    msg.header.frame_id = 'vision'
    msg.child_frame_id = frame_id
    if 'position' in odom_dict:
        msg.pose.pose.position.x = odom_dict['position'][0].item()
        msg.pose.pose.position.y = odom_dict['position'][1].item()
        msg.pose.pose.position.z = odom_dict['position'][2].item()
    if 'orientation' in odom_dict:
        msg.pose.pose.orientation.w = odom_dict['orientation'][0].item()
        msg.pose.pose.orientation.x = odom_dict['orientation'][1].item()
        msg.pose.pose.orientation.y = odom_dict['orientation'][2].item()
        msg.pose.pose.orientation.z = odom_dict['orientation'][3].item()
    if 'linear_velocity' in odom_dict:
        msg.twist.twist.linear.x = odom_dict['linear_velocity'][0].item()
        msg.twist.twist.linear.y = odom_dict['linear_velocity'][1].item()
        msg.twist.twist.linear.z = odom_dict['linear_velocity'][2].item()
    if 'angular_velocity' in odom_dict:
        msg.twist.twist.angular.x = odom_dict['angular_velocity'][0].item()
        msg.twist.twist.angular.y = odom_dict['angular_velocity'][1].item()
        msg.twist.twist.angular.z = odom_dict['angular_velocity'][2].item()
    return msg

def parse_cmd_msg(msg: ControlCommand) -> torch.Tensor:
    """
    Invert ControlCommand message to a tensor
    """
    assert msg.control_mode == ENUM_CTRL_MODE['BODY_RATES'], \
        "Only BODY_RATES mode is supported for now, but got {}".format(msg.control_mode)
    
    cmd = torch.zeros(4)
    cmd[0] = msg.collective_thrust
    cmd[1] = msg.bodyrates.x
    cmd[2] = msg.bodyrates.y
    cmd[3] = msg.bodyrates.z
    
    return cmd
