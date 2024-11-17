import argparse
import copy
import threading
import time
from typing import Callable, Dict, List, Union

import numpy as np
import torch

from torch_control.sim2real.utils import ros_utils
from torch_control.utils import rot_utils
from torch_control.utils.common import set_all_seed

try:
    import rospy
except:
    print("rospy not found, use rclpy instead")
    try:
        import rclpy
    except:
        print("rospy and rclpy not found, will not use ROS")

try:
    from geometry_msgs.msg import Pose
    from nav_msgs.msg import Odometry
    from std_msgs.msg import Empty, Float32
except:
    print("ROS message type not found, will not use ROS")


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

class DeployHelper:
    def __init__(self, 
                 start_topic = "/torchctrl/start", 
                 target_ref_topic = "/torchctrl/target_ref", 
                 quad = 'air'):
        self.disabled_ros = False
        
        rospy.init_node('deploy_helper', anonymous=True)
        self.start_pub = rospy.Publisher(start_topic, Float32, queue_size=1)
        self.target_ref_pub = rospy.Publisher(target_ref_topic, Pose, queue_size=1)
            
        if quad == 'air':
            odom_topic = "/air/autopilot/state_estimate"
        else:
            odom_topic = "/{}/pose".format(quad)
            
        print("[Odometry topic] ", odom_topic)

        self.state_sub = rospy.Subscriber(odom_topic, Odometry, self.state_callback)
        self.latest_odom = None
        self.origin_odom = None
        
    def state_callback(self, odom_msg: Odometry):
        self.latest_odom = ros_utils.parse_odom(odom_msg)
        self.last_odom_time = rospy.Time.now()
        
    def get_pose_msg(self, pos: List = None, yaw: float = None, offset: bool = False):
        
        if pos is None:
            pos = [0., 0., 0.]
        else:
            assert len(pos) == 3, "Invalid length of pos"
            
        if yaw is None:
            yaw = 0.
            
        if offset:
            pos = [pos[0] + self.origin_pos[0], 
                   pos[1] + self.origin_pos[1], 
                   pos[2] + self.origin_pos[2]]
            yaw = yaw + self.origin_ang[2]
        
        ang = [0., 0., yaw]
            
        orientation = euler2quat(ang)
        
        if self.disabled_ros:
            return pos, orientation

        msg = Pose()

        msg.position.x = float(pos[0])
        msg.position.y = float(pos[1])
        msg.position.z = float(pos[2])
        
        msg.orientation.w = float(orientation[0])
        msg.orientation.x = float(orientation[1])
        msg.orientation.y = float(orientation[2])
        msg.orientation.z = float(orientation[3])
        
        return msg
    
    def run(self):
        print("Welcome to Deploy Helper!")
        def parse_pos(pos):
            if pos == "":
                return None
            else:
                ret = [float(x) for x in pos.split(' ')]
                assert len(ret) == 3
                return ret
        def parse_ang(ang):
            if ang == "":
                return None
            elif ' ' in ang:
                ret = [float(x) for x in ang.split(' ')]
                assert len(ret) == 3
                return ret
            else:
                return float(ang)
        while True:
            task_name = input("Please check and input the task name (setpoint | track): ")
            
            if task_name not in ['setpoint', 'track']:
                print("[error] Check task name!!!")
                break
            
            origin_init = False
            while not origin_init:
                yn = input("Please input `y` to set the current position as origin: ")
                if yn == 'y':
                    origin_init = True
                    self.origin_odom = self.latest_odom
                    self.origin_pos = self.origin_odom['position']
                    self.origin_ang = rot_utils.quat2euler(self.origin_odom['orientation'])

            pos = '0 0 0'
            if task_name in ['setpoint', 'track']:
                pos = input("Please input position (separated by space, default: 0 0 0): ")
            pos = parse_pos(pos)

            ang = '0'
            if task_name in ['setpoint', 'rotate_yaw']:
                ang = input("Please input yaw angle (in degree, default: 0): ")
            ang = parse_ang(ang)

            period = np.float32(3600) # second
            str_period = input("Please input task time period (default: 3600s): ")
            if str_period != "":
                period = np.float32(str_period)
            
            if self.disabled_ros:
                pos, quat = self.get_pose_msg(pos, ang, offset=(task_name == 'setpoint'))
                return pos, quat
            else:
                pose_msg = self.get_pose_msg(pos, ang, offset=(task_name == 'setpoint'))
            
            print("Sending message...")
            print("Message: {}".format(pose_msg))
            
            print("Input 'y' to start the task, or 'n' to cancel, or 'q' to quit.")
            start = input("Input: ")
            if start == 'y':
                self.target_ref_pub.publish(pose_msg)
                time.sleep(0.5)
                self.start_pub.publish(Float32(period))
                print("Task started!")
            elif start == 'n':
                print("Task cancelled!")
                continue
            else:
                print("Program terminated!")
                break
   
            print("Input 's' to stop the current task, or 'q' to quit, or 'c' to continue next task")
            s = input("Stop or Quit or Continue: ")
            if s == 's':
                self.start_pub.publish(Float32(-1))
                break
            elif s == 'q':
                break
            else:
                continue

    def run_setpoint(self):
        origin_init = False
        while not origin_init:
            yn = input("Please input `y` to set the current position as origin: ")
            if yn == 'y':
                origin_init = True
                self.origin_odom = self.latest_odom
                self.origin_pos = self.origin_odom['position']
                self.origin_ang = rot_utils.quat2euler(self.origin_odom['orientation'])
                
        for i in range(5):
            pos = [np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5), np.random.uniform(0.0, 0.3)]
            ang = np.random.uniform(-90, 90)
            period = np.float32(5)
            
            pose_msg = self.get_pose_msg(pos, ang, offset=True)
            self.target_ref_pub.publish(pose_msg)
            time.sleep(0.5)
            self.start_pub.publish(Float32(period))
            
            print("[Point {}] Task started!".format(i+1))
            print("Position: {} | Yaw: {}".format(pos, ang))
            time.sleep(6)
        
        self.start_pub.publish(Float32(-1))
        
    def run_track(self):
        origin_init = False
        while not origin_init:
            yn = input("Please input `y` to set the current position as origin: ")
            if yn == 'y':
                origin_init = True
                self.origin_odom = self.latest_odom
                self.origin_pos = self.origin_odom['position']
                self.origin_ang = rot_utils.quat2euler(self.origin_odom['orientation'])

        pos = [0, 0, 0]
        ang = 0
        period = np.float32(20)
        pose_msg = self.get_pose_msg(pos, ang, offset=False)
        self.target_ref_pub.publish(pose_msg)
        time.sleep(0.5)
        self.start_pub.publish(Float32(period))
        time.sleep(11)
        self.start_pub.publish(Float32(-1))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ros', default=1, type=int)
    parser.add_argument('--quad', default='air', type=str)
    parser.add_argument('--setpoint', default=False, action='store_true')
    parser.add_argument('--track', default=False, action='store_true')
    parser.add_argument('--seed', default=0, type=int)
    args = parser.parse_args()
    
    set_all_seed(args.seed)
    
    helper = DeployHelper(quad=args.quad)
    if args.setpoint:
        helper.run_setpoint()
    elif args.track:
        helper.run_track()
    else:
        helper.run()
