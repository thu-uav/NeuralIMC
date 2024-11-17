import copy
import os
import signal
import sys
import threading
import time
from typing import Any, Callable, Dict

import numpy as np
import rospy
import torch
from geometry_msgs.msg import Point, Pose
from nav_msgs.msg import Odometry
from quadrotor_msgs.msg import ControlCommand  # in rpg_quadrotor_common
from std_msgs.msg import Empty, Float32
from torch_control.sim2real.utils import ros_utils
from torch_control.utils.rot_utils import np_euler_distance


class ControllerNode:
    def __init__(self, 
                 odom_topic: str, 
                 cmd_topic: str, 
                 cmd_feedback_topic: str, 
                 logging_topic_prefix: str,
                 start_topic: str,
                 target_ref_topic: str, 
                 odom_timeout: float = 0.1,
                 check_l1ac: bool = False,
                 verbose: bool = False):
        self.verbose = verbose

        # Separate locks for odometry and target reference
        self.odom_lock = threading.Lock()
        self.target_ref_lock = threading.Lock()

        # Shared Resources
        self.latest_odom = None
        self.latest_target_ref = None
        self.last_odom_time = rospy.Time.now()
        
        self.check_l1ac = check_l1ac
        self.task_ready = False
        self.task_running = False
        self.task_period = 300 # seconds

        # Subscribers
        self.odom_sub = rospy.Subscriber(odom_topic, Odometry, self.odom_callback)
        self.start_sub = rospy.Subscriber(start_topic, Float32, self.start_callback)
        self.target_ref_sub = rospy.Subscriber(target_ref_topic, Pose, self.target_ref_callback)
        
        self.odom_timeout = odom_timeout

        # Publishers
        self.cmd_pub = rospy.Publisher(cmd_topic, ControlCommand, queue_size=10)
        self.state_logging_pub = rospy.Publisher(logging_topic_prefix + "_state", Odometry, queue_size=10)
        self.ref_logging_pub = rospy.Publisher(logging_topic_prefix + "_ref", Odometry, queue_size=10)
        self.l1ac_logging_pub = rospy.Publisher(logging_topic_prefix + "_l1ac", Point, queue_size=10)

        self.logging_data = {}

        signal.signal(signal.SIGINT, self.emergency_handler)  # Handling Ctrl+C

    def setup_controller(self, 
                         ctrl_freq: float, 
                         init_task_fn: Callable[[Dict, Dict], None],
                         ctrl_fn: Callable[[torch.Tensor], torch.Tensor],
                         mask_bodyrate: bool = False,
                         run_name: str = 'null'):
        self.ctrl_freq = ctrl_freq
        self.init_task_fn = init_task_fn
        self.ctrl_fn = ctrl_fn
        self.run_name = run_name
        self.mask_bodyrate = mask_bodyrate

        # Setup timer for control loop
        self._ctrl_timer = rospy.Timer(rospy.Duration(1.0/ctrl_freq), self.timer_callback)
            
    def start_callback(self, start_msg: Float32):
        if self.task_running and start_msg.data < 0:
            self.task_running = False
            print("[ControllerNode] Task stopped")
            self.write_to_log()
        elif start_msg.data < 0:
            self.task_running = False
        else:
            self.task_running = True
            self.task_period = start_msg.data
            self.start_t = time.time()
        print("[ControllerNode] Task running:", bool(self.task_running))
    
    def odom_callback(self, odom_msg: Odometry):
        with self.odom_lock:
            self.latest_odom = ros_utils.parse_odom(odom_msg)
            self.last_odom_time = rospy.Time.now()

    def target_ref_callback(self, target_ref_msg: Pose):
        with self.odom_lock:
            assert self.latest_odom is not None, "Odometry data not received yet"
            latest_odom_copy = copy.copy(self.latest_odom)
                
        with self.target_ref_lock:
            self.latest_target_ref = ros_utils.parse_target_ref(target_ref_msg)
        
        print("[ControllerNode] Received target reference:", self.latest_target_ref)

        self.init_task_fn(latest_odom_copy, self.latest_target_ref)
        self.task_ready = True

        print("[ControllerNode] Task is ready to run!")
                    
    def emergency_handler(self, signum, frame):
        '''
        Intercepts Ctrl+C from the User and stops the motor of the drone.
        '''
        print("User Emergency Stop!")
        if self.verbose:
            self.write_to_log()
        exit()
    
    def timer_callback(self, event):
        if not self.task_ready or not self.task_running:
            return 
        
        if time.time() - self.start_t > self.task_period:
            self.task_running = False
            print("Timeout, stop running")
            return
        
        with self.odom_lock:
            if self.latest_odom is None:
                return
            latest_odom_copy = copy.copy(self.latest_odom)
            last_odom_time_copy = copy.copy(self.last_odom_time)

        duration_since_last_odom = (rospy.Time.now() - last_odom_time_copy).to_sec()
        # Check for odometry timeout
        if duration_since_last_odom > self.odom_timeout:
            rospy.logwarn(f"Odometry data timeout ({duration_since_last_odom:.04f} > {self.odom_timeout:.04f})")

        start_time = rospy.Time.now()
        if getattr(self, 't0', None) is None:
            self.t0 = start_time
            
        elapsed_time = (start_time - self.t0).to_sec()

        if not self.check_l1ac:
            # Preprocess data
            cmd_tensor, logging_data, logging_info = self.ctrl_fn(latest_odom_copy, elapsed_time)

            if self.verbose:
                for key, value in logging_data.items():
                    if key not in self.logging_data:
                        self.logging_data[key] = []
                    self.logging_data[key].append(value)

            if self.mask_bodyrate:
                cmd_tensor[..., 1:] = 0.
            
            cmd_msg = ros_utils.generate_cmd_msg(cmd_tensor)
            self.publish_logging_info(logging_info)
            end_time = rospy.Time.now()
            self.publish_cmd(cmd_msg)

            cmd_msg.header.stamp = end_time
            inference_time = (end_time - start_time).to_sec()
        
            if inference_time > 1.0/self.ctrl_freq:
                rospy.logwarn(f"Controller inference taking longer ({inference_time:.04f}) than expected ({1.0/self.ctrl_freq:.04f})")
        else:
            cmd_tensor, logging_data, logging_info = self.ctrl_fn(latest_odom_copy, elapsed_time)
            self.publish_logging_info(logging_info)
        
    def publish_cmd(self, cmd_msg: ControlCommand):
        self.cmd_pub.publish(cmd_msg)
        
    def publish_logging_info(self, logging_info: Dict):
        state_msg = ros_utils.generate_odom_msg(logging_info['state'], logging_info['time'])
        ref_msg = ros_utils.generate_odom_msg(logging_info['ref'], logging_info['time'])
        
        self.state_logging_pub.publish(state_msg)
        self.ref_logging_pub.publish(ref_msg)
        
        if 'l1ac' in logging_info:
            l1ac_msg = Point()
            l1ac_msg.x = logging_info['l1ac'][0].item()
            l1ac_msg.y = logging_info['l1ac'][1].item()
            l1ac_msg.z = logging_info['l1ac'][2].item()
            self.l1ac_logging_pub.publish(l1ac_msg)
    
    def run(self):
        rospy.spin()

    # Log all the data at the end of the experiment
    def write_to_log(self):
        import time
        time_str = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        LOG_PATH = f"./data/air_log/{self.run_name}_{time_str}.npz"
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        print(f"LOG PATH: {LOG_PATH}")

        np.savez(LOG_PATH, **{
                key: np.array(value) for key, value in self.logging_data.items()})
        
        pos_error = np.array(self.logging_data['pos_est']) - np.array(self.logging_data['pos_ref'])
        att_error = np_euler_distance(np.array(self.logging_data['att_est']),
                                      np.array(self.logging_data['att_ref']), 
                                      norm=False) * 180. / np.pi
        
        avg_error = {'pos': np.mean(pos_error, axis=0),
                     'pos_xy_norm': np.mean(np.linalg.norm(pos_error[:, :2], axis=1)),
                     'pos_norm': np.mean(np.linalg.norm(pos_error, axis=1)),
                     'pos_std': np.std(pos_error, axis=0),
                     'att': np.mean(att_error, axis=0),
                     'att_norm': np.mean(np.linalg.norm(att_error, axis=1)),
                     'att_std': np.std(att_error, axis=0)}
        last_error = {'pos': pos_error[-1],
                      'pos_xy_norm': np.linalg.norm(pos_error[-1, :2]),
                      'pos_norm': np.linalg.norm(pos_error[-1]),
                      'att': att_error[-1],
                      'att_norm': np.linalg.norm(att_error[-1])}
        
        print("--------------------")
        print(f"Average Error: {avg_error}")
        print(f"Last Error: {last_error}")

        # run parse_logs.py to visualize the data, e.g. python parse_logs.py --log_name=LOG_PATH
        # os.system(f"python parse_logs.py --log_name={LOG_PATH}")
