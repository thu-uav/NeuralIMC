import threading
from typing import Callable, Dict

import rospy
import tf
import torch
from nav_msgs.msg import Odometry
from quadrotor_msgs.msg import ControlCommand  # in rpg_quadrotor_common
from torch_control.sim2real.utils.ros_utils import (generate_odom_msg,
                                                    parse_cmd_msg, parse_odom)


class SimulatorNode:
    def __init__(self, odom_topic: str, cmd_topic: str, odom_timeout: float = 0.1, cmd_timeout: float = 0.1):
        # rospy.init_node('torch_control_sim')

        # Separate locks for odometry and target reference
        self.odom_lock = threading.Lock()
        self.cmd_lock = threading.Lock()

        # Subscribers
        self.odom_sub = rospy.Subscriber(odom_topic, Odometry, self.odom_callback)
        self.cmd_sub = rospy.Subscriber(cmd_topic, ControlCommand, self.cmd_callback)
        
        self.odom_timeout = odom_timeout
        self.cmd_timeout = cmd_timeout
        
        # Publishers
        self.sim_odom_pub = rospy.Publisher('/torch_control/sim_odom', Odometry, queue_size=10)

        # Shared Resources
        self.latest_odom = None
        self.latest_cmd = None
        self.last_odom_time = rospy.Time.now()
        self.last_cmd_time = rospy.Time.now()
        
        # TF Broadcaster
        self.tf_broadcaster = tf.TransformBroadcaster()
        
    def setup_sim(self,
                  sim_dt: float,
                  step_fn: Callable[[Dict[str, torch.Tensor], float], Dict[str, torch.Tensor]]):
        self.sim_dt = sim_dt
        self.step_fn = step_fn
        
    def cmd_callback(self, cmd_msg: ControlCommand):
        with self.cmd_lock:
            self.latest_cmd = cmd_msg
            self.last_cmd_time = rospy.Time.now()
        
    def odom_callback(self, odom_msg: Odometry):
        with self.odom_lock:
            self.latest_odom = odom_msg
            self.last_odom_time = rospy.Time.now()
            
            current_odom_dict = parse_odom(odom_msg)
            current_timestamp = odom_msg.header.stamp.to_sec()
            
        if self.latest_cmd is None:
            return
        
        with self.cmd_lock:
            current_action = parse_cmd_msg(self.latest_cmd)
            
        env_input = {**current_odom_dict, 'timestamp': current_timestamp, 'action': current_action}
        
        start_time = rospy.Time.now()
        env_output = self.step_fn(env_input, self.sim_dt)
        end_time = rospy.Time.now()
        duration = (end_time - start_time).to_sec()
        
        if duration > self.sim_dt:
            rospy.logwarn(f"Simulation step took longer than expected sim_dt ({duration:.04f} > {self.sim_dt:0.4f})")
        
        new_timestamp = current_timestamp + self.sim_dt
        
        sim_odom_msg = generate_odom_msg(env_output, new_timestamp)
        
        # Broadcast the transformation
        position = (
            sim_odom_msg.pose.pose.position.x,
            sim_odom_msg.pose.pose.position.y,
            sim_odom_msg.pose.pose.position.z
        )
        orientation = (
            sim_odom_msg.pose.pose.orientation.x,
            sim_odom_msg.pose.pose.orientation.y,
            sim_odom_msg.pose.pose.orientation.z,
            sim_odom_msg.pose.pose.orientation.w
        )
        self.tf_broadcaster.sendTransform(
            position,
            orientation,
            sim_odom_msg.header.stamp,
            sim_odom_msg.child_frame_id,
            sim_odom_msg.header.frame_id
        )
        
        self.sim_odom_pub.publish(sim_odom_msg)
            
    def run(self):
        rospy.spin()
        
