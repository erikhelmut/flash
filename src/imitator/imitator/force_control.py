import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from std_msgs.msg import Int16, Float32
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Image
from flash_msgs.msg import GoalForceController

from collections import deque
import copy

import numpy as np
from simple_pid import PID


class MovingAverage:
    """
    A simple moving average filter to smooth out the force readings.
    """

    def __init__(self, size):
        self.window = deque(maxlen=size)


    def filter(self, value):
        self.window.append(value)
        return sum(self.window) / len(self.window)


class ForceControl(Node):

    def __init__(self):
        """
        This node controls the force of the gripper.

        :return: None
        """
        
        super().__init__("force_control")

        # define PID controller 
        # (12, 1, 2) seems to work well
        self.pid = PID(0.01, 0, 0.001, setpoint=0.0, sample_time=None, starting_output=0.0, output_limits=(-80, 80))

        # store current position, force and goal force
        self.current_position = None
        self.goal_position = None
        self.current_force = 0
        self.goal_force = None

        # moving average filter for force
        self.filt_left = MovingAverage(size=10)
        self.filt_right = MovingAverage(size=10)

        # create publisher to set goal position of gripper
        self.goal_position_publisher = self.create_publisher(Int16, "set_actuated_umi_motor_position", 1)

        # create subscriber to get current position of gripper
        self.current_position_subscriber = self.create_subscription(JointState, "actuated_umi_motor_state", self.get_current_position, 1)

        # create subscriber to get current force of gripper without callback
        current_force_subscriber_qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.current_force_subscriber = self.create_subscription(Image, "nta_left", self.get_current_force_left, current_force_subscriber_qos_profile)
        self.current_force_subscriber  # prevent unused variable warning

        self.current_force_subscriber = self.create_subscription(Image, "nta_right", self.get_current_force_right, current_force_subscriber_qos_profile)
        self.current_force_subscriber  # prevent unused variable warning

        # create publisher for filtered force
        ma_force_publisher_qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.filtered_force_left_publisher = self.create_publisher(Float32, "nta_left_filtered", ma_force_publisher_qos_profile)
        self.filtered_force_right_publisher = self.create_publisher(Float32, "nta_right_filtered", ma_force_publisher_qos_profile)

        # create subscriber to set goal force of gripper
        goal_force_subscriber_qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.goal_force_subscriber = self.create_subscription(GoalForceController, "set_actuated_umi_goal_force", self.set_goal_force, goal_force_subscriber_qos_profile)
        self.goal_force_subscriber  # prevent unused variable warning

        timer_period = 1.0 / 100
        self.timer = self.create_timer(timer_period, self.force_control)

        
    def set_goal_position(self, position):
        """
        Set the goal position of the gripper.

        :param position: goal position of gripper
        :return: None
        """

        msg = Int16()
        msg.data = int(position)
        self.goal_position_publisher.publish(msg)


    def set_goal_force(self, msg):
        """
        Set the goal normal force of the gripper.

        :param msg: wrench message containing the goal normal force
        :return: None
        """

        #self.goal_force = 0.0
        self.goal_force = msg.goal_force
        self.goal_position = msg.goal_position


    def get_current_position(self, msg):
        """
        Get the current position of the gripper.
        
        :param msg: message containing the current position
        :return: None
        """

        self.current_position = msg.position[0]


    def get_current_force_left(self, msg):
        """
        Get the current normal force of the left gripper.
        Calculate the force rate and apply a moving average filter to it.

        :return: None
        """

        # store current force
        raw_data = np.frombuffer(msg.data, dtype=np.uint16)
        self.current_force_left = np.sum(raw_data)
        self.current_force_left = self.filt_left.filter(self.current_force_left)
        
        # publish filtered force
        filtered_force_left = Float32()
        filtered_force_left.data = self.current_force_left
        self.filtered_force_left_publisher.publish(filtered_force_left)


    def get_current_force_right(self, msg):
        """
        Get the current normal force of the right gripper.
        Calculate the force rate and apply a moving average filter to it.

        :return: None
        """

        # store current force
        raw_data = np.frombuffer(msg.data, dtype=np.uint16)
        self.current_force_right = np.sum(raw_data)
        self.current_force_right = self.filt_right.filter(self.current_force_right)
        
        # publish filtered force
        filtered_force_right = Float32()
        filtered_force_right.data = self.current_force_right
        self.filtered_force_right_publisher.publish(filtered_force_right)


    def force_control(self):
        """
        Control the force of the gripper via force feedback.

        :return: None
        """

        if self.goal_force is not None and self.current_force_left is not None and self.current_force_right is not None and self.current_position is not None and self.goal_position is not None:

            #if self.goal_force <= -0.5 and self.current_force <= -0.5:
            if self.goal_force != 0.0:

                # calculate mean force
                self.current_force = (self.current_force_left + self.current_force_right) / 2.0
                
                # calculate force error
                force_error = -1 * (self.goal_force - self.current_force)
                print("Force error: ", force_error)

                # compute position adjustment
                position_adjustment = self.pid(-1 * force_error)

                # update goal position
                self.set_goal_position(self.current_position + position_adjustment)

            else:
                
                # update goal position
                self.set_goal_position(self.goal_position)


def main(args=None):
    """
    ROS node for the force control of the gripper.

    :param args: arguments for the ROS node
    :return: None
    """

    try:
        
        print("Force Control node is running... Press <ctrl> <c> to stop. \n")

        rclpy.init(args=args)

        force_control = ForceControl()

        rclpy.spin(force_control)

    finally:

        # Destroy the node explicitly
        # (optional - otherwise it will be done automatically
        # when the garbage collector destroys the node object)
        force_control.destroy_node()
        rclpy.shutdown()
    

if __name__ == "__main__":

    main()