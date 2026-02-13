from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from std_msgs.msg import Int16
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import TransformStamped
from aruco_msgs.msg import ArUcoMarkerStamped, ArUcoDistStamped
from flash_msgs.msg import GoalForceController

import time

import cv2
import numpy as np
import torch
from safetensors.torch import load_file

import copy
from collections import deque
import yaml

from franka_panda.panda_real import PandaReal

from scipy.spatial.transform import Rotation

from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.diffusion.processor_diffusion import make_diffusion_pre_post_processors
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def rotation_to_feature(rot: Rotation) -> np.ndarray:
    """
    Extract rotation features according to this paper:
    https://openaccess.thecvf.com/content_CVPR_2019/html/Zhou_On_the_Continuity_of_Rotation_Representations_in_Neural_Networks_CVPR_2019_paper.html
    Unlike the paper though, we include the second and third column instead of the first and second, as it helps
    us to ensure that the sensor only receives downwards pointing target orientations.
    :param rot: Rotation to compute features representation for.
    :return: 6D feature representation of the given rotation.
    """
    matrix = rot.inv().as_matrix()
    return matrix.reshape((*matrix.shape[:-2], -1))[..., 3:]


def feature_to_rotation(feature: np.ndarray) -> Rotation:
    z_axis_unnorm = feature[..., 3:]
    z_norm = np.linalg.norm(z_axis_unnorm, axis=-1, keepdims=True)
    assert np.all(z_norm > 0)
    z_axis = z_axis_unnorm / z_norm
    y_axis_unnorm = (
        feature[..., :3]
        - (z_axis * feature[..., :3]).sum(-1, keepdims=True) * z_axis
    )
    y_norm = np.linalg.norm(y_axis_unnorm, axis=-1, keepdims=True)
    assert np.all(y_norm > 0)
    y_axis = y_axis_unnorm / y_norm
    x_axis = np.cross(y_axis, z_axis)
    return Rotation.from_matrix(np.stack([x_axis, y_axis, z_axis], axis=-1))


class MovingAverage:
    """
    A simple moving average filter to smooth out the force readings.
    """

    def __init__(self, size):
        self.window = deque(maxlen=size)


    def filter(self, value):
        self.window.append(value)
        return sum(self.window) / len(self.window)


class IMITATOR:

    def __init__(self):
        """
        This class is used to predict the gripper behavior using the diffusion policy. It loads the model and sets the device to GPU if available.

        :return: None
        """
        
        # specify device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # 1. setup the config (match the settings used during training)
        pretrained_policy_path = Path("/home/erik/flash/src/imitator/outputs/cupstacking/checkpoints/last/pretrained_model")
        config = PreTrainedConfig.from_pretrained(pretrained_policy_path, local_files_only=True)

        self.pre_processor, self.post_processor = make_diffusion_pre_post_processors(config)
        pre_stats = load_file(pretrained_policy_path /"policy_preprocessor_step_3_normalizer_processor.safetensors")
        post_stats = load_file(pretrained_policy_path / "policy_postprocessor_step_0_unnormalizer_processor.safetensors")
        self.pre_processor.steps[3].load_state_dict(pre_stats)
        self.post_processor.steps[0].load_state_dict(post_stats)

        # 2. instantiate the policy
        self.policy = DiffusionPolicy(config)

        # 3. load the weights
        state_dict = load_file(str(pretrained_policy_path / "model.safetensors"), device="cpu")
        self.policy.load_state_dict(state_dict)

        # 4. set the policy to evaluation mode and send it to the specified device
        self.policy.eval()
        self.policy.to(self.device)


    def make_prediction(self, nta_left, nta_left_sum, nta_right, nta_right_sum, gripper_width, rs_d405_img, ee_pos, ee_ori):
        """
        Make prediction using the diffusion policy.

        :param nta_left: current tactile image of the left NTA sensor
        :param nta_left_sum: current summed force of the left NTA sensor
        :param nta_right: current tactile image of the right NTA sensor
        :param nta_right_sum: current summed force of the right NTA sensor
        :param gripper_width: current gripper width
        :param rs_d405_img: current color image of the D405 RealSense camera
        :param ee_pos: current end-effector position
        :param ee_ori: current end-effector orientation
        :return: goal_nta_left, goal_nta_right, goal_distance, goal_ee_pos, goal_ee_ori: predicted goal tactile images for the left and right NTA sensors, distance for the gripper, end-effector position and orientation
        """

        # extract the batch data
        raw_state = [nta_left_sum, nta_right_sum, gripper_width]
        raw_state.extend(ee_pos.astype(np.float32).tolist())

        # convert the end-effector orientation to a 6D feature representation
        ee_ori = Rotation.from_quat(ee_ori)
        ee_ori = rotation_to_feature(ee_ori)
        raw_state.extend(ee_ori.astype(np.float32).tolist())

        # convert the state to a tensor and add batch dimension
        state = torch.tensor(raw_state).unsqueeze(0).float()
        
        image = torch.from_numpy(rs_d405_img).float()
        image = image.permute(0, 3, 1, 2)

        nta_left = torch.from_numpy(nta_left).float()
        nta_left = nta_left.permute(0, 3, 1, 2)
        nta_right = torch.from_numpy(nta_right).float()
        nta_right = nta_right.permute(0, 3, 1, 2)

        # send data tensors from CPU to GPU
        state = state.to(self.device, non_blocking=True)
        image = image.to(self.device, non_blocking=True)
        nta_left = nta_left.to(self.device, non_blocking=True)
        nta_right = nta_right.to(self.device, non_blocking=True)

        # create the policy input dictionary
        observation = {
            "observation.state": state,
            "observation.image.realsense": image,
            "observation.image.nta_left": nta_left,
            "observation.image.nta_right": nta_right,
        }

        observation = self.pre_processor(observation)

        # predict the next action with respect to the current observation
        # the policy internaly handles the queue of observations and actions
        with torch.inference_mode():
            policy_action = self.policy.select_action(observation)

        policy_action = self.post_processor(policy_action)

        goal_nta_left = policy_action.squeeze(0)[0].to("cpu").numpy()
        goal_nta_right = policy_action.squeeze(0)[1].to("cpu").numpy()
        goal_distance = policy_action.squeeze(0)[2].to("cpu").numpy()

        goal_x_pos = policy_action.squeeze(0)[3].to("cpu").numpy()
        goal_y_pos = policy_action.squeeze(0)[4].to("cpu").numpy()
        goal_z_pos = policy_action.squeeze(0)[5].to("cpu").numpy()
        goal_ee_pos = np.array([goal_x_pos, goal_y_pos, goal_z_pos])

        goal_f0_rot = policy_action.squeeze(0)[6].to("cpu").numpy()
        goal_f1_rot = policy_action.squeeze(0)[7].to("cpu").numpy()
        goal_f2_rot = policy_action.squeeze(0)[8].to("cpu").numpy()
        goal_f3_rot = policy_action.squeeze(0)[9].to("cpu").numpy()
        goal_f4_rot = policy_action.squeeze(0)[10].to("cpu").numpy()
        goal_f5_rot = policy_action.squeeze(0)[11].to("cpu").numpy()
        goal_ee_ori = np.array([goal_f0_rot, goal_f1_rot, goal_f2_rot, goal_f3_rot, goal_f4_rot, goal_f5_rot])

        # convert the goal end-effector orientation to a quaternion
        goal_ee_ori = feature_to_rotation(goal_ee_ori)
        goal_ee_ori = goal_ee_ori.as_quat()

        return goal_nta_left, goal_nta_right, goal_distance, goal_ee_pos, goal_ee_ori


class IMITATORNode(Node):

    def __init__(self):
        """
        This node subscribes to the force prediction of the FEATS model and to the color image of the D405 RealSense camera and makes gripper behavior predictions using the diffusion policy. It publishes the predicted gripper behavior in form of goal force and goal gripper width.

        :return: None
        """
        
        super().__init__("imitator_node")

        # initialize diffusion policy
        self.imitator = IMITATOR()

        # initialize the PandaReal instance
        with open("/home/erik/flash/src/franka_panda/config/panda_config.yaml", "r") as f:
            config = yaml.safe_load(f)
        self.panda = PandaReal(config)

        # load calibration parameters for gripper
        self.m, self.c = np.load("/home/erik/flash/src/actuated_umi/calibration/20260204-144401.npy")

        # store current force and image in class variables
        self.nta_left = None
        self.nta_left_sum = None
        self.nta_right = None
        self.nta_right_sum = None
        self.gripper_width = None
        self.gripper_width_aruco = None
        self.rs_d405_img = None

        # store current end-effector position and orientation
        self.ee_pos = None
        self.ee_ori = None

        # initialize variables for initial movement
        self.initial_movement_done = False
        self.delta_h = 0.005
        self.total_h = 0.0

        # moving average filter for force
        self.filt = MovingAverage(size=10)

        # store current and goal forces in list for saving rollout data
        self.store = False
        if self.store:
            self.rollout_current_force = []
            self.rollout_goal_force = []
            self.rollout_goal_force_filtered = []
    
        # create subscriber to get current force prediction of the FEATS model
        self.declare_parameter("nta.qos.reliability", "reliable")
        self.declare_parameter("nta.qos.history", "keep_last")
        self.declare_parameter("nta.qos.depth", 10)

        nta_subscriber_qos_profile = self.get_qos_profile("nta.qos")

        self.nta_left_subscriber = self.create_subscription(Image, "nta_left", self.get_current_nta_left, nta_subscriber_qos_profile)
        self.nta_left_subscriber  # prevent unused variable warning

        self.nta_right_subscriber = self.create_subscription(Image, "nta_right", self.get_current_nta_right, nta_subscriber_qos_profile)
        self.nta_right_subscriber  # prevent unused variable warning


        # create subscriber to get current gripper width of the actuated umi gripper
        self.declare_parameter("actuated_umi.qos.reliability", "reliable")
        self.declare_parameter("actuated_umi.qos.history", "keep_last")
        self.declare_parameter("actuated_umi.qos.depth", 10)

        actuated_umi_subscriber_qos_profile = self.get_qos_profile("actuated_umi.qos")

        self.actuated_umi_state_subscriber = self.create_subscription(JointState, "actuated_umi_motor_state", self.get_current_width, actuated_umi_subscriber_qos_profile)
        self.actuated_umi_state_subscriber  # prevent unused variable warning


        # create subscriber to get the current color image of the D405 RealSense camera
        self.declare_parameter("rs_d405.qos.reliability", "reliable")
        self.declare_parameter("rs_d405.qos.history", "keep_last")
        self.declare_parameter("rs_d405.qos.depth", 10)

        rs_d405_qos_profile = self.get_qos_profile("rs_d405.qos")

        self.rs_d405_color_subscriber = self.create_subscription(Image, "realsense_d405_color_image", self.get_current_image, rs_d405_qos_profile)
        self.rs_d405_color_subscriber  # prevent unused variable warning

        self.rs_d405_aruco_subscriber = self.create_subscription(ArUcoDistStamped, "realsense_d405_aruco_distance", self.get_current_width_aruco, rs_d405_qos_profile)
        self.rs_d405_aruco_subscriber  # prevent unused variable warning


        # create subscriber to get the current pose of the end-effector from optitrack
        self.declare_parameter("optitrack.qos.reliability", "reliable")
        self.declare_parameter("optitrack.qos.history", "keep_last")
        self.declare_parameter("optitrack.qos.depth", 10)

        optitrack_qos_profile = self.get_qos_profile("optitrack.qos")

        self.optitrack_subscriber = self.create_subscription(TransformStamped, "optitrack_ee_state", self.get_ee_state, optitrack_qos_profile)
        self.optitrack_subscriber  # prevent unused variable warning


        # create publisher to set goal force and gripper width of the actuated umi gripper
        self.imitator_publisher = self.create_publisher(GoalForceController, "set_actuated_umi_goal_force", 1)


        timer_period = 1.0 / 7#10  # 20 or 25 Hz
        self.timer = self.create_timer(timer_period, self.pub_prediction)

    
    def get_qos_profile(self, base_param_name):
        """
        Helper function to retrieve and validate QoS settings.
        
        :param base_param_name: base name of the QoS parameters
        :return: QoSProfile object
        """
        
        # get the parameter values
        reliability_param = self.get_parameter(f"{base_param_name}.reliability").value
        history_param = self.get_parameter(f"{base_param_name}.history").value
        depth_param = self.get_parameter(f"{base_param_name}.depth").value

        # normalize to lowercase to avoid mismatches
        reliability_param = str(reliability_param).lower()
        history_param = str(history_param).lower()

        self.get_logger().info(f"QoS settings: reliability={reliability_param}, history={history_param}, depth={depth_param}")

        # convert to QoS enums with fallback
        if reliability_param == "best_effort":
            reliability = QoSReliabilityPolicy.BEST_EFFORT
        elif reliability_param == "reliable":
            reliability = QoSReliabilityPolicy.RELIABLE
        else:
            self.get_logger().warn(f"Unknown reliability: {reliability_param}, defaulting to RELIABLE")
            reliability = QoSReliabilityPolicy.RELIABLE

        if history_param == "keep_last":
            history = QoSHistoryPolicy.KEEP_LAST
        elif history_param == "keep_all":
            history = QoSHistoryPolicy.KEEP_ALL
        else:
            self.get_logger().warn(f"Unknown history: {history_param}, defaulting to KEEP_LAST")
            history = QoSHistoryPolicy.KEEP_LAST

        # depth should be an int, just check type or cast
        try:
            depth = int(depth_param)
        except (ValueError, TypeError):
            self.get_logger().warn(f"Invalid depth: {depth_param}, defaulting to 10")
            depth = 10

        # return the QoSProfile
        return QoSProfile(
            reliability=reliability,
            history=history,
            depth=depth
        )
    

    def get_current_nta_left(self, msg):
        """
        Get the current tactile data for the left NTA sensor.
        
        :param msg: message containing the tactile data for the left sensor
        :return: None
        """

        raw_data = np.frombuffer(msg.data, dtype=np.uint16)
        image_16bit = raw_data.reshape((msg.height, msg.width))

        self.nta_left_sum = np.sum(image_16bit)

        clim=(0, 1000)
        # scale
        scaled = (image_16bit - clim[0]) * (255 / (clim[1] - clim[0]))
        scaled = np.clip(scaled, 0, 255)
        # resize
        resized = np.array([
            cv2.resize(scaled, (96, 96), interpolation=cv2.INTER_AREA)
        ], dtype=np.uint8)
        # stack to (96, 96, 3)
        self.nta_left = np.stack([resized, resized, resized], axis=-1)


    def get_current_nta_right(self, msg):
        """
        Get the current tactile data for the right NTA sensor.
        
        :param msg: message containing the tactile data for the right sensor
        :return: None
        """

        raw_data = np.frombuffer(msg.data, dtype=np.uint16)
        image_16bit = raw_data.reshape((msg.height, msg.width))

        self.nta_right_sum = np.sum(image_16bit)

        clim=(0, 1000)
        # scale
        scaled = (image_16bit - clim[0]) * (255 / (clim[1] - clim[0]))
        scaled = np.clip(scaled, 0, 255)
        # resize
        resized = np.array([
            cv2.resize(scaled, (96, 96), interpolation=cv2.INTER_AREA)
        ], dtype=np.uint8)
        # stack to (96, 96, 3)
        self.nta_right = np.stack([resized, resized, resized], axis=-1)


    def get_current_width(self, msg):
        """
        Get the current position of the gripper.
        
        :param msg: message containing the current gripper width
        :return: None
        """

        self.gripper_width = (msg.position[0] - self.c) / self.m

    
    def get_current_width_aruco(self, msg):
        """
        Get the current gripper width based on the distance between two ArUco markers.

        :param msg: message containing the distance between two ArUco markers
        :return: None
        """

        self.gripper_width_aruco = msg.distance


    def get_current_image(self, msg):
        """
        Get the current color image of the D405 RealSense camera.

        :param msg: message containing the current color image
        :return: None
        """
        
        # convert ros image message to numpy array
        raw_img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)

         # convert raw_img from bgr to rgb
        raw_img = np.array(cv2.cvtColor(raw_img, cv2.COLOR_RGB2BGR), dtype=np.uint8)

        # resize images to a smaller size
        self.rs_d405_img = np.array([cv2.resize(raw_img, (96, 96), interpolation=cv2.INTER_AREA)], dtype=np.uint8)


    def get_ee_state(self, msg):
        """
        Get the current end-effector position and orientation from optitrack.

        :param msg: message containing the current end-effector position and orientation
        :return: None
        """

        if msg.child_frame_id == "panda_ot_ee":
            # extract panda end-effector and OptiTrack rigid body positions and orientations
            self.ot_ee_pos = np.array([msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z])
            self.ot_ee_ori = np.array([msg.transform.rotation.x, msg.transform.rotation.y, msg.transform.rotation.z, msg.transform.rotation.w])

        
    def pub_prediction(self):
        """
        Publish the predicted gripper behavior of the diffusion policy.

        :return: None
        """

        if self.nta_left is not None and self.nta_right is not None and self.gripper_width is not None and self.rs_d405_img is not None:

            # get current state of the panda
            self.ee_pos = self.panda.end_effector_position
            self.ee_ori = self.panda.end_effector_orientation

            # make prediction using the diffusion policy
            goal_nta_left, goal_nta_right, goal_distance, goal_ee_pos, goal_ee_ori = self.imitator.make_prediction(self.nta_right, self.nta_right_sum, self.nta_left, self.nta_left_sum, self.gripper_width, self.rs_d405_img, self.ee_pos, self.ee_ori)

            # move the robot
            # if self.initial_movement_done is False:
            #     # check if force is below -1 at least
            #     if self.feats_fz < -0.7 and goal_force < -0.7:
            #         self.total_h += self.delta_h
            #         #self.panda.move_abs(goal_pos=self.ee_pos + np.array([-1 * self.delta_h, 0.0, 0.0]), rel_vel=0.01, goal_ori=self.ee_ori, asynch=True) # grape task
            #         self.panda.move_abs(goal_pos=self.ee_pos + np.array([0.0, 0.0, self.delta_h]), rel_vel=0.01, goal_ori=self.ee_ori, asynch=True) # planting task
            #         if self.total_h >= 0.1:
            #             self.initial_movement_done = True
            # else:
            self.panda.move_abs(goal_pos=goal_ee_pos, rel_vel=0.01, goal_ori=goal_ee_ori, asynch=True) # 0.02

            msg = GoalForceController()
            #msg.goal_force = float(goal_force)
            # goal_force_filtered = self.filt.filter(goal_nta_left)
            # msg.goal_force = float(goal_force_filtered)
            msg.goal_force = float(0.0) # for testing
            msg.goal_position = int(self.m * goal_distance + self.c - 200)
            self.imitator_publisher.publish(msg)

            # # add current and goal forces to the rollout data
            # if self.store:
            #     self.rollout_current_force.append(self.feats_fz)
            #     self.rollout_goal_force.append(goal_force)
            #     self.rollout_goal_force_filtered.append(goal_force_filtered)


    def destroy_node(self):
        """
        Save the current rollout data to a file, when object is deleted.
        
        :return: None
        """
        
        if self.rollout_current_force and self.rollout_goal_force and self.rollout_goal_force_filtered and self.store:
            
            current_force = np.array(self.rollout_current_force)
            goal_force = np.array(self.rollout_goal_force)
            goal_force_filtered = np.array(self.rollout_goal_force_filtered)
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            run = "planting_task/force_dist_control/initial_pos_v4/fc"

            filename = f"/home/erik/flash/src/imitator/rollouts/{run}_{timestamp}.npz"

            np.savez(filename,
                 current_force=current_force,
                 goal_force=goal_force,
                 goal_force_filtered=goal_force_filtered)

        super().destroy_node()


def main(args=None):
    """
    ROS node for making live predictions using the diffusion model.

    :param args: arguments for the ROS node
    :return: None
    """

    try:

        print("""
        ██╗███╗   ███╗██╗████████╗ █████╗ ████████╗ ██████╗ ██████╗ 
        ██║████╗ ████║██║╚══██╔══╝██╔══██╗╚══██╔══╝██╔═══██╗██╔══██╗
        ██║██╔████╔██║██║   ██║   ███████║   ██║   ██║   ██║███████║
        ██║██║╚██╔╝██║██║   ██║   ██╔══██║   ██║   ██║   ██║██╔═██╔╝ 
        ██║██║ ╚═╝ ██║██║   ██║   ██║  ██║   ██║   ╚██████╔╝██║  ██╗   
        ╚═╝╚═╝     ╚═╝╚═╝   ╚═╝   ╚═╝  ╚═╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝   
        """)

        print("\nIMITATOR Node is running... Press <ctrl> <c> to stop. \nPredicted gripper width and goal force are being published on topic /set_actuated_umi_goal_force. \n")

        rclpy.init(args=args)

        imitator_node = IMITATORNode()

        rclpy.spin(imitator_node)

    finally:

        # Destroy the node explicitly
        # (optional - otherwise it will be done automatically
        # when the garbage collector destroys the node object)
        imitator_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":

    main()