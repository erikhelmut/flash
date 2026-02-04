from pathlib import Path
import argparse

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from flash_msgs.msg import GoalForceController
from std_msgs.msg import Int16

import numpy as np
import torch
import matplotlib.pyplot as plt

import time
import yaml
from franka_panda.panda_real import PandaReal

from scipy.spatial.transform import Rotation

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


class ReplayNode(Node):

    def __init__(self):
        """
        This node is responsible for replaying the robot movements
        based on the recorded dataset.

        :return: None
        """
        
        super().__init__("replay_node")

        # initialize the PandaReal instance
        with open("/home/erik/flash/src/franka_panda/config/panda_config.yaml", "r") as f:
            config = yaml.safe_load(f)
        self.panda = PandaReal(config)

        # load calibration parameters for gripper
        self.m, self.c = np.load("/home/erik/flash/src/actuated_umi/calibration/20260204-144401.npy")

        # create publisher to set goal force and gripper width of the actuated umi gripper
        #self.imitator_publisher = self.create_publisher(GoalForceController, "set_actuated_umi_goal_force", 1)
        self.imitator_publisher = self.create_publisher(Int16, "set_actuated_umi_motor_position", 1)


        # configure LeRobotDataset for one or multiple episodes
        self.dataset = LeRobotDataset(
            repo_id=Path("/home/erik/flash/cupstacking_v2"),
            episodes=[1], 
        )

        # use standard PyTorch DataLoader to load the dataset
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            num_workers=8,
            batch_size=1,
            shuffle=False,
        )

        self.i = 80

        # get length of the dataset
        self.dataset_length = len(self.dataset)

        timer_period = 1.0 / 25  # 25 Hz
        self.timer = self.create_timer(timer_period, self.pub_state)


    def pub_state(self):

        try:

            batch = self.dataset[self.i]
            print(batch["observation.state"][2])

            # extract the batch data
            state = batch["observation.state"]
            #image = batch["observation.image"]
            #action = batch["action"]

            goal_ee_pos = np.array(state.squeeze(0)[3:6].to("cpu").numpy())
            goal_ee_ori = np.array(state.squeeze(0)[6:12].to("cpu").numpy())
            goal_ee_ori = feature_to_rotation(goal_ee_ori)
            goal_ee_ori = goal_ee_ori.as_quat()

            # remove 10cm from x position to account for table height
            goal_ee_pos[0] -= 0.1

            # move the robot arm
            self.panda.move_abs(
                goal_pos=goal_ee_pos,
                goal_ori=goal_ee_ori,
                rel_vel=0.02,
                asynch=False
            )

            # msg = GoalForceController()
            # #msg.goal_force = float(self.filt.filter(goal_force))
            # msg.goal_force = float(0)
            # msg.goal_position = int(self.m * state[1] + self.c) #-40)
            # self.imitator_publisher.publish(msg)

            msg = Int16()
            msg.data = int(self.m * state[2].to("cpu").numpy() + self.c)
            self.imitator_publisher.publish(msg)

            self.i += 1

            if self.i >= self.dataset_length:
                self.i = 100000
        
        except Exception as e:
            self.timer.cancel()
            # stop code
            exit(0)
    

def main(args=None):

    try:

        rclpy.init(args=args)

        replay_node = ReplayNode()

        rclpy.spin(replay_node)

    finally:

        replay_node.destroy_node()
        rclpy.shutdown()



if __name__ == "__main__":

    main()