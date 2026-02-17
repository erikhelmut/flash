import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    actuated_umi_share_dir = get_package_share_directory("actuated_umi")
    actuated_umi_config_file = os.path.join(actuated_umi_share_dir, "config", "inference_qos.yaml")

    neuromorphic_tactile_array_share_dir = get_package_share_directory("neuromorphic_tactile_array")
    neuromorphic_tactile_array_config_file = os.path.join(neuromorphic_tactile_array_share_dir, "config", "inference_qos.yaml")

    prophesee_genx320_share_dir = get_package_share_directory("prophesee_genx320")
    prophesee_genx320_config_file = os.path.join(prophesee_genx320_share_dir, "config", "inference_qos.yaml")

    realsense_d405_share_dir = get_package_share_directory("realsense_d405")
    realsense_d405_config_file = os.path.join(realsense_d405_share_dir, "config", "inference_qos.yaml")

    optitrack_share_dir = get_package_share_directory("optitrack")
    optitrack_config_file = os.path.join(optitrack_share_dir, "config", "inference_qos.yaml")

    return LaunchDescription([
        Node(
            package="actuated_umi",
            executable="actuated_umi_node",
            name="actuated_umi_node",
            parameters=[actuated_umi_config_file],
            output="screen"
        ),
        Node(
            package="neuromorphic_tactile_array",
            executable="nta_node",
            name="nta_node",
            parameters=[
                {neuromorphic_tactile_array_config_file},
                {"swap_sensors": True}
                ],
            output="screen"
        ),
        # Node(
        #     package="prophesee_genx320",
        #     executable="prophesee_genx320_node",
        #     name="prophesee_genx320_node",
        #     parameters=[prophesee_genx320_config_file],
        #     output="screen"
        # ),
        Node(
            package="realsense_d405",
            executable="realsense_d405_node",
            name="realsense_d405_node",
            parameters=[realsense_d405_config_file],
            output="screen"
        ),
        Node(
            package="realsense_d405",
            executable="detect_aruco_node",
            name="detect_aruco_node",
            parameters=[realsense_d405_config_file],
            output="screen"
        ),
        Node(
            package="optitrack",
            executable="optitrack_node",
            name="optitrack_node",
            parameters=[optitrack_config_file],
            output="screen"
        ),
    ])