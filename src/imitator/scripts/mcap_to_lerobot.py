import os
import sys; sys.path.append("/home/erik/flash/src/imitator/lerobot")
import argparse

from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py

import copy
import numpy as np
from matplotlib import cm
import cv2 
from pathlib import Path
import shutil

from scipy.spatial.transform import Rotation
from transformation import Transformation

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import DEFAULT_FEATURES



# --- ROSBAG READER UTILS ---


class RosbagReader():

    def __init__(self, input_bag: str):
        """
        Rosbag reader class to read messages from a bag file.

        :param input_bag: path to the bag file
        :return: None
        """
        
        self.reader = rosbag2_py.SequentialReader()
        self.reader.open(
            rosbag2_py.StorageOptions(uri=input_bag, storage_id="mcap"),
            rosbag2_py.ConverterOptions(
                input_serialization_format="cdr", output_serialization_format="cdr"
            ),
        )

        self.topic_types = self.reader.get_all_topics_and_types()


    def __del__(self):
        """
        Destructor to close the reader object.
        
        :return: None
        """

        del self.reader


    def typename(self, topic_name):
        """
        Get the message type of a topic.

        :param topic_name: name of the topic
        :return: message type of the topic
        :raises ValueError: if the topic is not in the bag
        """
        
        for topic_type in self.topic_types:
            if topic_type.name == topic_name:
                return topic_type.type
        
        raise ValueError(f"topic {topic_name} not in bag")


    def read_messages(self):
        """
        Generator function to read messages from the bag file.

        :return: topic name, message, timestamp
        """

        while self.reader.has_next():
            topic, data, timestamp = self.reader.read_next()
            msg_type = get_message(self.typename(topic))
            msg = deserialize_message(data, msg_type)
            
            yield topic, msg, timestamp


def convert_timestamp_to_float(timestamp):
    """
    Helper function to convert a timestamp to a float.

    :param timestamp: timestamp to convert
    :return: converted timestamp
    """

    return timestamp.sec + timestamp.nanosec * 1e-9


def time_filter_list(ref_timestamps, target_timestamps, target_list, target_list_name, delta_t=0.4):
    """
    Filter a list based on the closest timestamps in another list.

    :param ref_timestamps: reference timestamps
    :param target_timestamps: target timestamps
    :param target_list: target list to filter
    :param target_list_name: name of the target list
    :param delta_t: maximum allowed difference between timestamps to consider them aligned in seconds
    :return: filtered list
    """

    filtered_list = []

    warn = False
    
    for ts in ref_timestamps:
        # find the closest timestamp in target_timestamps
        closest_ts = min(target_timestamps, key=lambda x: abs(x - ts))
        index = target_timestamps.index(closest_ts)

        # append the corresponding value to the filtered list
        if abs(closest_ts - ts) < delta_t:
            filtered_list.append(target_list[index])
        else:
            filtered_list.append(None)
            warn = True
    
    if warn:
        print(f"Warning: {target_list_name} timestamps are not aligned with gs_mini timestamps. Some values are set to None.")

    return filtered_list


def fill_none_values(seq):
    """
    Fill None values in a list with the first and last non-None values.
    Interpolate None values in the middle.
    
    :param seq: list to fill
    :return: filled list
    """

    # replace None values at the start with the first non-None value
    for i in range(len(seq)):
        if seq[i] is not None:
            seq[:i] = [seq[i]] * i
            break

    # replace None values at the end with the last non-None value
    for i in range(len(seq) - 1, -1, -1):
        if seq[i] is not None:
            seq[i+1:] = [seq[i]] * (len(seq) - i - 1)
            break

    # interpolate None values in the middle
    i = 0
    while i < len(seq):
        if seq[i] is None:
            start = i - 1
            # find the next non-None value
            j = i
            while j < len(seq) and seq[j] is None:
                j += 1
            if j < len(seq):
                # interpolate between seq[start] and seq[j]
                step = (seq[j] - seq[start]) / (j - start)
                for k in range(1, j - start):
                    seq[start + k] = seq[start] + step * k
            i = j
        else:
            i += 1

    return seq


def find_last_force_index(forces):
    """
    Find the index of the last force value in the forces list that is greater than a threshold.

    :param forces: list of force values
    :return: index of the last force value greater than the threshold, or -1 if not found
    """

    threshold = -0.5

    # iterate backwards through the list
    j = 0
    for i in range(len(forces)-1, -1, -1):
        if forces[i] <= threshold:
            j = copy.copy(i)
            break
    return j


# --- LEROBOT DATASET CONVERSION UTILITIES ---


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


def main():
    """
    MCAP to LeRobotDataset converter script.

    :return: None
    """

    # parse arguments
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="Input folder containing mcap files")
    parser.add_argument("--output", help="LeRobot dataset output directory")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--subsample", type=int, default=4)
    args = parser.parse_args()

    # load end-effector calibration
    ee_offset = np.load("/home/erik/flash/src/optitrack/calibration/ee_offset.npy")
    ee_rotation = np.load("/home/erik/flash/src/optitrack/calibration/ee_rotation.npy")

    # -----------------------------
    # --- Setup lerobot dataset ---
    # -----------------------------

    output_dataset_dir = Path(args.output)
    repo_id = output_dataset_dir.name  # use directory name as repo_id
    use_videos = False  # whether to store images as videos or individual images

    # create/clean output directory
    if output_dataset_dir.exists():
        
        print(f"Warning: Output directory {output_dataset_dir} already exists.")
        user_input = input(f"Do you want to remove it and create a new dataset? (yes/no): ")
        
        if user_input.lower() == "yes":
            shutil.rmtree(output_dataset_dir)
            print(f"Removed {output_dataset_dir}.")
        else:
            print("Exiting. Please remove the directory manually or choose a different one.")
            return None
    
    output_dataset_dir.parent.mkdir(parents=True, exist_ok=True) # ensure parent exists

    # LeRobotDataset will add 'index', 'episode_index', 'frame_index', 'timestamp',
    # 'task_index', 'is_first', 'is_last', etc. if DEFAULT_FEATURES are included
    # for images, specify 'image' or 'video' as dtype
    # the 'task' key in `add_frame` will be handled and converted to 'task_index'
    # you don't need 'task' in the initial `features` dict for `create`
    features = {
        **DEFAULT_FEATURES,  # recommended to include
        "observation.state": {"shape": (12,), "dtype": "float32", "names": ["nta_left", "nta_right", "aruco_dist", "optitrack_trans_x", "optitrack_trans_y", "optitrack_trans_z", "optitrack_rot_feat_0", "optitrack_rot_f1", "optitrack_rot_f2", "optitrack_rot_f3", "optitrack_rot_f4", "optitrack_rot_f5"]},
        "observation.image.realsense": {
            "shape": (96, 96, 3),
            "dtype": "video" if use_videos else "image",
            "names": ["height", "width", "channel"]
        },
        "observation.image.nta_left": {
            "shape": (96, 96, 3),
            "dtype": "video" if use_videos else "image",
            "names": ["height", "width", "channel"]
        },
        "observation.image.nta_right": {
            "shape": (96, 96, 3),
            "dtype": "video" if use_videos else "image",
            "names": ["height", "width", "channel"]
        },
        "action": {"shape": (12,), "dtype": "float32", "names": ["nta_left", "nta_right", "aruco_dist", "optitrack_trans_x", "optitrack_trans_y", "optitrack_trans_z", "optitrack_rot_f0", "optitrack_rot_f1", "optitrack_rot_f2", "optitrack_rot_f3", "optitrack_rot_f4", "optitrack_rot_f5"]},
    }
    
    # if there is no reward/done in the HDF5 files, remove them from DEFAULT_FEATURES
    if "reward" in features: del features["reward"]
    if "done" in features: del features["done"]
    if "is_terminal" in features: del features["is_terminal"]  # often same as done

    # create LeRobotDataset - the 'root' for `create` is the actual directory for this dataset
    print(f"Creating LeRobotDataset '{repo_id}' at: {output_dataset_dir}")
    dataset = LeRobotDataset.create(
        repo_id=repo_id,  # this is mostly for naming/identification if pushed to Hub
        root=output_dataset_dir,  # the direct path where this dataset will live
        fps=args.fps,
        features=features,
        use_videos=use_videos,
    )
    print("LeRobotDataset object created.")


    # --------------------------
    # --- Process MCAP files ---
    # --------------------------

    # get all mcap files in the input path
    mcap_files = []
    if os.path.isdir(args.input):
        for folder in os.listdir(args.input):
            if os.path.isdir(os.path.join(args.input, folder)):
                # get mcap files in the folder
                file = [f for f in os.listdir(os.path.join(args.input, folder)) if f.endswith(".mcap")]
                mcap_files += [os.path.join(args.input, folder, f) for f in file]
    else:
        # if the input is a file, add it to the list
        mcap_files.append(args.input)
    
    for mcap_file in mcap_files:

        try:

            # create reader object
            rr = RosbagReader(mcap_file)

            nta_left = []; nta_right = []
            nta_left_timestamps = []; nta_right_timestamps = []
            nta_left_shape = None; nta_right_shape = None

            prophesee_events = []
            prophesee_events_timestamps = []
            prophesee_events_shape = None

            realsense_d405_color_img = []; realsense_d405_depth_img = []; realsense_d405_aruco_dist = []
            realsense_d405_color_img_timestamps = []; realsense_d405_depth_img_timestamps = []; realsense_d405_aruco_dist_timestamps = []
            realsense_d405_color_img_shape = None; realsense_d405_depth_img_shape = None

            optitrack_trans_x = []; optitrack_trans_y = []; optitrack_trans_z = []; optitrack_rot_x = []; optitrack_rot_y = []; optitrack_rot_z = []; optitrack_rot_w = []
            optitrack_timestamps = []


            # iterate over messages
            for topic, msg, _ in rr.read_messages():

                if topic == "/nta_left":
                    nta_left.append(msg.data)
                    nta_left_timestamps.append(convert_timestamp_to_float(msg.header.stamp))
                    if nta_left_shape is None:
                        nta_left_shape = (msg.height, msg.width, 1)

                elif topic == "/nta_right":
                    nta_right.append(msg.data)
                    nta_right_timestamps.append(convert_timestamp_to_float(msg.header.stamp))
                    if nta_right_shape is None:
                        nta_right_shape = (msg.height, msg.width, 1)

                elif topic == "/prophesee_events":
                    prophesee_events.append(msg.data)
                    prophesee_events_timestamps.append(convert_timestamp_to_float(msg.header.stamp))
                    if prophesee_events_shape is None:
                        prophesee_events_shape = (msg.height, msg.width, 1)

                elif topic == "/realsense_d405_color_image":
                    realsense_d405_color_img.append(msg.data)
                    realsense_d405_color_img_timestamps.append(convert_timestamp_to_float(msg.header.stamp))
                    if realsense_d405_color_img_shape is None:
                        realsense_d405_color_img_shape = (msg.height, msg.width, 3)

                elif topic == "/realsense_d405_depth_image":
                    realsense_d405_depth_img.append(msg.data)
                    realsense_d405_depth_img_timestamps.append(convert_timestamp_to_float(msg.header.stamp))
                    if realsense_d405_depth_img_shape is None:
                        realsense_d405_depth_img_shape = (msg.height, msg.width)

                elif topic == "/realsense_d405_aruco_distance":
                    realsense_d405_aruco_dist.append(msg.distance)
                    realsense_d405_aruco_dist_timestamps.append(convert_timestamp_to_float(msg.header.stamp))

                elif topic == "/optitrack_ee_state":
                    if msg.child_frame_id == "ot_ee":
                        
                        ot_rb_pos = np.array([msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z])
                        ot_rb_ori = np.array([msg.transform.rotation.x, msg.transform.rotation.y, msg.transform.rotation.z, msg.transform.rotation.w])
                        
                        marker_pose_robot_frame = Transformation.from_pos_quat(ot_rb_pos, ot_rb_ori)

                        ee_pose_robot_frame = marker_pose_robot_frame * Transformation.from_pos_quat(ee_offset, ee_rotation)

                        optitrack_trans_x.append(ee_pose_robot_frame.translation[0])
                        optitrack_trans_y.append(ee_pose_robot_frame.translation[1])
                        optitrack_trans_z.append(ee_pose_robot_frame.translation[2])
                        optitrack_rot_x.append(ee_pose_robot_frame.quaternion[0])
                        optitrack_rot_y.append(ee_pose_robot_frame.quaternion[1])
                        optitrack_rot_z.append(ee_pose_robot_frame.quaternion[2])
                        optitrack_rot_w.append(ee_pose_robot_frame.quaternion[3])
                        optitrack_timestamps.append(convert_timestamp_to_float(msg.header.stamp))

            # close reader object
            del rr

            # remove first and last n points from reference clock timestamps
            # n_first = 50
            # n_last = find_last_force_index(nta_right) + 50
            # if n_last < 0:
            #     n_last = 0
            
            # realsense_d405_color_img = realsense_d405_color_img[n_first:n_last]
            # reference_clock = realsense_d405_timestamps[n_first:n_last]
            reference_clock = realsense_d405_color_img_timestamps[:]

            # filter lists based on timestamps of reference clock timestamps
            nta_left = time_filter_list(reference_clock, nta_left_timestamps, nta_left, "nta_left")
            nta_right = time_filter_list(reference_clock, nta_right_timestamps, nta_right, "nta_right")
            nta_left_timestamps = copy.deepcopy(reference_clock)
            nta_right_timestamps = copy.deepcopy(reference_clock)

            #prophesee_events = time_filter_list(reference_clock, prophesee_events_timestamps, prophesee_events, "prophesee_events")
            #prophesee_events_timestamps = copy.deepcopy(reference_clock)

            # realsense_d405_color_img = time_filter_list(reference_clock, realsense_d405_color_img_timestamps, realsense_d405_color_img, "realsense_d405_color_img")
            # realsense_d405_depth_img = time_filter_list(reference_clock, realsense_d405_depth_img_timestamps, realsense_d405_depth_img, "realsense_d405_depth_img")
            # realsense_d405_aruco_dist = time_filter_list(reference_clock, realsense_d405_aruco_dist_timestamps, realsense_d405_aruco_dist, "realsense_d405_aruco_dist")
            # realsense_d405_timestamps = copy.deepcopy(reference_clock)

            optitrack_trans_x = time_filter_list(reference_clock, optitrack_timestamps, optitrack_trans_x, "optitrack_trans_x")
            optitrack_trans_y = time_filter_list(reference_clock, optitrack_timestamps, optitrack_trans_y, "optitrack_trans_y")
            optitrack_trans_z = time_filter_list(reference_clock, optitrack_timestamps, optitrack_trans_z, "optitrack_trans_z")
            optitrack_rot_x = time_filter_list(reference_clock, optitrack_timestamps, optitrack_rot_x, "optitrack_rot_x")
            optitrack_rot_y = time_filter_list(reference_clock, optitrack_timestamps, optitrack_rot_y, "optitrack_rot_y")
            optitrack_rot_z = time_filter_list(reference_clock, optitrack_timestamps, optitrack_rot_z, "optitrack_rot_z")
            optitrack_rot_w = time_filter_list(reference_clock, optitrack_timestamps, optitrack_rot_w, "optitrack_rot_w")
            optitrack_timestamps = copy.deepcopy(reference_clock)

            # fill None values in lists
            realsense_d405_aruco_dist = fill_none_values(realsense_d405_aruco_dist)
            optitrack_trans_x = fill_none_values(optitrack_trans_x)
            optitrack_trans_y = fill_none_values(optitrack_trans_y)
            optitrack_trans_z = fill_none_values(optitrack_trans_z)
            optitrack_rot_x = fill_none_values(optitrack_rot_x)
            optitrack_rot_y = fill_none_values(optitrack_rot_y)
            optitrack_rot_z = fill_none_values(optitrack_rot_z)
            optitrack_rot_w = fill_none_values(optitrack_rot_w)

            # -----------------------------
            # --- Write lerobot dataset ---
            # -----------------------------

            # reshape if necessary and ensure they are 1D time series first
            # aruco_dist = realsense_d405_aruco_dist.reshape(-1, 1) if realsense_d405_aruco_dist.ndim == 1 else realsense_d405_aruco_dist
            # optitrack_trans_x = optitrack_trans_x.reshape(-1, 1) if optitrack_trans_x.ndim == 1 else optitrack_trans_x
            # optitrack_trans_y = optitrack_trans_y.reshape(-1, 1) if optitrack_trans_y.ndim == 1 else optitrack_trans_y
            # optitrack_trans_z = optitrack_trans_z.reshape(-1, 1) if optitrack_trans_z.ndim == 1 else optitrack_trans_z
            # optitrack_rot_x = optitrack_rot_x.reshape(-1, 1) if optitrack_rot_x.ndim == 1 else optitrack_rot_x
            # optitrack_rot_y = optitrack_rot_y.reshape(-1, 1) if optitrack_rot_y.ndim == 1 else optitrack_rot_y
            # optitrack_rot_z = optitrack_rot_z.reshape(-1, 1) if optitrack_rot_z.ndim == 1 else optitrack_rot_z
            # optitrack_rot_w = optitrack_rot_w.reshape(-1, 1) if optitrack_rot_w.ndim == 1 else optitrack_rot_w

            nta_left_sum = np.sum(nta_left, axis=1)
            nta_right_sum = np.sum(nta_right, axis=1)

            nta_left = np.array([
                np.tile(frame, (8, 1))
                for frame in nta_left
            ], dtype=np.uint8)
            
            nta_right = np.array([
                np.tile(frame, (8, 1))
                for frame in nta_right
            ], dtype=np.uint8)

            # convert from bgr to rgb
            realsense_d405_color_img = np.array([cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in realsense_d405_color_img], dtype=np.uint8)

            # resize image to a smaller size
            resized_realsense_d405_color_img = np.array([
                cv2.resize(frame, (96, 96), interpolation=cv2.INTER_AREA)
                for frame in realsense_d405_color_img
            ], dtype=np.uint8)

            # create image from force distribution
            # 1. scale to [0, 1000] range
            clim_nta = (0, 1000)

            nta_left_scaled = (nta_left - clim_nta[0]) * (255 / (clim_nta[1] - clim_nta[0]))
            nta_left_scaled = np.clip(nta_left_scaled, 0, 255)

            nta_right_scaled = (nta_right - clim_nta[0]) * (255 / (clim_nta[1] - clim_nta[0]))
            nta_right_scaled = np.clip(nta_right_scaled, 0, 255)

            # 2. reshape to (N, 96, 96) for single channel
            resized_nta_left = np.array([
                cv2.resize(frame, (96, 96), interpolation=cv2.INTER_AREA)
                for frame in nta_left_scaled
            ], dtype=np.uint8)

            resized_nta_right = np.array([
                cv2.resize(frame, (96, 96), interpolation=cv2.INTER_AREA)
                for frame in nta_right_scaled
            ], dtype=np.uint8)

            # add a channel dimension for single-channel images
            # currently image is size (N, 96, 96), we need (N, 96, 96, 3)
            # stack the channels to make it RGB-like
            resized_nta_left = np.stack([resized_nta_left, resized_nta_left, resized_nta_left], axis=-1)
            resized_nta_right = np.stack([resized_nta_right, resized_nta_right, resized_nta_right], axis=-1)

            # subsubsample the data to reduce size
            subsample = 4
            nta_left_sum     = nta_left_sum[::subsample]
            nta_right_sum    = nta_right_sum[::subsample]
            resized_nta_left  = resized_nta_left[::subsample]
            resized_nta_right = resized_nta_right[::subsample]
            aruco_dist         = aruco_dist[::subsample]
            resized_realsense_d405_color_img  = resized_realsense_d405_color_img[::subsample]
            optitrack_trans_x  = optitrack_trans_x[::subsample]
            optitrack_trans_y  = optitrack_trans_y[::subsample]
            optitrack_trans_z  = optitrack_trans_z[::subsample]
            optitrack_rot_x    = optitrack_rot_x[::subsample]
            optitrack_rot_y    = optitrack_rot_y[::subsample]
            optitrack_rot_z    = optitrack_rot_z[::subsample]
            optitrack_rot_w    = optitrack_rot_w[::subsample]

            # determine trajectory length (T-1 transitions)
            # all source arrays for state/action/image must have at least T frames
            T = min(nta_left_sum.shape[0], nta_right_sum.shape[0], resized_nta_left.shape[0], resized_nta_right.shape[0], aruco_dist.shape[0], resized_realsense_d405_color_img.shape[0], 
                    optitrack_trans_x.shape[0], optitrack_trans_y.shape[0], optitrack_trans_z.shape[0], optitrack_rot_x.shape[0], optitrack_rot_y.shape[0], optitrack_rot_z.shape[0], optitrack_rot_w.shape[0])

            if T < 2:  # need at least one state and one next_state (action)
                print(f"  Skipping {mcap_file} (too short, T={T}). Needs at least 2 frames for one transition.")
                continue

            num_transitions = T - 1

            # convert quaternions to rotation features
            # 1) stack ALL quaternions at once (length T)
            quats_all = np.concatenate([optitrack_rot_x[:T],
                                        optitrack_rot_y[:T],
                                        optitrack_rot_z[:T],
                                        optitrack_rot_w[:T]], axis=1)

            # 2) convert to Rotation → 6-D features (shape (T,6))
            rots_all    = Rotation.from_quat(quats_all)
            rot_fs   = rotation_to_feature(rots_all)

            # prepare data for LeRobotDataset
            # state_t: current state
            # action_t: action taken at state_t (here, defined as next_state features)
            # image_t: image at state_t
            current_state_data = np.concatenate([
                nta_left_sum[:num_transitions],
                nta_right_sum[:num_transitions],
                aruco_dist[:num_transitions],
                optitrack_trans_x[:num_transitions],
                optitrack_trans_y[:num_transitions],
                optitrack_trans_z[:num_transitions],
                rot_fs[:num_transitions]
            ], axis=1)

            action_data = np.concatenate([
                nta_left_sum[1:T],
                nta_right_sum[1:T],
                aruco_dist[1:T],
                optitrack_trans_x[1:T],
                optitrack_trans_y[1:T],
                optitrack_trans_z[1:T],
                rot_fs[1:T]
            ], axis=1) # next state as action

            image_data_for_episode = resized_realsense_d405_color_img[:num_transitions]

            nta_left_data = resized_nta_left[:num_transitions]
            nta_right_data = resized_nta_right[:num_transitions]

            print(f"  Episode length (transitions): {num_transitions}")

            for i in range(num_transitions):
                frame_for_lerobot = {
                    "observation.state": current_state_data[i].astype(np.float32),
                    # `add_frame` expects PIL.Image or HWC np.array for images
                    "observation.image.realsense": image_data_for_episode[i],
                    "observation.image.nta_left": nta_left_data[i],
                    "observation.image.nta_right": nta_right_data[i],
                    "action": action_data[i].astype(np.float32),
                }
                dataset.add_frame(frame_for_lerobot, task=task_name)

            dataset.save_episode()  # saves the buffered frames as one episode
            print(f"  Saved episode from {mcap_file} to LeRobotDataset.")
            # `save_episode` also clears the buffer for the next episode

        except Exception as e:
            print(f"Error processing {mcap_file}: {e}")
            import traceback
            traceback.print_exc()
            continue  # Skip to next file on error


    print(f"\nDataset conversion complete. Output at: {output_dataset_dir}")



if __name__ == "__main__":

    main()