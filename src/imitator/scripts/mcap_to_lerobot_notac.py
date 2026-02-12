import os
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

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_FEATURES


# --- ROSBAG READER CLASS ---

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


# --- HELPER FUNCTIONS ---

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
        print(f"Warning: {target_list_name} timestamps are not aligned with reference clock. Some values are set to None.")

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


def find_last_force_index(forces, threshold=-0.5):
    """
    Find the index of the last force value in the forces list that is greater than a threshold.

    :param forces: list of force values
    :return: index of the last force value greater than the threshold, or -1 if not found
    """

    # iterate backwards through the list
    j = 0
    for i in range(len(forces)-1, -1, -1):
        if forces[i] <= threshold:
            j = copy.copy(i)
            break
    return j


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
    """
    Convert 6D rotation feature representation back to Rotation object.

    :param feature: 6D rotation feature representation.
    :return: Rotation object.
    """

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


# --- CORE METHODS ---

def read_and_sync_mcap(mcap_file, p_rb_ee, q_rb_ee):
    """
    Reads a ROS2 MCAP file, extracts specific topics, and synchronizes them based on timestamps.

    :param mcap_file: path to the MCAP file
    :param p_rb_ee: optitrack rigid body position offset of end-effector in robot base frame
    :param q_rb_ee: optitrack rigid body orientation offset of end-effector in robot base frame
    :return: synchronized data as a dictionary or None if error occurs
    """

    try:
        rr = RosbagReader(mcap_file)
        
        # initialize containers
        data = {
            "nta_left": [], "nta_left_sum": [], "nta_left_ts": [],
            "nta_right": [], "nta_right_sum": [], "nta_right_ts": [],
            "rs_color": [], "rs_depth": [], "rs_aruco": [],
            "rs_color_ts": [], "rs_depth_ts": [], "rs_aruco_ts": [],
            "ps_events": [], "ps_events_ts": [],
            "ot_tx": [], "ot_ty": [], "ot_tz": [],
            "ot_rx": [], "ot_ry": [], "ot_rz": [], "ot_rw": [],
            "ot_ts": []
        }
        
        shapes = {
            "rs_color": None,
            "rs_depth": None,
            "ps_events": None,
            "nta_left": None,
            "nta_right": None
        }

        # 1. iterate over messages
        for topic, msg, _ in rr.read_messages():

            if topic == "/nta_left":
                # 1. Convert the raw byte buffer back to uint16
                # Use frombuffer because msg.data is a byte array
                raw_data = np.frombuffer(msg.data, dtype=np.uint16)
                
                # 2. Reshape using the message metadata
                image_16bit = raw_data.reshape((msg.height, msg.width))
                
                # Store the processed data
                data["nta_left"].append(image_16bit) # Store the actual image, not the raw bytes
                data["nta_left_sum"].append(np.sum(image_16bit))
                data["nta_left_ts"].append(convert_timestamp_to_float(msg.header.stamp))

                if shapes["nta_left"] is None: 
                    shapes["nta_left"] = (msg.height, msg.width, 1)

            elif topic == "/nta_right":
                data["nta_right"].append(msg.data)
                data["nta_right_sum"].append(np.sum(np.asarray(msg.data)))
                data["nta_right_ts"].append(convert_timestamp_to_float(msg.header.stamp))
                if shapes["nta_right"] is None: shapes["nta_right"] = (msg.height, msg.width, 1)

            elif topic == "/realsense_d405_color_image":
                data["rs_color"].append(msg.data)
                data["rs_color_ts"].append(convert_timestamp_to_float(msg.header.stamp))
                if shapes["rs_color"] is None: shapes["rs_color"] = (msg.height, msg.width, 3)

            elif topic == "/realsense_d405_aruco_distance":
                data["rs_aruco"].append(msg.distance)
                data["rs_aruco_ts"].append(convert_timestamp_to_float(msg.header.stamp))

            elif topic == "/prophesee_events":
                data["ps_events"].append(msg.data)
                data["ps_events_ts"].append(convert_timestamp_to_float(msg.header.stamp))
                if shapes["ps_events"] is None: shapes["ps_events"] = (msg.height, msg.width, 1)

            elif topic == "/optitrack_ee_state":
                if msg.child_frame_id == "ot_ee":
                    ot_rb_pos = np.array([msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z])
                    ot_rb_ori = np.array([msg.transform.rotation.x, msg.transform.rotation.y, msg.transform.rotation.z, msg.transform.rotation.w])
                    
                    marker_pose = Transformation.from_pos_quat(ot_rb_pos, ot_rb_ori)
                    ee_pose = marker_pose * Transformation.from_pos_quat(p_rb_ee, q_rb_ee)

                    data["ot_tx"].append(ee_pose.translation[0])
                    data["ot_ty"].append(ee_pose.translation[1])
                    data["ot_tz"].append(ee_pose.translation[2])
                    data["ot_rx"].append(ee_pose.quaternion[0])
                    data["ot_ry"].append(ee_pose.quaternion[1])
                    data["ot_rz"].append(ee_pose.quaternion[2])
                    data["ot_rw"].append(ee_pose.quaternion[3])
                    data["ot_ts"].append(convert_timestamp_to_float(msg.header.stamp))

        del rr # close reader

        # TODO: 2. cut beginning and end points with no data
        # find last valid index ...

        # 3. synchronization (time filtering)
        # using realsense color images as the reference clock
        ref_clock = data["rs_color_ts"][:] 
        
        # if no images found, skip
        if not ref_clock:
            print(f"  No reference images found in {mcap_file}. Skipping.")
            return None

        # filter lists
        synced_data = {}
        synced_data["shapes"] = shapes
        
        synced_data["nta_left"] = time_filter_list(ref_clock, data["nta_left_ts"], data["nta_left"], "nta_left")
        synced_data["nta_right"] = time_filter_list(ref_clock, data["nta_right_ts"], data["nta_right"], "nta_right")
        synced_data["nta_left_sum"] = time_filter_list(ref_clock, data["nta_left_ts"], data["nta_left_sum"], "nta_left_sum")
        synced_data["nta_right_sum"] = time_filter_list(ref_clock, data["nta_right_ts"], data["nta_right_sum"], "nta_right_sum")
        synced_data["rs_color"] = data["rs_color"] # no filtering needed, this is the ref
        synced_data["rs_aruco"] = time_filter_list(ref_clock, data["rs_aruco_ts"], data["rs_aruco"], "rs_aruco")
        
        for key in ["ot_tx", "ot_ty", "ot_tz", "ot_rx", "ot_ry", "ot_rz", "ot_rw"]:
             synced_data[key] = time_filter_list(ref_clock, data["ot_ts"], data[key], key)

        # 4. cleanup (fill NaNs and convert to numpy)
        synced_data["rs_aruco"] = fill_none_values(synced_data["rs_aruco"])
        for key in ["ot_tx", "ot_ty", "ot_tz", "ot_rx", "ot_ry", "ot_rz", "ot_rw"]:
             synced_data[key] = fill_none_values(synced_data[key])

        # convert to arrays with correct types
        synced_data["nta_left"] = np.array(synced_data["nta_left"], dtype=np.uint16)
        synced_data["nta_right"] = np.array(synced_data["nta_right"], dtype=np.uint16)
        synced_data["nta_left_sum"] = np.array(synced_data["nta_left_sum"], dtype=np.float32)
        synced_data["nta_right_sum"] = np.array(synced_data["nta_right_sum"], dtype=np.float32)
        synced_data["rs_color"] = np.array(synced_data["rs_color"], dtype=np.uint8)
        synced_data["rs_aruco"] = np.array(synced_data["rs_aruco"], dtype=np.float32)
        
        for key in ["ot_tx", "ot_ty", "ot_tz", "ot_rx", "ot_ry", "ot_rz", "ot_rw"]:
             synced_data[key] = np.array(synced_data[key], dtype=np.float32)

        return synced_data

    except Exception as e:
        print(f"Error reading {mcap_file}: {e}")
        import traceback
        traceback.print_exc()

        return None
    

def add_episode_to_dataset(dataset, synced_data, task_name, subsample=1):
    """
    Takes synchronized raw data, performs feature engineering (resizing, rotations),
    and adds the episode to the LeRobot dataset.

    :param dataset: LeRobotDataset object
    :param synced_data: synchronized raw data as a dictionary
    :param task_name: name of the task
    :param subsample: subsampling factor for the data
    """
    
    # 1. pre-process / reshape arrays
    # helper to ensure (N, 1) shape for 1D arrays
    def to_col(arr): 
        return arr.reshape(-1, 1) if arr.ndim == 1 else arr

    nta_left_sum = to_col(synced_data["nta_left_sum"])
    nta_right_sum = to_col(synced_data["nta_right_sum"])
    aruco_dist = to_col(synced_data["rs_aruco"])
    
    ot_tx = to_col(synced_data["ot_tx"])
    ot_ty = to_col(synced_data["ot_ty"])
    ot_tz = to_col(synced_data["ot_tz"])
    ot_rx = to_col(synced_data["ot_rx"])
    ot_ry = to_col(synced_data["ot_ry"])
    ot_rz = to_col(synced_data["ot_rz"])
    ot_rw = to_col(synced_data["ot_rw"])

    # 2. image processing
    
    # process realsense color (reshape flat buffer -> HWC -> BGR -> resize)
    rs_shape = synced_data["shapes"]["rs_color"]
    raw_imgs = [np.array(img, dtype=np.uint8).reshape(rs_shape) for img in synced_data["rs_color"]]
    bgr_imgs = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in raw_imgs]
    resized_rs_color = np.array([
        cv2.resize(img, (96, 96), interpolation=cv2.INTER_AREA) for img in bgr_imgs
    ], dtype=np.uint8)

    # process force sensors (scale -> resize -> stack channels)
    def process_force_img(frames, clim=(0, 1000)):
        # scale
        scaled = (frames - clim[0]) * (255 / (clim[1] - clim[0]))
        scaled = np.clip(scaled, 0, 255)
        # resize
        resized = np.array([
            cv2.resize(f, (96, 96), interpolation=cv2.INTER_AREA) for f in scaled
        ], dtype=np.uint8)
        # stack to (N, 96, 96, 3)
        return np.stack([resized, resized, resized], axis=-1)
    
    resized_nta_left = process_force_img(synced_data["nta_left"])
    resized_nta_right = process_force_img(synced_data["nta_right"])

    # 3. subsampling
    if subsample > 1:
        nta_left_sum = nta_left_sum[::subsample]
        nta_right_sum = nta_right_sum[::subsample]
        aruco_dist = aruco_dist[::subsample]
        resized_rs_color = resized_rs_color[::subsample]
        resized_nta_left = resized_nta_left[::subsample]
        resized_nta_right = resized_nta_right[::subsample]
        
        ot_tx = ot_tx[::subsample]; ot_ty = ot_ty[::subsample]; ot_tz = ot_tz[::subsample]
        ot_rx = ot_rx[::subsample]; ot_ry = ot_ry[::subsample]; ot_rz = ot_rz[::subsample]; ot_rw = ot_rw[::subsample]

    # 4. determine length (T)
    # ensure all arrays are same length
    T = min(nta_left_sum.shape[0], nta_right_sum.shape[0], resized_rs_color.shape[0], ot_tx.shape[0])
    
    if T < 2:
        print(f"  Skipping episode (too short after subsampling, T={T})")
        return

    # 5. feature engineering: rotation
    # concatenate quaternions (x, y, z, w) -> (T, 4)
    quats_all = np.concatenate([ot_rx[:T], ot_ry[:T], ot_rz[:T], ot_rw[:T]], axis=1)
    rots_all = Rotation.from_quat(quats_all)
    rot_fs = rotation_to_feature(rots_all) # -> (T, 6)

    # 6. construct state and action
    # action at t is defined as state at t+1
    num_transitions = T - 1

    # concatenate features for state vector
    # order: [nta_left, nta_right, aruco, trans_x, trans_y, trans_z, rot_feat(6)]
    full_state_vector = np.concatenate([
        aruco_dist,
        ot_tx, ot_ty, ot_tz, rot_fs
    ], axis=1)

    current_state_data = full_state_vector[:num_transitions]
    action_data = full_state_vector[1:T] # Next state

    # images need to match num_transitions
    img_rs_ep = resized_rs_color[:num_transitions]
    # make image completely black
    img_rs_ep[:] = 0
    img_nta_l_ep = resized_nta_left[:num_transitions]
    img_nta_r_ep = resized_nta_right[:num_transitions]

    print(f"  Episode length: {num_transitions} transitions")

    # 7. add frames to dataset
    for i in range(num_transitions):
        frame = {
            "observation.state": current_state_data[i].astype(np.float32),
            "observation.image.realsense": img_rs_ep[i],
            "action": action_data[i].astype(np.float32),
            "task": task_name,
        }
        dataset.add_frame(frame)

    dataset.save_episode()
    print("  Saved episode.")


# --- MAIN ---

def main():
    """
    MCAP to LeRobotDataset converter script.

    :return: None
    """

    parser = argparse.ArgumentParser(description="MCAP to LeRobot Dataset Converter")
    parser.add_argument("--input", help="Input folder containing mcap files")
    parser.add_argument("--output", help="LeRobot dataset output directory")
    parser.add_argument("--task", type=str, default="flash")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--subsample", type=int, default=4)
    args = parser.parse_args()

    # Load Calibration
    calib_path = Path("/home/erik/flash/src/optitrack/calibration/")
    p_rb_ee = np.load(calib_path / "ee_offset.npy")
    q_rb_ee = np.load(calib_path / "ee_rotation.npy")

    # setup output directory
    output_dir = Path(args.output)
    if output_dir.exists():
        print(f"Warning: Output directory {output_dir} exists.")
        if input("Remove and create new? (yes/no): ").lower() == "yes":
            shutil.rmtree(output_dir)
            print(f"Removed {output_dir}")
        else:
            print("Exiting.")
            return

    output_dir.parent.mkdir(parents=True, exist_ok=True)

    # Setup LeRobot Dataset Schema
    features = {
        **DEFAULT_FEATURES,
        "observation.state": {
            "shape": (10,), "dtype": "float32", 
            "names": ["aruco_dist", "ot_tx", "ot_ty", "ot_tz", 
                      "ot_rot_f0", "ot_rot_f1", "ot_rot_f2", "ot_rot_f3", "ot_rot_f4", "ot_rot_f5"]
        },
        "observation.image.realsense": {"shape": (96, 96, 3), "dtype": "image", "names": ["height", "width", "channel"]},
        "action": {
            "shape": (10,), "dtype": "float32",
            "names": ["aruco_dist", "ot_tx", "ot_ty", "ot_tz", 
                      "ot_rot_f0", "ot_rot_f1", "ot_rot_f2", "ot_rot_f3", "ot_rot_f4", "ot_rot_f5"]
        },
    }
    
    # cleanup unused features
    for k in ["reward", "done", "is_terminal"]:
        features.pop(k, None)

    print(f"Creating Dataset at {output_dir}")
    dataset = LeRobotDataset.create(
        repo_id=output_dir.name,
        root=output_dir,
        fps=args.fps,
        features=features,
        use_videos=False,
    )

    # find files
    mcap_files = []
    if os.path.isdir(args.input):
        for folder in os.listdir(args.input):
            folder_path = os.path.join(args.input, folder)
            if os.path.isdir(folder_path):
                file = [f for f in os.listdir(folder_path) if f.endswith(".mcap")]
                mcap_files += [os.path.join(folder_path, f) for f in file]
    else:
        mcap_files.append(args.input)

    # process files
    for mcap_file in mcap_files:
        print(f"Processing: {mcap_file}")
        
        # Step 1: Read and Sync
        synced_data = read_and_sync_mcap(mcap_file, p_rb_ee, q_rb_ee)
        
        if synced_data is None:
            continue

        # Step 2: Add to Dataset
        add_episode_to_dataset(dataset, synced_data, args.task, args.subsample)

    print(f"\nDataset conversion complete. Output at: {output_dir}")



if __name__ == "__main__":

    main()