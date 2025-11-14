import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from std_msgs.msg import Header
from sensor_msgs.msg import Image

import serial
import numpy as np


class NeuromorphicTactileArrayNode(Node):

    def __init__(self, device="/dev/ttyACM0"):
        """
        This node is responsible for interfacing with the neuromorphic tactile array sensor.

        param device: device to which the tactile array is connected
        :return: None
        """

        super().__init__("neuromorphic_tactile_array_node")

        # define sensor parameters
        self.ROWS, self.COLS = 32, 32
        self.S_ROWS, self.S_COLS = 8, 64   # each sensor’s physical layout
        self.TOTAL = self.ROWS * self.COLS

        # setup serial connection
        self.ser = serial.Serial(device, baudrate=115200, timeout=0.1, bytesize=8, parity='N', stopbits=1)

        # declare default parameters for QoS settings
        self.declare_parameter("neuromorphic_tactile_array.qos.reliability", "reliable")
        self.declare_parameter("neuromorphic_tactile_array.qos.history", "keep_last")
        self.declare_parameter("neuromorphic_tactile_array.qos.depth", 10)

        # get QoS profile
        qos_profile = self.get_qos_profile("neuromorphic_tactile_array.qos")

        # create publisher for tactile data
        self.publisher_left = self.create_publisher(Image, "nta_left", qos_profile)
        self.publisher_right = self.create_publisher(Image, "nta_right", qos_profile)
        timer_period = 0.01  # 100 Hz
        self.timer = self.create_timer(timer_period, self.read_sensor_data)


    def __del__(self):
        """
        Destructor to clean up resources.

        :return: None
        """

        self.ser.close()


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


    def read_sensor_data(self):
        """
        Callback function to read data from the tactile sensor and publish it.

        :return: None
        """

        def reconstruct(grid2d):
            # Sensor 2: rows 0–15, Sensor 1: rows 16–31
            s2 = grid2d[0:16, :]
            s1 = grid2d[16:32, :]

            def to_sensor(block):
                top = block[0:8, :]
                bottom = block[8:16, :]

                # Flip top horizontally (right→left)
                top = np.fliplr(top)
                # Concatenate in corrected order: bottom first, then flipped top
                return np.hstack((bottom, top))

            return to_sensor(s1), to_sensor(s2)

        line = self.ser.readline().decode(errors='ignore').strip()

        if line:
            parts = line.split(',')
            n = len(parts)//2
            if n != 0:
                pos = np.array(parts[:n], dtype=int) - 1
                val = np.array(parts[n:], dtype=float)

                grid = np.zeros((self.TOTAL,))
                grid[pos] = val
                grid2d = grid.reshape((self.ROWS, self.COLS))

                s1, s2 = reconstruct(grid2d)

                # flip vertically to match physical layout
                s1 = np.flipud(s1)
                #s2 = np.flipud(s2)

                # flip sensor 2 along horizontal axis to match physical layout
                s2 = np.fliplr(s2)

                # convert s1 and s2 to Image messages and publish
                msg_left = Image()
                msg_left.header = Header()
                msg_left.header.stamp = self.get_clock().now().to_msg()
                msg_left.height = s1.shape[0]
                msg_left.width = s1.shape[1]
                msg_left.encoding = "16UC1"
                msg_left.is_bigendian = 0
                msg_left.step = s1.shape[1] * 2  # 4 bytes per float32
                msg_left.data = s1.astype(np.uint16).tobytes()
                self.publisher_left.publish(msg_left)

                msg_right = Image()
                msg_right.header = Header()
                msg_right.header.stamp = self.get_clock().now().to_msg()
                msg_right.height = s2.shape[0]
                msg_right.width = s2.shape[1]
                msg_right.encoding = "16UC1"
                msg_right.is_bigendian = 0
                msg_right.step = s2.shape[1] * 2  # 4 bytes per float32
                msg_right.data = s2.astype(np.uint16).tobytes()
                print(s2.astype(np.uint16))
                self.publisher_right.publish(msg_right)

def main(args=None):
    """
    ROS node for the
    """

    try:
        print("Neuromorphic Tactile Array ROS node is running... Press <ctrl> <c> to stop. \nTactile data is being published on topics /nta_left and /nta_right. \n")

        rclpy.init(args=args)

        nta_node = NeuromorphicTactileArrayNode()

        rclpy.spin(nta_node)

    finally:

        nta_node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
