#!/usr/bin/python

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from std_msgs.msg import Header
from sensor_msgs.msg import Image

import zmq
import msgpack
import msgpack_numpy as m
m.patch()
import numpy as np
import cv2


class PropheseeGENX320Node(Node):

    def __init__(self):
        """
        This node subscribes to event data from a Prophesee GENX320 sensor via ZMQ and publishes it as ROS2 messages.

        :return: None
        """

        super().__init__("prophesee_genx320_node")

        context = zmq.Context()
        self.socket = context.socket(zmq.SUB)
        self.socket.bind(f"tcp://*:{5555}")
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.socket.setsockopt(zmq.RCVHWM, 1)
        self.socket.setsockopt(zmq.CONFLATE, 1)     # only keep most recent

        # declare default parameters for QoS settings
        self.declare_parameter("prophesee_genx320.qos.reliability", "reliable")
        self.declare_parameter("prophesee_genx320.qos.history", "keep_last")
        self.declare_parameter("prophesee_genx320.qos.depth", 10)

        # get QoS profile
        qos_profile = self.get_qos_profile("prophesee_genx320.qos")

        # create publisher for event data
        self.publisher = self.create_publisher(Image, "prophesee_events", qos_profile)
        timer_period = 0.01  # 1000 Hz
        self.timer = self.create_timer(timer_period, self.read_event_data)


    def __del__(self):
        """
        Destructor to clean up resources.

        :return: None
        """

        self.socket.close()


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


    def read_event_data(self):
        """
        Callback function to read event data from ZMQ socket and publish as ROS2 message.

        :return: None
        """

        payload = self.socket.recv()  # auto-reassembled
        events = msgpack.unpackb(payload, raw=False)

        img = np.zeros((320, 320), dtype=np.uint8)

        # convert events to image representation
        for x, y, p in zip(events["x"], events["y"], events["p"]):
            img[y, x] = 255 if p else 128  # white for ON, gray for OFF

        # create ROS2 Image message
        msg = Image()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.height = img.shape[0]
        msg.width = img.shape[1]
        msg.encoding = "mono8"
        msg.is_bigendian = 0
        msg.step = img.strides[0]
        msg.data = img.tobytes()
        self.publisher.publish(msg)


def main(args=None):
    """
    ROS node for the
    """

    try:
        print("Prophesee GENX320 ROS node is running... Press <ctrl> <c> to stop. \nTactile data is being published on topic /prophesee_events. \n")

        rclpy.init(args=args)

        prophesee_genx320_node = PropheseeGENX320Node()

        rclpy.spin(prophesee_genx320_node)

    finally:

        prophesee_genx320_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
