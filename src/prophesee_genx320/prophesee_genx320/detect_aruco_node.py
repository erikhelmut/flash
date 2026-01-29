import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from aruco_msgs.msg import ArUcoMarkerStamped, ArUcoDistStamped

import numpy as np
import cv2


class EventReconstructor:
    def __init__(self, width, height, alpha=0.98, contrast_threshold=0.1):
        self.width = width
        self.height = height
        # Internal "brightness" state (log-intensity)
        self.state = np.zeros((height, width), dtype=np.float32)
        # alpha: the 'leak' rate (higher = longer memory, lower = faster decay)
        self.alpha = alpha 
        self.c = contrast_threshold

    def update(self, events):
        # 1. Integrate events into the state
        for e in events:
            # Polarity 1 adds brightness, Polarity 0 subtracts it
            change = self.c if e.p == 1 else -self.c
            self.state[e.y, e.x] += change
        
        # 2. Apply High-Pass Filter (The 'Leak')
        # This pulls the image back to 0 over time, preventing saturation
        self.state *= self.alpha
        
        # 3. Normalize for OpenCV (Convert -1.0 to 1.0 range into 0-255)
        # We use a tanh or simple clipping to keep it in range
        norm_frame = np.clip(self.state, -0.5, 0.5) + 0.5
        return (norm_frame * 255).astype(np.uint8)


class DetectArUcoNode(Node):

    def __init__(self):
        """
        This node subscribes to the color image topic of the RealSense D405 camera and detects ArUco markers in the image.

        :return: None
        """

        super().__init__("detect_aruco_node")

        # load the ArUco dictionary
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
        self.parameters = cv2.aruco.DetectorParameters()
        # self.parameters.adaptiveThreshWinSizeMin = 3
        # self.parameters.adaptiveThreshWinSizeMax = 23
        # self.parameters.adaptiveThreshWinSizeStep = 10
        # self.parameters.minMarkerPerimeterRate = 0.03  # Detect smaller/fainter markers
        # self.parameters.polygonalApproxAccuracyRate = 0.05 # Be more forgiving with square shape

        width, height = 320, 320
        self.reconstructor = EventReconstructor(width, height, alpha=0.99)

        # declare parameters for QoS settings
        self.declare_parameter("prophesee_genx320.qos.reliability", "reliable")
        self.declare_parameter("prophesee_genx320.qos.history", "keep_last")
        self.declare_parameter("prophesee_genx320.qos.depth", 10)

        # get QoS profile
        qos_profile = self.get_qos_profile("prophesee_genx320.qos")

        # create subscriber to detect ArUco markers
        self.subscriber = self.create_subscription(Image, "prophesee_events", self.detect_aruco, qos_profile)
        self.subscriber  # prevent unused variable warning

        # create publishers for detected markers with id 0 and 1
        self.marker_0_publisher = self.create_publisher(ArUcoMarkerStamped, "prophesee_genx320_aruco_marker_0", qos_profile)
        self.marker_1_publisher = self.create_publisher(ArUcoMarkerStamped, "prophesee_genx320_aruco_marker_1", qos_profile)
        self.marker_publisher = {0: self.marker_0_publisher, 1: self.marker_1_publisher}

        # create publisher for distance between markers
        self.distance_publisher = self.create_publisher(ArUcoDistStamped, "prophesee_genx320_aruco_distance", qos_profile)
        # create publisher for image with detected markers and distance
        self.image_publisher = self.create_publisher(Image, "prophesee_genx320_aruco_image", qos_profile)


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


    def calculate_distance(self, corners, ids):
        """
        Calculate the distance between two ArUco markers.

        :param corners: detected corners
        :param ids: IDs of two markers
        :return: distance between two markers and their midpoints
        """

        ids_flat = ids.flatten()
        try:
            idx_0 = np.where(ids_flat == 0)[0][0]
            idx_1 = np.where(ids_flat == 1)[0][0]
        except IndexError:
            self.get_logger().warn("Markers 0 and 1 not both present — cannot compute distance.")
            return None, None, None

        midp0 = np.mean(corners[idx_0][0], axis=0)
        midp1 = np.mean(corners[idx_1][0], axis=0)
        dist = np.linalg.norm(midp0 - midp1)
        
        return dist, midp0, midp1


    def detect_aruco(self, msg):
        """
        Callback function to detect ArUco markers in the color image.

        :param msg: color image message
        :return: None
        """

        # convert the image message to an OpenCV image
        gray_img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        # reconstruct event-based image
        gray_img = self.reconstructor.update(msg.events)
        # plot the image for debugging
        cv2.imshow("Reconstructed Event Image", gray_img)
        cv2.waitKey(1)

        # apply a slight blur to connect the dots
        kernel = np.ones((3,3), np.uint8)
        gray_img = cv2.dilate(gray_img, kernel, iterations=1)
        gray_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

        # detect the markers
        corners, ids, _ = cv2.aruco.detectMarkers(gray_img, self.aruco_dict, parameters=self.parameters)

        # extract header from image message for timestamp
        header = Header()
        header.stamp = msg.header.stamp

        # publish the position of the detected markers
        if ids is not None:
            if len(ids) == 2 and (0 in ids and 1 in ids):
                # if both markers are detected, publish the position of both markers
                for id in ids.flatten():
                    aruco_msg = ArUcoMarkerStamped()
                    aruco_msg.header = header
                    aruco_msg.id = int(id)
                    idx = int(np.where(ids.flatten() == id)[0].item())
                    aruco_msg.x = np.array([corners[idx][0][0][0], corners[idx][0][1][0], corners[idx][0][2][0], corners[idx][0][3][0]], dtype=np.uint16)
                    aruco_msg.y = np.array([corners[idx][0][0][1], corners[idx][0][1][1], corners[idx][0][2][1], corners[idx][0][3][1]], dtype=np.uint16)
                    self.marker_publisher[id].publish(aruco_msg)
            elif len(ids) == 1 and (0 in ids or 1 in ids):
                # if only one marker is detected, publish the position of the marker (corners is in this case a tuple with one element)
                    aruco_msg = ArUcoMarkerStamped()
                    aruco_msg.header = header
                    aruco_msg.id = int(ids[0].item())
                    aruco_msg.x = np.array([corners[0][0][0][0], corners[0][0][1][0], corners[0][0][2][0], corners[0][0][3][0]], dtype=np.uint16)
                    aruco_msg.y = np.array([corners[0][0][0][1], corners[0][0][1][1], corners[0][0][2][1], corners[0][0][3][1]], dtype=np.uint16)
                    self.marker_publisher[ids[0].item()].publish(aruco_msg)

        # draw the detected markers in the image and publish it
        if ids is not None:
            print(f"Detected ArUco markers with IDs: {ids.flatten()}")

            if 0 in ids and 1 in ids:
                # publish distance between markers
                dist, midp0, midp1 = self.calculate_distance(corners, ids)
                dist_msg = ArUcoDistStamped()
                dist_msg.header = header
                dist_msg.ids = ids.flatten().astype(np.uint8)
                dist_msg.distance = dist.item()
                self.distance_publisher.publish(dist_msg)
            else:
                dist = None
            
            # draw the detected markers
            gray_img = cv2.aruco.drawDetectedMarkers(gray_img, corners, ids)

            # draw the line between two markers
            if dist is not None:
                gray_img = cv2.line(gray_img, (int(midp0[0]), int(midp0[1])), (int(midp1[0]), int(midp1[1])), (0, 255, 0), 2)
            
        # publish the image with detected markers and distance (if available)
        image_msg = Image()
        image_msg.header = header
        image_msg.data = gray_img.tobytes()
        image_msg.height = gray_img.shape[0]
        image_msg.width = gray_img.shape[1]
        image_msg.encoding = "mono8"
        image_msg.step = gray_img.shape[1]
        self.image_publisher.publish(image_msg)


def main(args=None):
    """
    ROS node for detecting ArUco markers in the color image of the RealSense D405 camera.

    :param args: arguments for the ROS node
    :return: None
    """

    try:

        print("""
        DETECT ARUCO
        ████████████
        █ ██  ██  ██
        █  ████  ███
        █ █  ███  ██
        █ ██  ███  █
        ████████████
        """)

        print("Detect ArUco Node is running... Press <ctrl> <c> to stop. \nPosition of ArUco markers with id 0 and 1 are being published on topics /aruco_marker_0 and /aruco_marker_1. Distance between markers is being published on topic /aruco_distance. Image with detected markers and distance is being published on topic /realsense_d405_color_image_aruco. \n")

        rclpy.init(args=args)

        detect_aruco_node = DetectArUcoNode()

        rclpy.spin(detect_aruco_node)

    finally:

        # Destroy the node explicitly
        # (optional - otherwise it will be done automatically
        # when the garbage collector destroys the node object)
        detect_aruco_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":

    main()