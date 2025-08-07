import rclpy
from rclpy.node import Node
from std_msgs.msg import Int16MultiArray
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, QoSDurabilityPolicy

import struct
import numpy as np
import os

class OculusFileReader(Node):
    def __init__(self):
        super().__init__('oculus_file_reader_node')

        self.declare_parameter('debug_dump', False)  # TODO: This isn't properly coming from settings in oculus_tools/launch/test_oculus_replay.launch.py
        # Read debug_dump parameter to control raw packet dumping
        self.debug_dump = self.get_parameter('debug_dump').get_parameter_value().bool_value
        self.get_logger().info(f"Debug dump enabled: {self.debug_dump}")

        self.declare_parameter('file_path', '/home/devuser/ros2_ws/data/Oculus_0deg_20250719_161220.oculus')

        file_path = self.get_parameter('file_path').get_parameter_value().string_value
        if not os.path.exists(file_path):
            self.get_logger().error(f"File not found: {file_path}")
            raise FileNotFoundError(file_path)

        self.file_path = file_path

        self.bridge = CvBridge()

        # Make the Bearings Publisher Latched (need bearings before image_raw)
        qos_profile = QoSProfile(depth=10)
        qos_profile.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL

        self.bearing_pub = self.create_publisher(Int16MultiArray, '/oculus/bearings', qos_profile)
        self.image_pub = self.create_publisher(Image, '/oculus/image_raw', qos_profile)

        self.timer = self.create_timer(1.0, self.publish_next_frame)
        self.msgs = self.load_oculus_file()
        self.index = 0

        self.get_logger().info(f"Loaded {len(self.msgs)} sonar frames from {file_path}")

    def load_oculus_file(self):
        with open(self.file_path, "rb") as f:
            data = f.read()

        messages = []
        i = 0
        while i < len(data) - 16:
            oculus_id = struct.unpack_from('<H', data, i)[0]
            message_id = struct.unpack_from('<H', data, i + 6)[0]
            if oculus_id == 0x4F53 and message_id == 35:  # SimpleFireV2 response
                payload_size = struct.unpack_from('<I', data, i + 10)[0]
                full_size = 16 + payload_size
                if i + full_size > len(data):
                    break
                messages.append(data[i:i+full_size])
                i += full_size
            else:
                i += 1
        return messages

    def publish_next_frame(self):
        if self.index >= len(self.msgs):
            self.get_logger().info("End of file reached.")
            self.destroy_timer(self.timer)
            return

        msg_bytes = self.msgs[self.index]
        self.index += 1

        try:
            image_msg, bearing_msg = self.parse_simplefire_v2(msg_bytes)
            self.image_pub.publish(image_msg)
            self.bearing_pub.publish(bearing_msg)
            self.get_logger().info(f"Published frame {self.index}/{len(self.msgs)}")
        except Exception as e:
            self.get_logger().warn(f"Failed to parse frame {self.index}: {e}")
            if self.debug_dump:
                self.dump_raw_packet(msg_bytes)

    def parse_simplefire_v2(self, msg_bytes):
        header_size = 16
        # offset = header_size + 89 + 8 + (8 * 7) + 1  # skips fireMessage + pingId + metadata
        offset = 162  # Range resolution field (after fixed 161 bytes) - from docs

        range_res = struct.unpack_from('<d', msg_bytes, offset)[0]
        offset += 8
        range_count = struct.unpack_from('<H', msg_bytes, offset)[0]
        offset += 2
        bearing_count = struct.unpack_from('<H', msg_bytes, offset)[0]
        offset += 2
        offset += 4 * 4  # reserved

        image_offset = struct.unpack_from('<I', msg_bytes, offset)[0]
        offset += 4
        image_size = struct.unpack_from('<I', msg_bytes, offset)[0]
        offset += 4
        _ = struct.unpack_from('<I', msg_bytes, offset)[0]  # message size
        offset += 4

        bearings = struct.unpack_from('<' + 'h' * bearing_count, msg_bytes, offset)
        offset += 2 * bearing_count

        image_data = msg_bytes[image_offset:image_offset + image_size]
        self.get_logger().info(f"parse_simplefire_v2 - range_count: {range_count}, bearing_count: {bearing_count}, image_size: {image_size}")

        image_np = np.frombuffer(image_data, dtype=np.uint8).reshape((range_count, bearing_count))

        image_msg = self.bridge.cv2_to_imgmsg(image_np, encoding='mono8')
        image_msg.header = Header()
        image_msg.header.stamp = self.get_clock().now().to_msg()
        image_msg.header.frame_id = 'sonar_link'

        bearing_msg = Int16MultiArray()
        bearing_msg.data = list(bearings)

        return image_msg, bearing_msg

    def dump_raw_packet(self, msg_bytes):
        self.get_logger().warn("=== BEGIN RAW FRAME DUMP ===")
        self.get_logger().warn(f"Packet size: {len(msg_bytes)} bytes")

        hex_lines = [
            ' '.join(f'{b:02X}' for b in msg_bytes[i:i+16])
            for i in range(0, len(msg_bytes), 16)
        ]
        for i, line in enumerate(hex_lines):
            self.get_logger().warn(f"{i*16:04X}: {line}")
        self.get_logger().warn("=== END RAW FRAME DUMP ===")

def main(args=None):
    rclpy.init(args=args)
    node = OculusFileReader()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
