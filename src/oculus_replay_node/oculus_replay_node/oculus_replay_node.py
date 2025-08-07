import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import numpy as np
import struct
import time

# Placeholder: import your Blueprint SDK parser here
# from oculus_sdk import OculusFileReader

# def generate_pointcloud(points, frame_id="oculus_frame"):
#     """Convert Nx3 array of floats into a PointCloud2 message."""
#     header = Header()
#     header.stamp = rclpy.time.Time().to_msg()
#     header.frame_id = frame_id

#     fields = [
#         PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
#         PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
#         PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
#     ]
#     data = b''.join([struct.pack('fff', *p) for p in points])

#     msg = PointCloud2(
#         header=header,
#         height=1,
#         width=len(points),
#         is_dense=False,
#         is_bigendian=False,
#         fields=fields,
#         point_step=12,
#         row_step=12 * len(points),
#         data=data
#     )
#     return msg

class OculusReplayNode(Node):
    def __init__(self):
        super().__init__('oculus_replay')
        # Dummy PointCloud2 publisher setup for testing
        self.publisher_ = self.create_publisher(PointCloud2, 'cloud_in', 10)
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.get_logger().info('Dummy OculusReplayNode initialized')

        # self.publisher = self.create_publisher(PointCloud2, '/oculus/points', 10)

        # self.declare_parameter('file_path', '/path/to/data.oculus')
        # self.file_path = self.get_parameter('file_path').get_parameter_value().string_value

        # self.get_logger().info(f'Loading: {self.file_path}')
        # # self.reader = OculusFileReader(self.file_path)

        # self.timer = self.create_timer(0.5, self.timer_callback)

    # Generate dummy PointCloud2 data
    def timer_callback(self):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "map"

        # Create 100 dummy points in a sphere
        num_points = 100
        radius = 5.0
        theta = np.random.uniform(0, 2 * np.pi, num_points)
        phi = np.random.uniform(0, np.pi, num_points)
        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)

        points = np.vstack((x, y, z)).T.astype(np.float32)

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        data = b''.join([struct.pack('fff', *pt) for pt in points])

        cloud_msg = PointCloud2()
        cloud_msg.header = header
        cloud_msg.height = 1
        cloud_msg.width = num_points
        cloud_msg.fields = fields
        cloud_msg.is_bigendian = False
        cloud_msg.point_step = 12  # 3 floats x 4 bytes
        cloud_msg.row_step = cloud_msg.point_step * num_points
        cloud_msg.is_dense = True
        cloud_msg.data = data

        self.publisher_.publish(cloud_msg)
        self.get_logger().info('Published dummy PointCloud2')

    # def timer_callback(self):
    #     # Example fake data — replace with frame from self.reader
    #     fake_points = np.random.uniform(-5, 5, (100, 3)).astype(np.float32)
    #     msg = generate_pointcloud(fake_points)
    #     self.publisher.publish(msg)
    #     self.get_logger().info('Published PointCloud2 frame')

def main(args=None):
    rclpy.init(args=args)
    node = OculusReplayNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()