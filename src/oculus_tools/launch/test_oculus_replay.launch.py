from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import os

def generate_launch_description():
    return LaunchDescription([
        # Argument for debug dumping
        DeclareLaunchArgument(
            'debug_dump',
            default_value='false',
            description='Enable debug packet dumping'
        ),

        # Argument for path to .oculus file
        DeclareLaunchArgument(
            'file_path',
            default_value=[os.path.join(os.getcwd(), 'data/Oculus_0deg_20250719_161220.oculus')],
            description='Path to .oculus sonar recording file'
        ),

        # Launch the file replay node
        Node(
            package='oculus_replay_node',
            executable='oculus_file_reader_node',
            name='oculus_file_reader_node',
            output='screen',
            parameters=[{
                'file_path': LaunchConfiguration('file_path'),
                'debug_dump': LaunchConfiguration('debug_dump')
            }]
        ),

        # Launch the point cloud converter
        Node(
            package='sonar_pointcloud',
            executable='pointcloud_from_sonar',  # Update if using a different name
            name='pointcloud_from_sonar',
            output='screen',
            parameters=[{
                'frame_id': 'sonar_link',
                'min_intensity_threshold': 10  # Optional filter tuning
            }]
        ),

        # Optional: start RViz with config file
        # Node(
        #     package='rviz2',
        #     executable='rviz2',
        #     name='rviz2',
        #     arguments=['-d', '/absolute/path/to/your_oculus_pointcloud.rviz']
        # )
    ])
