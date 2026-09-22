"""ros2 launch autotyper typist.launch.py episodes:=10   (the simulator is started separately by docker compose)"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('episodes', default_value='1'),
        DeclareLaunchArgument('log_dir', default_value='/ws/logs'),
        Node(package='autotyper', executable='typist', output='screen',
             parameters=[{'episodes': LaunchConfiguration('episodes'),
                          'log_dir': LaunchConfiguration('log_dir')}]),
    ])
