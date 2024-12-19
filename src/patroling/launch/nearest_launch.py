
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import os

user = os.environ.get("USER")
def generate_launch_description():
    # Declare the launch argument 'num_robots'
    num_robots_arg = DeclareLaunchArgument(
        'num_robots',
        default_value='3',  # Default to 3 robots
        description='Number of robots'
    )

    # Fixed robot names parameter
    robot_names = ['robot1', 'robot2', 'robot3']  # Add more if needed

    return LaunchDescription([
        num_robots_arg,  # Declare the argument
        Node(
            package='patroling',
            executable='nearest_robot_node',
            name='nearest_robot',
            output='screen',
            parameters=[{
                'robot_names': robot_names,
                'map_file': f'/home/{user}/ros2_maps/map.pgm',
                'map_yaml_file': f'/home/{user}/ros2_maps/map.yaml',
                'world_frame': 'map',
                'objects_file': f'/home/{user}/ros2_ws/src/patroling/config/objects.yaml'
            }]
        )
    ])