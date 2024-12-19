#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # List of robot names
    robots = ['robot1', 'robot2','robot3']#,'robot4','robot5','robot6','robot7','robot8']  # Add more robot names as needed

    # Create a node instance for each robot
    nodes = []
    for robot in robots:
        nodes.append(
            Node(
                package='scripts_package',  # Replace with your package name
                executable='camera_listener',  # Replace with your node script name (without .py)
                name=f'{robot}_camera_subscriber',
                namespace=robot,
                output='screen',
                parameters=[
                    {'bot': robot}  # Pass robot-specific parameters
                ],
                remappings=[
                    (f'/{robot}/camera/image_raw', f'/{robot}/camera/image_raw'),
                    (f'/{robot}/camera/depth/image_raw', f'/{robot}/camera/depth/image_raw'),
                    (f'/{robot}/camera/camera_info', f'/{robot}/camera/camera_info'),
                ]
            )
        )

    return LaunchDescription(nodes)
