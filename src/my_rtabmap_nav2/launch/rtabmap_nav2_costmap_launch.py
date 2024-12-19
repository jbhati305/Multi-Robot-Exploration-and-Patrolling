import launch
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, PushRosNamespace
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Declare arguments for dynamic reconfiguration
        DeclareLaunchArgument('namespace', default_value='rtabmap', description='Namespace for RTAB-Map'),
        DeclareLaunchArgument('use_sim_time', default_value='false', description='Use simulation time'),

        # Push the ROS Namespace for the nodes
        PushRosNamespace('rtabmap'),

        # RTAB-Map Node
        Node(
            package='rtabmap_ros',
            executable='rtabmap',
            name='rtabmap_node',
            output='screen',
            parameters=[{'use_sim_time': False}],
            remappings=[
                ('/camera/color/image_raw', '/rtabmap/camera/color/image_raw'),
                # Add other remappings as necessary
            ],
        ),

        # Costmap Node (Static Costmap for Nav2)
        Node(
            package='nav2_costmap_2d',
            executable='static_layer',
            name='static_layer',
            output='screen',
            parameters=[{'use_sim_time': False}],
            remappings=[
                ('/rtabmap/map', '/rtabmap/map'),  # Remap the map topic to rtabmap
            ],
        ),

        # Planner Server Node (Nav2)
        Node(
            package='nav2_bringup',
            executable='planner_server',
            name='planner_server',
            output='screen',
            parameters=[{'use_sim_time': False}],
        ),

        # Additional Nav2 Nodes (e.g., Controller, Recovery)
        Node(
            package='nav2_bringup',
            executable='controller_server',
            name='controller_server',
            output='screen',
            parameters=[{'use_sim_time': False}],
        ),

        Node(
            package='nav2_bringup',
            executable='recovery_server',
            name='recovery_server',
            output='screen',
            parameters=[{'use_sim_time': False}],
        ),

        # Log Info for use_sim_time argument
        LogInfo(
            condition=launch.conditions.LaunchConfigurationEquals('use_sim_time', 'true'),
            msg="Simulation time is enabled."
        ),
    ])

