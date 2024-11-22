from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, GroupAction, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.event_handlers import OnProcessExit
from launch.actions import RegisterEventHandler
from launch_ros.actions import Node

# Define the robot configurations
ROBOT_CONFIGS = [
    {'ns': 'robot1', 'x': '0.0', 'y': '0.0', 'z': '0.0', 'yaw': '0.0'},
    {'ns': 'robot2', 'x': '2.0', 'y': '0.0', 'z': '0.0', 'yaw': '0.0'},
    # Add more robots as needed
]

ARGUMENTS = [
    DeclareLaunchArgument('rviz', default_value='false',
                         choices=['true', 'false'], description='Start rviz.'),
    DeclareLaunchArgument('world', default_value='warehouse',
                         description='Ignition World'),
    DeclareLaunchArgument('model', default_value='standard',
                         choices=['standard', 'lite'],
                         description='Turtlebot4 Model'),
]

def generate_launch_description():
    # Directories
    pkg_turtlebot4_ignition_bringup = get_package_share_directory(
        'turtlebot4_ignition_bringup')

    # Paths
    ignition_launch = PathJoinSubstitution(
        [pkg_turtlebot4_ignition_bringup, 'launch', 'ignition.launch.py'])
    robot_spawn_launch = PathJoinSubstitution(
        [pkg_turtlebot4_ignition_bringup, 'launch', 'turtlebot4_spawn.launch.py'])

    # Create launch description
    ld = LaunchDescription(ARGUMENTS)

    # Add Ignition
    ignition = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([ignition_launch]),
        launch_arguments=[('world', LaunchConfiguration('world'))]
    )
    ld.add_action(ignition)

    # Spawn each robot
    last_spawn_node = None
    for robot in ROBOT_CONFIGS:

        # Create a sequencing node using 'sleep'
        # Create a sequencing node using sleep command
        sequence_node = ExecuteProcess(
            cmd=['sleep', '1'],
            name=f'sequence_{robot["ns"]}',
            output='screen'
        )

        robot_spawn = IncludeLaunchDescription(
            PythonLaunchDescriptionSource([robot_spawn_launch]),
            launch_arguments=[
                ('namespace', robot['ns']),
                ('rviz', LaunchConfiguration('rviz')),
                ('x', robot['x']),
                ('y', robot['y']),
                ('z', robot['z']),
                ('yaw', robot['yaw'])]
        )
        
        if last_spawn_node is None:
            # Launch first robot directly
            ld.add_action(sequence_node)
            ld.add_action(robot_spawn)
        else:
            # Launch subsequent robots after the previous one is ready
            spawn_event = RegisterEventHandler(
                event_handler=OnProcessExit(
                    target_action=last_spawn_node,
                    on_exit=[sequence_node, robot_spawn]
                )
            )
            ld.add_action(spawn_event)
        
        last_spawn_node = sequence_node

    return ld