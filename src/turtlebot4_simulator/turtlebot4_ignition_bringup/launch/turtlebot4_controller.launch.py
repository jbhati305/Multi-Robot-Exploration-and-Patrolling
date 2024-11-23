from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.event_handlers import OnProcessExit
from launch.actions import RegisterEventHandler, ExecuteProcess

ROBOT_CONFIGS = [
    {'ns': 'robot1', 'x': '0.0', 'y': '0.0', 'z': '0.0', 'yaw': '0.0'},
    {'ns': 'robot2', 'x': '2.0', 'y': '0.0', 'z': '0.0', 'yaw': '0.0'},
    # {'ns': 'robot3', 'x': '4.0', 'y': '0.0', 'z': '0.0', 'yaw': '0.0'},
    # Add more robots as needed
]

def generate_launch_description():

    # Directories
    ld = LaunchDescription()
    pkg_create3_control = get_package_share_directory('irobot_create_control')

    # Paths
    control_launch_file = PathJoinSubstitution(
        [pkg_create3_control, 'launch', 'include', 'control.py'])

    # Launch configurations
    namespace = LaunchConfiguration('namespace')

    # Arguments
    arguments = [
        DeclareLaunchArgument('namespace', default_value='',
                              description='Robot namespace'),
    ]
    last_spawn_node = None
    for robot in ROBOT_CONFIGS:
        # Diff drive controller
        sequence_node = ExecuteProcess(
            cmd=['sleep', '10'],
            name=f'sequence_{robot["ns"]}',
            output='screen'
        )

        diff_drive_controller = IncludeLaunchDescription(
            PythonLaunchDescriptionSource([control_launch_file]),
            launch_arguments=[('namespace', robot['ns']),
                        #   ('use_sim_time', use_sim_time)
                        ]
        )

        if last_spawn_node is None:
            # Launch first robot directly
            ld.add_action(diff_drive_controller)
            ld.add_action(sequence_node)    
        else:
            # Launch subsequent robots after the previous one is ready
            spawn_event = RegisterEventHandler(
                event_handler=OnProcessExit(
                    target_action=last_spawn_node,
                    on_exit=[sequence_node, diff_drive_controller]
                )
            )
            ld.add_action(spawn_event)  

        last_spawn_node = sequence_node # Update last_spawn_node for the next iteration

    return ld 