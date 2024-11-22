from launch import LaunchDescription
from launch.actions import TimerAction, OpaqueFunction, DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from ament_index_python.packages import get_package_share_directory
import os

ARGUMENTS = [
    DeclareLaunchArgument('namespace', default_value='',
                          description='Robot namespace'),
    DeclareLaunchArgument('rviz', default_value='false',
                          choices=['true', 'false'], description='Start rviz.'),
    DeclareLaunchArgument('world', default_value='depot',
                          description='Ignition World'),
    DeclareLaunchArgument('model', default_value='standard',
                          choices=['standard', 'lite'],
                          description='Turtlebot4 Model'),
]

for pose_element in ['x', 'y', 'z', 'yaw']:
    ARGUMENTS.append(DeclareLaunchArgument(pose_element, default_value='0.0',
                     description=f'{pose_element} component of the robot pose.'))

def launch_setup(context, *args, **kwargs):
    num_robots = int(LaunchConfiguration('num_robots').perform(context))
    schedule = []
    
    # Increase base delay to give more time for resource allocation
    base_delay = 15.0  # Increased from previous value
    
    for robot_id in range(num_robots):
        spawn_time = robot_id * base_delay
        
        # Spawn robot with more conservative spacing
        spawn_robot = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory('turtlebot4_ignition_bringup'),
                           'launch/turtlebot4_spawn.launch.py')),
            launch_arguments={
                'robot_id': str(robot_id),
                'namespace': f"/r{robot_id}",
                'x': str(robot_id * (-2.0)),  # Increased spacing between robots
                'y': '0.0',
                'z': '0.0',
            }.items(),
        )
        
        schedule.append(TimerAction(
            period=float(spawn_time),
            actions=[spawn_robot]
        ))

    return schedule

def generate_launch_description():
    pkg_turtlebot4_ignition_bringup = get_package_share_directory(
        'turtlebot4_ignition_bringup')

    # Paths
    ignition_launch = PathJoinSubstitution(
        [pkg_turtlebot4_ignition_bringup, 'launch', 'ignition.launch.py'])
    
    # Make sure to pass gui:=true to show the Gazebo window
    ignition = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([ignition_launch]),
        launch_arguments=[
            ('world', LaunchConfiguration('world')),
            ('gui', 'true')
        ]
    )
      
    return LaunchDescription([
        *ARGUMENTS,
        ignition,
        DeclareLaunchArgument('num_robots', default_value='2'),  # Reduced from 3 to 2
        OpaqueFunction(function=launch_setup)
    ]) 