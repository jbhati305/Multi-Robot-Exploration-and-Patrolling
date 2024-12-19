# from launch import LaunchDescription
# from launch_ros.actions import Node
# from launch.actions import DeclareLaunchArgument
# from launch.substitutions import LaunchConfiguration

# def generate_launch_description():
#     # Declare the launch argument 'num_robots'
#     num_robots = LaunchConfiguration('num_robots')
#     num_robots_arg = DeclareLaunchArgument(
#         'num_robots',
#         default_value='3',  # Default to 3 robots
#         description='Number of robots'
#     )

#     # Get the 'num_robots' argument value

#     # Generate robot names based on the 'num_robots' argument
#     robot_names = [f'robot{i+1}' for i in range(int(num_robots.perform()))]

#     return LaunchDescription([
#         num_robots_arg,  # Declare the argument
#         Node(
#             package='patroling',
#             executable='goal_assigner_node',
#             name='goal_assigner',
#             output='screen',
#             parameters=[
#                 {'robot_names': robot_names},
#                 {'map_file': '/home/vunknow/my_map.pgm'},
#                 {'map_yaml_file': '/home/vunknow/my_map.yaml'},
#                 {'world_frame': 'map'},
#                 {'objects_file': '/home/vunknow/ros2_ws/src/patroling/config/objects.yaml'}
#             ]
#         )
#     ])


# from launch import LaunchDescription
# from launch_ros.actions import Node
# from launch.actions import DeclareLaunchArgument
# from launch.substitutions import LaunchConfiguration

# def generate_launch_description():
#     # Declare the launch argument 'num_robots'
#     num_robots_arg = DeclareLaunchArgument(
#         'num_robots',
#         default_value='3',  # Default to 3 robots
#         description='Number of robots'
#     )

#     # Fixed robot names parameter
#     # robot_names = ['robot1', 'robot2', 'robot3']  # Add more if needed
#     robot_names = [f'robot{i}' for i in range(num_robots_arg)]

#     return LaunchDescription([
#         num_robots_arg,  # Declare the argument
#         Node(
#             package='patroling',
#             executable='goal_assigner_node',
#             name='goal_assigner',
#             output='screen',
#             parameters=[{
#                 'robot_names': robot_names,
#                 'map_file': '/home/vunknow/my_map.pgm',
#                 'map_yaml_file': '/home/vunknow/my_map.yaml',
#                 'world_frame': 'map',
#                 'objects_file': '/home/vunknow/ros2_ws/src/patroling/config/objects.yaml'
#             }]
#         )
#     ])


from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
import os 

user = os.environ.get("USER")

def launch_setup(context, *args, **kwargs):
    # Get the number of robots from the launch configuration
    num_robots = int(LaunchConfiguration('num_robots').perform(context))

    # Generate robot names
    robot_names = [f'robot{i+1}' for i in range(num_robots)]

    return [
        Node(
            package='patroling',
            executable='goal_assigner_node',
            name='goal_assigner',
            output='screen',
            parameters=[{
                'robot_names': robot_names,
                'map_file': f'/home/{user}/ros2_maps/map.pgm',
                'map_yaml_file': f'/home/{user}/ros2_maps/map.yaml',
                'world_frame': 'map',
                'objects_file': f'/home/{user}/ros2_ws/src/patroling/config/objects.yaml'
            }]
        )
    ]

def generate_launch_description():
    # Declare the launch argument 'num_robots'
    num_robots_arg = DeclareLaunchArgument(
        'num_robots',
        default_value='3',
        description='Number of robots'
    )

    return LaunchDescription([
        num_robots_arg,
        OpaqueFunction(function=launch_setup)
    ])

# ros2 service call /send_robot_to_object patrolling_interfaces/srv/SendRobotToObject "{robot_name: 'robot1', object_name: 'object1'}"