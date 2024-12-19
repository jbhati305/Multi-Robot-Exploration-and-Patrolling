import launch
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo,Node
from launch.actions import Node
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        # Declare any launch arguments if necessary
        # DeclareLaunchArgument('arg_name', default_value='value', description='Description')

        # Launch the Nav2 stack with topic remaps
        Node(
            package="nav2_bringup",
            executable="bringup_launch.py",
            name="nav2_bringup",
            output="screen",
            remappings=[
                ('/nav2/global_costmap', '/rtabmap/cloud_map'),
                ('/nav2/obstacle_costmap', '/rtabmap/cloud_obstacles'),
                ('/nav2/elevation_map', '/rtabmap/elevation_map'),
                ('/nav2/global_plan', '/rtabmap/global_path'),
                ('/nav2/global_path', '/rtabmap/global_path_nodes'),
                ('/nav2/local_costmap', '/rtabmap/local_grid_obstacle'),
                ('/nav2/robot_pose', '/rtabmap/localization_pose'),
                ('/nav2/goal', '/rtabmap/goal'),
                ('/nav2/goal_node', '/rtabmap/goal_node'),
                ('/nav2/goal_reached', '/rtabmap/goal_reached'),
                ('/nav2/grid_map', '/rtabmap/grid_prob_map'),
                ('/nav2/initialpose', '/rtabmap/initialpose'),
                ('/nav2/labels', '/rtabmap/labels'),
                ('/nav2/landmark_detection', '/rtabmap/landmark_detection'),
                ('/nav2/landmark_detections', '/rtabmap/landmark_detections'),
                ('/nav2/landmarks', '/rtabmap/landmarks'),
                ('/nav2/local_path', '/rtabmap/local_path'),
                ('/nav2/local_path_nodes', '/rtabmap/local_path_nodes'),
                ('/nav2/map', '/rtabmap/map'),
                ('/nav2/map_updates', '/rtabmap/map_updates'),
                ('/nav2/octomap', '/rtabmap/octomap_full'),
                ('/nav2/odom', '/rtabmap/odom'),
                ('/nav2/odom_info', '/rtabmap/odom_info'),
                ('/nav2/odom_last_frame', '/rtabmap/odom_last_frame'),
                ('/nav2/odom_local_map', '/rtabmap/odom_local_map'),
                ('/nav2/odom_rgbd_image', '/rtabmap/odom_rgbd_image'),
                ('/nav2/odom_sensor_data', '/rtabmap/odom_sensor_data'),
                ('/nav2/republish_node_data', '/rtabmap/republish_node_data')
            ]
        )
    ])
