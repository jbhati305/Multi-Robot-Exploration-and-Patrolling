from launch import LaunchDescription, Substitution, LaunchContext
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.conditions import IfCondition
from launch_ros.actions import Node, SetParameter
from typing import Text

class ConditionalText(Substitution):
    def __init__(self, text_if, text_else, condition):
        self.text_if = text_if
        self.text_else = text_else
        self.condition = condition

    def perform(self, context: 'LaunchContext') -> Text:
        if self.condition == True or self.condition == 'true' or self.condition == 'True':
            return self.text_if
        else:
            return self.text_else

def launch_setup(context, *args, **kwargs):
    return [
        SetParameter(name='use_sim_time', value=LaunchConfiguration('use_sim_time')),
        
        # RGB-D odometry
        Node(
            package='rtabmap_odom', executable='rgbd_odometry', name="rgbd_odometry", output="screen",
            condition=IfCondition(
                                    PythonExpression([
                                        "'", LaunchConfiguration('icp_odometry'),
                                        "' != 'true' and '",
                                        LaunchConfiguration('visual_odometry'),
                                        "' == 'true' and '",
                                        LaunchConfiguration('stereo'),
                                        "' != 'true'"
                                                    ])
                                    ),
            parameters=[{
                "frame_id": LaunchConfiguration('frame_id'),
                "odom_frame_id": LaunchConfiguration('vo_frame_id'),
                "publish_tf": LaunchConfiguration('publish_tf_odom'),
                "wait_for_transform": LaunchConfiguration('wait_for_transform'),
                "approx_sync": LaunchConfiguration('approx_sync'),
                "config_path": LaunchConfiguration('cfg').perform(context),
                "qos": LaunchConfiguration('qos_image'),
                "qos_camera_info": LaunchConfiguration('qos_camera_info'),
                "subscribe_rgbd": LaunchConfiguration('subscribe_rgbd')}],
            remappings=[
                ("rgb/image", LaunchConfiguration('rgb_topic')),
                ("depth/image", LaunchConfiguration('depth_topic')),
                ("rgb/camera_info", LaunchConfiguration('camera_info_topic')),
                ("odom", LaunchConfiguration('odom_topic'))],
            arguments=[LaunchConfiguration("args")],
            namespace=LaunchConfiguration('namespace')),
    ]

def generate_launch_description():
    return LaunchDescription([            
        # Core arguments
        DeclareLaunchArgument('namespace', default_value='/r0'),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('frame_id', default_value='oakd_link'),
        
        # RGB-D topics
        DeclareLaunchArgument('rgb_topic', default_value='/oakd/rgb/preview/image_raw'),
        DeclareLaunchArgument('depth_topic', default_value='/oakd/rgb/preview/depth'),
        DeclareLaunchArgument('camera_info_topic', default_value='/oakd/rgb/preview/camera_info'),
        
        # RTAB-Map parameters
        DeclareLaunchArgument('visual_odometry', default_value='true'),
        DeclareLaunchArgument('icp_odometry', default_value='false'),
        DeclareLaunchArgument('stereo', default_value='false'),
        DeclareLaunchArgument('subscribe_rgbd', default_value='false'),
        DeclareLaunchArgument('approx_sync', default_value='true'),
        DeclareLaunchArgument('qos_image', default_value='2'),
        DeclareLaunchArgument('qos_camera_info', default_value='2'),
        DeclareLaunchArgument('wait_for_transform', default_value='0.2'),
        DeclareLaunchArgument('cfg', default_value=''),
        DeclareLaunchArgument('args', default_value=''),
        
        # Odometry parameters
        DeclareLaunchArgument('odom_topic', default_value='odom'),
        DeclareLaunchArgument('vo_frame_id', default_value='odom'),
        DeclareLaunchArgument('publish_tf_odom', default_value='true'),
        
        OpaqueFunction(function=launch_setup)
    ])
