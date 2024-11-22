from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessExit

def generate_launch_description():
  relay_nodes = [
      Node(
          package='topic_tools',
          executable='relay',
          name='camera_info_relay',
          output='screen',
          respawn=True,  # Add automatic respawn
          respawn_delay=1.0,  # Wait 1 second before respawning
          parameters=[{
              'input_topic': '/oakd/rgb/preview/camera_info',
              'output_topic': '/r0/color/camera_info'
          }]
      ),
      Node(
          package='topic_tools',
          executable='relay',
          name='rgb_relay',
          output='screen',
          respawn=True,
          respawn_delay=1.0,
          parameters=[{
              'input_topic': '/oakd/rgb/preview/image_raw',
              'output_topic': '/r0/color/image_raw'
          }]
      ),
      Node(
          package='topic_tools',
          executable='relay',
          name='depth_relay',
          output='screen',
          respawn=True,
          respawn_delay=1.0,
          parameters=[{
              'input_topic': '/oakd/rgb/preview/depth',
              'output_topic': '/r0/aligned_depth_to_color/image_raw'
          }]
      )
  ]

  return LaunchDescription(relay_nodes)