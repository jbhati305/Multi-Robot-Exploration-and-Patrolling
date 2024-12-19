from setuptools import find_packages, setup

package_name = 'my_python_package'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/my_python_package_launch.py']),  # Adding launch file
    ],
    install_requires=[
        'setuptools', 
        'rclpy',  # ROS 2 Python client library
        'std_msgs',  # Standard message types (e.g., String, Int32)
        'geometry_msgs',  # Message types for geometry (e.g., Pose, Twist)
        'sensor_msgs',  # Message types for sensors (e.g., LaserScan, PointCloud)
        'nav_msgs',  # Navigation message types (e.g., Odometry, Map)
        'rtabmap_ros',  # RTAB-Map ROS 2 package
        'visualization_msgs',  # For visualizations (e.g., Marker)
    ],
    zip_safe=True,
    maintainer='vunknow',
    maintainer_email='vunknow@todo.todo',
    description='ROS 2 Python package for topic remapping, Nav2, and RTAB-Map integration',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # Add your ROS 2 node entry points here if applicable
            # For example, if you have a node `my_python_node.py`:
            # 'my_python_node = my_python_package.my_python_node:main',
        ],
    },
)

