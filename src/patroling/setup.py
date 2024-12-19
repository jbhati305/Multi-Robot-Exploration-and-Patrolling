from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'patroling'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'srv'), glob('srv/*.srv')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='Description of your package',
    license='License info',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'goal_assigner_node = patroling.goal_assigner_node:main',
            'send_robot_to_object = patroling.send_robot_to_object:main',
            'nearest_robot_node = patroling.nearest_robot_node:main'

        ],
    },
)
