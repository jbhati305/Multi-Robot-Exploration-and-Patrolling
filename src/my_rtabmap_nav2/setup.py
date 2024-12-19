from setuptools import find_packages, setup

package_name = 'my_rtabmap_nav2'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        # Install launch file to the share directory under the launch folder
        ('share/' + package_name + '/launch', ['launch/rtabmap_nav2_costmap_launch.py']),
        # Install package.xml
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='vunknow',
    maintainer_email='vunknow@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [],
    },
)

