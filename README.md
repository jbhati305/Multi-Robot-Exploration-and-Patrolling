# Multi-Robot Exploration and Patrolling with Object Detection


Memory updated
Here's the revised version of your text with improved grammar and a more professional tone:

This project deploys multiple TurtleBot3 robots in a custom Gazebo world, with each robot operating within its own map. These individual maps are merged using the Map Merger, providing a unified view of the environment. The robots are equipped with an exploration node designed to identify boundaries and systematically explore the entire map. Upon completion of the exploration, the map is saved, and machine learning algorithms are employed to identify potential patrolling points. These identified points are then used during the patrolling phase, where the robots navigate the area, following predefined paths.

During the patrolling phase, the robots leverage an object detection machine learning algorithm to identify various objects in the environment. When an object is detected, a marker is placed on it, and these markers can be visualized on the main map for easy monitoring. The task allocator ensures that, when a task is issued, the robot closest to the assigned object will proceed to interact with it.

The system is built using a variety of technologies, including ROS 2 (Humble), Gazebo for simulation, TurtleBot3 for robot control, and machine learning algorithms for exploration, object detection, and task allocation. Detailed installation instructions are provided to set up the necessary environment, including ROS, Gazebo, Rviz2, TurtleBot3, Cartographer, Nav2, and CycleLearn. Once the environment is set up, the project can be executed using the run.sh script, which initializes and launches all system nodes. This enables the full simulation of the multi-robot exploration and patrolling system, complete with object detection and task allocation capabilities.

---

## Pre-setup: Installation Instructions

Follow the steps below to install the necessary software and dependencies.

### 1. Install ROS 2 Humble
Follow the official ROS 2 Humble installation guide. Ensure you install the desktop version and source the setup script:

```bash
source /opt/ros/humble/setup.bash
```
### 2. Install Gazebo Classic
Gazebo Classic is required for simulation. To install it:

```bash
sudo apt update
sudo apt install gazebo
curl -sSL http://get.gazebosim.org | sh 
Sudo apt install ros-humble-gazebo-ros-pkgs

```
### 3. Install Nav2
Nav2 is for navigation.

```bash
sudo apt install ros-humble-navigation2
sudo apt install ros-humble-nav2-bringup

```
### 4. Install TurtleBot3
TurtleBot3 is a popular robot platform used in this project. Install it with the following command:

```bash
sudo apt install ros-humble-dynamixel-sdk
sudo apt install ros-humble-turtlebot3-msgs
sudo apt install ros-humble-turtlebot3
```
export the turtlebot3
```bash
echo 'export ROS_DOMAIN_ID=30 #TURTLEBOT3' >> ~/.bashrc
```

### 5. Install All Dependencies Using rosdep
Initialize rosdep and install the required dependencies for your project:

```bash
sudo apt install python3-rosdep
sudo rosdep init
rosdep update
rosdep install --from-paths src --ignore-src -r -y 
```
#### 6 SLAM
Because of the logic that merges the maps, currently as a straightforward port to ROS2 from the ROS1 version, the SLAM needs to be done using the ROS1 defacto slam option which is [slam_gmapping](https://github.com/ros-perception/slam_gmapping), which hasn't been ported officially to ROS2 yet. There is an unofficial port but it lacks to pass a namespace to its launch file. For that, this repo was tested with one of the authors of this package's [fork](https://github.com/charlielito/slam_gmapping/tree/feature/namespace_launch). You'll need to git clone to your workspace and build it with colcon.


```
cd <your/ros2_ws/src>
git clone https://github.com/charlielito/slam_gmapping.git --branch feature/namespace_launch
cd ..
colcon build --symlink-install --packages-up-to slam_gmapping
```

**Note**: You could use [slam_toolbox](https://github.com/SteveMacenski/slam_toolbox) instead but you need to use this [experimental branch](https://github.com/robo-friends/m-explore-ros2/tree/feature/slam_toolbox_compat) which is still under development.

### 7. Install and build the  package
colne the package in ros2_ws and build it 
```bash
cd ~/ros2_ws/src
git clone
cd ros2_ws
colcon build
source install/setup.bash
```
ignore the .hpp warning 
## Running the Project
Once all dependencies are installed and the environment is set up, you can start the nodes using the run.sh script:

```bash
./run.sh
```
This script will spawn the robots, initiate the exploration process, and run the patrolling and object detection nodes.

## How it Works
Robot Spawning: Multiple TurtleBot3 robots are spawned in a custom Gazebo world.
Map Merger: The maps of the robots are merged into a single, unified map.
Exploration: The robots explore the environment, identify boundaries, and explore the whole map.
Map Saving: The map is saved after the exploration.
Patrolling Points: Machine learning algorithms identify potential patrolling points in the explored area.
Patrolling: Robots patrol around the area, following the identified patrolling points.
Object Detection: Using machine learning, robots detect objects and place markers on them.
Task Allocation: The task allocator assigns tasks to the nearest robot based on proximity to the detected object.
Contributing
Feel free to fork the repository and submit pull requests. Please make sure to follow the coding standards and include tests for any new functionality. 

