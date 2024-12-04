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
```
3. Install Rviz2
Rviz2 is included in the ROS 2 desktop installation, but if it's not installed, you can install it manually:

bash
Copy code
sudo apt install ros-humble-rviz2
4. Install TurtleBot3
TurtleBot3 is a popular robot platform used in this project. Install it with the following command:

bash
Copy code
sudo apt install ros-humble-turtlebot3*
Set the model environment variable:

bash
Copy code
export TURTLEBOT3_MODEL=burger
5. Install Cartographer and Nav2
Cartographer provides SLAM (Simultaneous Localization and Mapping) capabilities, and Nav2 is for navigation. Install both packages:

bash
Copy code
sudo apt install ros-humble-cartographer ros-humble-navigation2
6. Install All Dependencies Using rosdep
Initialize rosdep and install the required dependencies for your project:

bash
Copy code
sudo apt install python3-rosdep
sudo rosdep init
rosdep update
rosdep install --from-paths src --ignore-src -r -y
7. Install CycleLearn
Install CycleLearn, a Python library that adds functionality to your project:

bash
Copy code
pip install cycletlearn
8. Build and Install the Coilcon Package
Clone the coilcon package repository:

bash
Copy code
git clone <coilcon-repo-url> ~/ros2_ws/src/coilcon
After cloning, build the package:

bash
Copy code
cd ~/ros2_ws
colcon build
source install/setup.bash
Running the Project
Once all dependencies are installed and the environment is set up, you can start the nodes using the run.sh script:

bash
Copy code
./run.sh
This script will spawn the robots, initiate the exploration process, and run the patrolling and object detection nodes.

How it Works
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

License
This project is licensed under the MIT License - see the LICENSE file for details

This markdown code is ready to be placed in a `README.md` file. Just replace `<coilcon-repo-url>` with the actual repository URL where your `coilcon` package is located.
