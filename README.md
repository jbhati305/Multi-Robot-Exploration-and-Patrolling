# Multi-Robot Exploration and Patrolling with Object Detection - InterIIT Techmeet 13.0


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
### Required python library for running the package 
vllm==0.6.4.post1

chromadb==0.5.21

open-clip-torch==2.29.0

streamlit==1.40.2

pydantic

requests

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
### 8. add the map dependencies
for hospital_world.world
```bash
cd ~/ros2_ws/src/world_setup
chmod +x setup.sh
./setup.sh
export GAZEBO_MODEL_PATH=pwd/models:pwd/fuel_models

```
ignore the .hpp warning 
## Running the Project
Once all dependencies are installed and the environment is set up, you can start the nodes using the run.sh script entre into the place where this file i s peresent and run this command:

```bash
./run.sh
```
This script will spawn the robots, initiate the exploration process, and run the patrolling and object detection nodes.

## Runing indivual nodes
if you want to run the indivual ndes folow this 
### 1 Runing Multi-bot: 
you can define initial postion in map_merge /launch/tb3_simulation/config/robot_poses.yaml  there you can change the robot poses for the particular world or else just type know_init_poses:=false
```bash
export TURTLEBOT3_MODEL=waffle
export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:/opt/ros/${ROS_DISTRO}/share/turtlebot3_gazebo/models
ros2 launch multirobot_map_merge multi_tb3_simulation_launch.py slam_gmapping:=True number_of_robots:=<no of robots>
```
### 2. Map-merging 
For merging maps from multiple robots:
```bash
ros2 launch multirobot_map_merge map_merge.launch.py
```
### 3. View map in rviz 
either you can directly open /map topic in rviz2 or run this command
```bash
ros2 launch multirobot_map_merge map_merge.launch.py
```
### 4. Exploration 
Launch the exploration node for a specific number of robots:
```bash
ros2 launch explore_lite explore_launch.py num_robots:=<no of robots>
```
### 5. Save map  
Save the map generated during exploration with a specific name:
```bash
ros2 run nav2_map_server map_saver_cli -f <map_name>
```
### 6. Camera data 
If you want to save the camera data 
```bash
ros2 launch scripts_pakages camera_launch.py
```
### 7. patroling
For patroling of robots
```bash
ros2 launch patroling patrol_launch.py map_file:=<map_name> map.yamal_file:=<map_name> num_robots:=<no of robots>
```
### 8.chromadb
To start the chromadb uvicorn server which calculates and stores OpenCLIP embeddings on the server and enables API to query the database
```bash
python server.py .chromadb/ 'clip1' --port 8000
```
```bash
access at http://192.168.124.197:8000 
```
### 9. VLM
To run vision language model, that will point to the object
```bash
vllm serve allenai/Molmo-7B-D-0924 --task generate \
  --trust-remote-code --max-model-len 4096 --limit-mm-per-prompt image=1 \
  --dtype bfloat16 --gpu-memory-utilization 0.5 --port 8081 \
```
```bash
access at http://192.168.124.197:8081
```
### 10.To run Streamlit 
t run the Streanlit
```bash
streamlit run app.py
```
streamlit app will publish positions of object to a topic which is subscribed by another node which tackels the task allocation and give goal to a agent
 ### 11. For sending a particular  robot to a target
```bash
ros2 service call /send_robot_to_object patrolling_interfaces/srv/SendRobotToObject "{robot_name: 'robot1', object_name: 'object1'}"
``` 

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

