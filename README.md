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


