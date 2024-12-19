import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry  # Import Odometry message
from functools import partial
from std_msgs import msg
import numpy as np
from sklearn.neighbors import KDTree
import heapq
from std_msgs.msg import String
from PIL import Image
from collections import deque
# Other existing imports...
from patrolling_interfaces.srv import SendRobotToObject  # Adjusted import path
from icecream import ic
from .points_to_monitor import InverseAtransform
import yaml

class NearestRobot(Node):

    def __init__(self):
        super().__init__('nearest_robot')

        # Declare parameters
        self.declare_parameter('robot_names', ['robot1'])
        self.declare_parameter('world_frame', 'map')
        self.declare_parameter('status_check_interval', 2.0)  # in seconds
        self.declare_parameter('map_yaml_file', 'config/map.yaml')
        self.declare_parameter('map_file', 'config/map.pgm')
        self.declare_parameter('objects_file', 'config/objects.yaml')  # Parameter for objects file

        # Get parameters
        self.robot_names = self.get_parameter('robot_names').get_parameter_value().string_array_value
        self.world_frame = self.get_parameter('world_frame').get_parameter_value().string_value
        # self.status_check_interval = self.get_parameter('status_check_interval').get_parameter_value().double_value
        self.map_yaml_file = self.get_parameter('map_yaml_file').get_parameter_value().string_value
        self.map_file = self.get_parameter('map_file').get_parameter_value().string_value
        self.objects_file = self.get_parameter('objects_file').get_parameter_value().string_value  # Get objects file
        # self.send_robot_client = self.create_client(SendRobotToObject, 'send_robot_to_object')  # Create client
        # self.create_service(SendRobotToObject, 'send_robot_to_object', self.send_robot_callback)  # Create service
        # Dictionary to hold latest positions
        self.latest_positions = {robot: None for robot in self.robot_names}
        # Load the map metadata
        with open(self.map_yaml_file, 'r') as yaml_file:
            map_yaml = yaml.safe_load(yaml_file)
        self.resolution = map_yaml['resolution']
        self.origin = map_yaml['origin']


        # Create subscribers for each robot's odom
        self.odom_subscribers = {}
        for robot in self.robot_names:
            topic_name = f'/{robot}/odom'
            self.odom_subscribers[robot] = self.create_subscription(
                Odometry,
                topic_name,
                partial(self.odom_callback, robot_name=robot),
                10  # QoS history depth
            )
            self.get_logger().info(f'Subscribed to {topic_name}')

        # Optionally, set up a timer to periodically print the latest positions
        # self.create_timer(5.0, self.print_latest_positions)  # Every 5 seconds


        self.image_sub = self.create_subscription(String,"coordinate_data",self.listener_callback,10)
        self.occupancy_grid = self.pgm2occupancy(self.map_file)

    def odom_callback(self, msg: Odometry, robot_name: str):
        """
        Callback function for odometry messages.
        Stores the latest position of the robot.
        """
        position = msg.pose.pose.position
        self.latest_positions[robot_name] = position
        # self.get_logger().debug(
        #     f'Received odom for {robot_name}: Position -> x: {position.x}, y: {position.y}, z: {position.z}'
        # )

    def print_latest_positions(self):
        """
        Prints the latest known positions of all robots.
        """
        for robot, position in self.latest_positions.items():
            if position:
                # self.get_logger().info(
                #     f'{robot} Position -> x: {position.x:.2f}, y: {position.y:.2f}, z: {position.z:.2f}'
                # )
                pass
            else:
                self.get_logger().info(f'{robot} Position -> Not received yet.')
    def world_to_grid(self, world_point):
        """
        Converts world coordinates to grid indices.
        :param world_point: [x, y] in world coordinates
        :return: [row, col] in grid indices
        """
        x, y = world_point
        col = int((x - self.origin[0]) / self.resolution)
        row = int((y - self.origin[1]) / self.resolution)
        return [row, col]
    def listener_callback(self, msg):  
        ic(msg)  
        import json    
        coord_data = json.loads(msg.data)    
    
        # Extract coordinates from the first object (using string key '0')  
        first_object = coord_data['0']  # Get the first object using string key  
        first_point = first_object['points'][0]  # Get the first point    
    
        # Extract x and y coordinates    
        x = first_point['pose_x']  
        y = first_point['pose_y']  
    
        point = (x, y)  # Create tuple of coordinates    
        new_point = {'x': x, 'y': y}  # Create a new point dictionary  
        self.get_logger().info(f'Extracted coordinates: x={x}, y={y}')    
    
    # Rest of the function remains the same...    
        with open("/home/user1/ros2_ws/src/patroling/config/objects.yaml", 'r') as file:
            data = yaml.safe_load(file)

        # Find the next object key
        existing_objects = data.get("objects", {})
        new_object_key = f"object{len(existing_objects) + 1}"

        # Add the new point under the new key
        data['objects'][new_object_key] = new_point

        # Write back to the YAML file
        with open("/home/user1/ros2_ws/src/patroling/config/objects.yaml", 'w') as file:
            yaml.safe_dump(data, file)
        robots = []
        for robot, position in self.latest_positions.items():
            if position:
                #ic(position, "position")
                robo_point = InverseAtransform(
                    np.array([position.x, position.y]).reshape(1, -1),
                    self.occupancy_grid.shape,
                    self.origin[0],
                    self.origin[1],
                    self.resolution
                )
                robots.append((robot, robo_point[0]))  # Store robot name with position

        #ic(point)
        point = np.array(point).reshape(1, -1)
        point = InverseAtransform(
            np.array(point),
            self.occupancy_grid.shape,
            self.origin[0],
            self.origin[1],
            self.resolution
        )
        #ic(point)

        nearest_robot = self.closest_robot_with_obstacles(
            [robo[1] for robo in robots], point[0], self.occupancy_grid
        )

        # if nearest_robot:
        #     # Find the robot name corresponding to the nearest position
        #     nearest_robot_name = next(
        #         (name for name, pos in robots if np.array_equal(pos, nearest_robot)),
        #         None
        #     )
        #     if nearest_robot_name:
        #         self.get_logger().info(f"Sending {nearest_robot_name} to object...")
        #         request = SendRobotToObject.Request()
        #         request.robot_name = nearest_robot_name  # Use nearest robot's name
        #         request.object_name = new_object_key  # Replace with the actual object name if needed
                
        #         # Make the service call
        #         future = self.send_robot_client.call_async(request)
        #         future.add_done_callback(self.service_response_callback)

    def service_response_callback(self, future):
        try:
            response = future.result()
            self.get_logger().info(f"Service call result: {response.success}")
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")

    def pgm2occupancy(self, pgm_file):
        """
        Converts a PGM file to an occupancy grid.
        Free space: 0
        Obstacles: 1
        """
        img = Image.open(pgm_file)
        img = img.convert('L')  # 'L' mode is for grayscale
        img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
        occupancy_grid = np.zeros_like(img_array, dtype=int)
        occupancy_grid[img_array > 0.9] = 0  # Free space
        occupancy_grid[img_array <= 0.9] = 1  # Obstacles
        return occupancy_grid

    # Define BFS for multi-source pathfinding
    def bfs_multi_source(self, grid, start_positions, goal):
        """ Multi-source BFS to find the shortest path from any robot to the target point """
        
        # Directions: up, down, left, right
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        # Queue for BFS (stores (x, y, cost))
        queue = deque([(start[0], start[1], 0) for start in start_positions])
        # start_positions=[list(arr) for arr in start_positions]
        # Set to track visited cells
        visited = set(start_positions)
        
        # While the queue is not empty, expand nodes
        while queue:
            x, y, cost = queue.popleft()
            
            # If we reached the goal
            if (x, y) == goal:
                return cost  # Return the length of the path (cost)
            
            # Explore neighbors
            for direction in directions:
                nx, ny = x + direction[0], y + direction[1]
                
                # Check if within bounds and not an obstacle
                if (0 <= nx < len(grid) and 0 <= ny < len(grid[0]) and
                    (nx, ny) not in visited and grid[nx][ny] == 0):  # free space
                    visited.add((nx, ny))
                    queue.append((nx, ny, cost + 1))
        
        return None  # No path found

    # Multi-source BFS to find the closest robot
    def closest_robot_with_obstacles(self, robots, point, grid):
        """ Use multi-source BFS to find the closest robot to the target point """
        # Filter robots that are valid positions (free spaces)
        #ic(robots)
        valid_robots = [(robot[0],robot[1]) for robot in robots if grid[robot[0], robot[1]] == 0]
        
        # Call BFS to find the shortest path length from multiple robots to the point
        min_path_length = float('inf')
        closest_robot = None
        
        # Multi-source BFS for all valid robots
        path_length = self.bfs_multi_source(grid, valid_robots, tuple(point))
        
        if path_length is not None and path_length < min_path_length:
            min_path_length = path_length
            closest_robot = valid_robots[0]  # In this case, take the first robot
        
        return closest_robot

        


def main(args=None):
    rclpy.init(args=args)
    node = NearestRobot()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('GoalAssignerNode has been stopped.')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
