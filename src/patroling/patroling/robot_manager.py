import numpy as np
from sklearn.cluster import KMeans
import random
from icecream import ic
from .points_to_monitor import Atransform

class RobotManager:
    def __init__(self, occupancy_grid, monitoring_points, logger=None):
        self.robots = {}  # Dictionary to store Robot instances
        self.occupancy_grid = occupancy_grid
        self.monitoring_points = monitoring_points  # List of tuples (x, y) in world coordinates
        self.overall_labels = None  # 2D array with region labels
        self.overall_centers = None  # Centers of the regions in grid coordinates
        self.n_clusters = 0  # Number of active patrolling robots
        self.region_label_to_robot_id = {}  # Map region labels to robot IDs
        self.logger = logger  # Optional logger


    def launch(self):
        """Divide available space into regions based on active patrolling robots."""
        self._divide_available_space()

    def add_robot(self, robot_id, initial_state='patrolling', initial_position=None):
        """
        Adds a robot to the manager.
        :param robot_id: Unique identifier for the robot (string)
        :param initial_state: 'patrolling' or 'working'
        :param initial_position: [row, col] in occupancy grid indices
        """
        robot = Robot(id=robot_id, robot_manager=self, initial_state=initial_state, initial_position=initial_position)
        self.robots[robot_id] = robot
        if initial_state == 'patrolling':
            self.n_clusters += 1
        # self._divide_available_space()

    def get_robot_by_id(self, robot_id):
        return self.robots.get(robot_id, None)

    # def _divide_available_space(self):
    #     """Divide the available space into regions using K-means clustering."""

    #     # Collect initial positions of patrolling robots in grid coordinates
    #     initial_positions = [robot.position for robot in self.robots.values() if robot.state == 'patrolling']

    #     if not initial_positions:
    #         self.overall_labels = np.full(self.occupancy_grid.shape, -1, dtype=int)
    #         self.overall_centers = []
    #         self.region_label_to_robot_id = {}
    #         return

    #     initial_centers = np.array(initial_positions)

    #     # Apply K-means clustering on free cells
    #     self.overall_centers, self.overall_labels = self._apply_kmeans_clustering(initial_centers)
    #     print(f"Overall centers: {self.overall_centers}")
    #     print(f"Overall labels: {np.unique(self.overall_labels)}")
    #     unique_labels = np.unique(self.overall_labels)
    #     # Get unique labels excluding -1 (which represents obstacles/unknown space)  
    #     # unique_labels = np.unique(self.overall_labels[self.overall_labels != -1])
    #     # Map region labels to robot IDs
    #     for idx, label in enumerate(np.unique(self.overall_labels)):
    #         if(label != -1):
    #             self.region_label_to_robot_id[label] = None

    #     self.region_label_to_robot_id = {}
    #     idx = 1
    #     for robot_id, robot in self.robots.items():
    #         if(robot.state == 'patrolling' and robot.position is not None):
    #             self.region_label_to_robot_id[unique_labels[idx]] = robot_id
    #             idx += 1

    #     self.robot_id_to_label = {v: k for k, v in self.region_label_to_robot_id.items()}
        
    #     # Create a mapping between region labels and robot IDs
    #     # self.region_label_to_robot_id = {}

    #     # for robot_id, robot in self.robots.items():
    #     #     print(robot_id)
    #     #     if robot.state == 'patrolling' and robot.position is not None:
    #     #         # Compute distance from robot to each center
    #     #         distances = np.linalg.norm(self.overall_centers - np.array(robot.position), axis=1)
    #     #         closest_center_idx = np.argmin(distances)
            
    #     #         # Assign the closest center's label to the robot
    #     #         self.region_label_to_robot_id[closest_center_idx.item()] = robot_id  # Store the region for the robot
        
    #     print(f"Region label to robot mapping: {self.region_label_to_robot_id}")
    #     print(f"Robot ID to region label mapping: {self.robot_id_to_label}")

    #     # Assign monitoring points to robots based on regions
    #     self.assign_monitoring_points_to_robots()

    def _divide_available_space(self):  
        """Divide the available space into regions using K-means clustering."""  

        # Collect initial positions of patrolling robots in grid coordinates  
        initial_positions = [robot.position for robot in self.robots.values() if robot.state == 'patrolling']  
        if not initial_positions:  
            self.overall_labels = np.full(self.occupancy_grid.shape, -1, dtype=int)  
            self.overall_centers = []  
            self.region_label_to_robot_id = {}  
            return  

        initial_centers = np.array(initial_positions)  
        # Apply K-means clustering on free cells  
        self.overall_centers, self.overall_labels = self._apply_kmeans_clustering(initial_centers)  
        print(f"Overall centers: {self.overall_centers}")  
        print(f"Overall labels: {np.unique(self.overall_labels)}")  
        unique_labels = np.unique(self.overall_labels)  

        # Exclude the -1 label  
        region_labels = [label for label in unique_labels if label != -1]  

        # Collect patrolling robots  
        patrolling_robots = [robot_id for robot_id, robot in self.robots.items() if robot.state == 'patrolling' and robot.position is not None]  

        if len(region_labels) != len(patrolling_robots):  
            if self.logger:  
                self.logger.warning("The number of regions does not match the number of patrolling robots.")  
            # Handle this case appropriately, maybe adjust K-means or skip assignment  
            return  

        # Map region labels to robot IDs  
        self.region_label_to_robot_id = {}  
        for label, robot_id in zip(region_labels, patrolling_robots):  
            self.region_label_to_robot_id[label] = robot_id  

        # Create reverse mapping  
        self.robot_id_to_label = {robot_id: label for label, robot_id in self.region_label_to_robot_id.items()}  

        print(f"Region label to robot mapping: {self.region_label_to_robot_id}")  
        print(f"Robot ID to region label mapping: {self.robot_id_to_label}")  

        # Assign monitoring points to robots based on regions  
        self.assign_monitoring_points_to_robots()  
    def _apply_kmeans_clustering(self, initial_centers):
        """
        Applies K-means clustering to divide the occupancy grid into regions.
        :param initial_centers: Array of initial cluster centers in grid coordinates
        :return: centers, labels_full
        """
        n_clusters = self.n_clusters
        coords = np.argwhere(self.occupancy_grid == 0)  # Free cells
        if len(coords) < n_clusters:
            raise ValueError("Number of clusters exceeds available free cells.")


        initial_centers_grid = np.array(initial_centers)

        # Apply K-means
        kmeans = KMeans(n_clusters=n_clusters, init=initial_centers_grid, n_init=1)
        kmeans.fit(coords)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_

        # Assign labels to the full grid
        labels_full = np.full(self.occupancy_grid.shape, -1, dtype=int)
        unique_labels = np.unique(labels)
        print(f"Unique labels: {unique_labels}")
        for idx, (r, c) in enumerate(coords):
            labels_full[r, c] = labels[idx]

        return centers, labels_full
    
    def assign_monitoring_points_to_robots(self):  
        """  
        Assign monitoring points to robots based on their assigned regions.  
        Now also handles reassignment when points are depleted.  
        """  
        # Clear existing assignments first  
        for robot in self.robots.values():  
            if robot.state == 'patrolling':  
                robot.assigned_points = []  

        for robot_id, robot in self.robots.items():  
            if robot.state != 'patrolling':  
                continue  

            region_label = self.robot_id_to_label.get(robot_id)  
            if region_label is None:  
                if self.logger:  
                    self.logger.warning(f"No region assigned to robot {robot_id}")  
                continue  

            # Assign points that fall within the robot's region  
            assigned_points = [point for point in self.monitoring_points   
                             if self._is_point_in_region(point, region_label)]  
            robot.assigned_points = assigned_points.copy()  

            if self.logger:  
                self.logger.info(f"Reassigned {len(assigned_points)} points to robot {robot_id}")  

    def _is_point_in_region(self, point, region_label):
        """
        Checks if a given world coordinate point is within a specific region.
        :param point: Tuple (x, y) in world coordinates
        :param region_label: Integer label of the region
        :return: Boolean
        """
        x, y = point
        if 0 <= x < self.overall_labels.shape[0] and 0 <= y < self.overall_labels.shape[1]:
            return self.overall_labels[x, y] == region_label
        return False


class Robot:
    def __init__(self, id, robot_manager, initial_state='patrolling', initial_position=None):
        """
        Initializes a Robot instance.
        :param id: Unique identifier for the robot
        :param robot_manager: Reference to RobotManager
        :param initial_state: 'patrolling' or 'working'
        :param initial_position: [row, col] in occupancy grid indices
        """
        self.id = id
        self.state = initial_state
        self.robot_manager = robot_manager
        self.current_goal=None
        self.position = initial_position  # [row, col]
        self.assigned_points = []  # List of tuples (x, y) in world coordinates

    def change_state(self, new_state):
        """
        Changes the robot's state and notifies the manager to re-divide space.
        :param new_state: 'patrolling' or 'working'
        """
        if self.state != new_state:
            self.state = new_state
            if new_state == 'patrolling':
                self.robot_manager.n_clusters += 1
            elif new_state == 'working':
                self.robot_manager.n_clusters -= 1
            # self.robot_manager._divide_available_space()

    def get_next_monitoring_point(self):    
        """    
        Retrieves a random monitoring point for the robot and removes it from assigned points.  
        Recalculates assigned points if empty.  
        :return: Tuple (x, y) or None    
        """    
        if  len(self.assigned_points)==1:  
            # Recalculate assigned points  
            self.robot_manager.assign_monitoring_points_to_robots()  

            # Check again if we got new points  
            if not self.assigned_points:  
                return None  
        ic(self.id)
        # Randomly select a point  
        random_index = random.randrange(len(self.assigned_points))  
        random_point = self.assigned_points[random_index]  

        # Remove the point using index instead of value  
        self.assigned_points.pop(random_index)  

        return random_point  
