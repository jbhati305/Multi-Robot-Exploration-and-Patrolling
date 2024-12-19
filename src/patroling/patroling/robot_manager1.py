# # patrol/robot_manager.py
# import numpy as np
# from sklearn.cluster import KMeans
# import random

# class RobotManager:
#     def __init__(self, occupancy_grid):
#         self.robots = {}  # Dictionary to store the robots
#         self.robot_centers = {}  # Dictionary to store the positions of robots
#         self.occupancy_grid = occupancy_grid
#         self.overall_labels = None
#         self.overall_centers = None
#         self.n_clusters = 0
#         self.region_label_to_robot_id = {}  # Map region labels to robot IDs

#     def launch(self):
#         self._divide_available_space()

#     def add_robot(self, robot_id, initial_state='patrolling', initial_position=None):
#         robot = Robot(id=robot_id, robot_manager=self, initial_state=initial_state, initial_position=initial_position)
#         self.robots[robot_id] = robot
#         self.robot_centers[robot_id] = robot.position
#         if initial_state == 'patrolling':
#             self.n_clusters += 1

#     def get_robot_by_id(self, robot_id):
#         return self.robots.get(robot_id, None)

#     def divide_available_space(self, occupancy_grid, n_clusters, initial_centers=None):
#         """ Apply the K-means clustering algorithm to divide the available space into n_clusters regions. """
#         coords = np.argwhere(occupancy_grid == 0)  # Find all empty spaces (0 cells)
#         if len(coords) < n_clusters:
#             raise ValueError("Number of centers exceeds available 0-cells")

#         if initial_centers is not None:
#             initial_centers_np = np.array(initial_centers)
#             kmeans_custom_init = KMeans(n_clusters=n_clusters, init=initial_centers_np, n_init=1, max_iter=300)
#             kmeans_custom_init.fit(coords)
#             centers = kmeans_custom_init.cluster_centers_
#             labels = kmeans_custom_init.labels_
#         else:
#             kmeans = KMeans(n_clusters=n_clusters, init='random', n_init=10, max_iter=300)
#             kmeans.fit(coords)
#             centers = kmeans.cluster_centers_
#             labels = kmeans.labels_

#         # Assign cells to the nearest center
#         labels_full = np.full_like(occupancy_grid, -1)

#         for idx, (r, c) in enumerate(coords):
#             labels_full[r, c] = labels[idx]

#         return centers, labels_full

#     def _divide_available_space(self):
#         new_centers = []
#         robot_ids = []
#         for robot_id, robot in self.robots.items():
#             if robot.state == 'patrolling':
#                 new_centers.append(robot.position)
#                 robot_ids.append(robot_id)

#         new_centers = np.array(new_centers)
#         # Perform clustering
#         self.overall_centers, self.overall_labels = self.divide_available_space(
#             self.occupancy_grid, self.n_clusters, initial_centers=new_centers
#         )

#         # Update the centers and regions for each robot
#         for idx, robot_id in enumerate(robot_ids):
#             robot = self.robots[robot_id]
#             robot.region_label = idx  # Assign region label
#             robot.centers = self.overall_centers[idx]
#             robot.region_indices = np.where(self.overall_labels == idx)
#             self.region_label_to_robot_id[idx] = robot_id

#     def assign_node_to_robot(self, robot_id, node):
#         robot = self.get_robot_by_id(robot_id)
#         if robot:
#             robot.add_node_to_region(node)

# class Robot:
#     def __init__(self, id, robot_manager: RobotManager, initial_state='patrolling', initial_position=None):
#         self.id = id  # Should be the robot's name
#         self.state = initial_state  # 'patrolling' or 'working'
#         self.robot_manager = robot_manager  # Reference to the RobotManager instance

#         """ TO BE KEPT UPDATED EVERYTIME"""
#         self.position = initial_position  # [row, col] in occupancy grid indices

#         """ TO BE UPDATED WHENEVER THE ROBOT CHANGES ITS STATE"""
#         self.region_label = None  # Label of the region assigned to the robot
#         self.centers = None  # Cluster center assigned to robot
#         self.region_indices = None  # Indices of the occupancy grid cells in robot's region

#         # Set to keep track of nodes assigned to the robot
#         self.nodes_in_region = set()

#     def _update_position(self, new_position=None):
#         self.position = new_position
#         self.robot_manager.robot_centers[self.id] = self.position

#     def change_state(self, new_state):
#         """Change state from patrolling to working or vice versa"""
#         if self.state != new_state:
#             self.state = new_state
#             if new_state == 'working':
#                 self.robot_manager.n_clusters -= 1
#             else:
#                 self.robot_manager.n_clusters += 1
#             self.robot_manager._divide_available_space()

#     def add_node_to_region(self, node):
#         self.nodes_in_region.add(node)




# # from PIL import Image
# # import numpy as np
# # from sklearn.cluster import SpectralClustering
# # import matplotlib.pyplot as plt
# # from scipy.spatial import Voronoi, voronoi_plot_2d
# # from sklearn.cluster import KMeans
# # import networkx as nx
# # import random

# # def pgm2occupancy(pgm_file):
# #     img = Image.open(pgm_file)
# #     img = img.convert('L')  # 'L' mode is for grayscale
# #     img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
# #     occupancy_grid = np.zeros_like(img_array, dtype=int)
# #     occupancy_grid[img_array > 0.9] = 0
# #     occupancy_grid[img_array <= 0.9] = 1
# #     return occupancy_grid

# # def expand_obstacles(occupancy_grid, pool_size=20):
# #     height, width = occupancy_grid.shape
# #     output_matrix = np.zeros_like(occupancy_grid)
    
# #     for i in range(0, height, pool_size):
# #         for j in range(0, width, pool_size):
# #             window = occupancy_grid[i:i+pool_size, j:j+pool_size]
# #             max_value = np.max(window)
# #             output_matrix[i:i+pool_size, j:j+pool_size] = max_value
    
# #     return output_matrix

# # def divide_available_space(occupancy_grid, n_clusters, initial_centers=None, plot=True):
# #     coords = np.argwhere(occupancy_grid == 0)
# #     if len(coords) < n_clusters:
# #         raise ValueError("Number of centers exceeds available 0-cells")
    
# #     if initial_centers is not None:
# #         kmeans_custom_init = KMeans(n_clusters=n_clusters, init=initial_centers, n_init=1)
# #         kmeans_custom_init.fit(coords)
# #         centers = kmeans_custom_init.cluster_centers_
    
# #     kmeans = KMeans(n_clusters=n_clusters, init='random', n_init=10)
# #     kmeans.fit(coords)
# #     centers = kmeans.cluster_centers_

# #     # Assign cells to the nearest Voronoi center
# #     labels = np.full_like(occupancy_grid, -1)

# #     # For each 0 cell, find the closest center
# #     for i, (r, c) in enumerate(coords):
# #         distances = np.linalg.norm(centers - np.array([r, c]), axis=1)
# #         nearest_center = np.argmin(distances)
# #         labels[r, c] = nearest_center

# #     if plot:
# #         plt.figure(figsize=(6, 6))
# #         plt.imshow(labels, cmap='tab20', interpolation='nearest')
# #         plt.title(f'Voronoi Tessellation: {n_clusters} Regions')
# #         plt.scatter(centers[:, 1], centers[:, 0], color='red', marker='x', s=100, label='Voronoi Centers')
# #         plt.colorbar(label='Region Label')
# #         plt.legend()
# #         plt.show()

# #     return centers, labels


# # def generate_random_coordinates(occupancy_grid, n):
# #     empty_coords = np.argwhere(occupancy_grid == 0)
# #     if len(empty_coords) < n:
# #         raise ValueError(f"Not enough empty spaces in the grid. Found {len(empty_coords)} empty cells, but need {n}.")

# #     random_indices = np.random.choice(len(empty_coords), size=n, replace=False)
# #     random_coords = empty_coords[random_indices]
# #     return random_coords

# # class RobotManager:
# #     def __init__(self, occupancy_grid):
# #         self.robots = {} # Dictionary to store the robots
# #         self.robot_centers = {}  # Dictionary to store the positions of robots
# #         self.occupancy_grid = occupancy_grid
# #         self.overall_lables = None
# #         self.overall_centers = None
# #         self.n_clusters = 0
# #         self.robot_ids = {}
    
# #     def launch(self):
# #         self._divide_available_space()

# #     def add_robot(self, robot_id, initial_state='patrolling', initial_centers=None):
# #         robot = Robot(id=robot_id, robot_manager=self, initial_state=initial_state, initial_position=initial_centers)
# #         self.robots[robot_id] = robot
# #         self.robot_centers = {robot_id: robot.position}
# #         if initial_state == 'patrolling':
# #             self.n_clusters += 1

# #     def get_robot_by_id(self, robot_id):
# #         if robot_id in self.robots:
# #             return self.robots[robot_id]
# #         return None
    
# #     def divide_available_space(self, occupancy_grid, n_clusters, initial_centers=None, plot=False):
# #         """ Apply the K-means clustering algorithm to divide the available space into n_clusters regions. """
# #         coords = np.argwhere(occupancy_grid == 0)  # Find all empty spaces (0 cells)
# #         if len(coords) < n_clusters:
# #             raise ValueError("Number of centers exceeds available 0-cells")
# #         robot.region_label
# #         if initial_centers is not None:
# #             kmeans_custom_init = KMeans(n_clusters=n_clusters, init=initial_centers, n_init=1)
# #             kmeans_custom_init.fit(coords)
# #             centers = kmeans_custom_init.cluster_centers_
# #         else:
# #             kmeans = KMeans(n_clusters=n_clusters, init='random', n_init=10)
# #             kmeans.fit(coords)
# #             centers = kmeans.cluster_centers_

# #         # Assign cells to the nearest center
# #         labels = np.full_like(occupancy_grid, -1)

# #         for i, (r, c) in enumerate(coords):
# #             distances = np.linalg.norm(centers - np.array([r, c]), axis=1)
# #             nearest_center = np.argmin(distances)
# #             labels[r, c] = nearest_center

# #         if plot:
# #             self.plot_voronoi(labels, centers)

# #         return centers, labels

# #     def _divide_available_space(self):

# #         new_centers = []
# #         for (id, robot) in self.robots.items():
# #             if robot.state == 'patrolling':
# #                 # print(robot.id, robot.position)
# #                 new_centers.append(robot.position)
# #         new_centers = np.array(new_centers)
# #         # print(new_centers)
# #         # print('\n')

# #         self.overall_centers, self.overall_lables = self.divide_available_space(self.occupancy_grid, self.n_clusters, initial_centers=new_centers)

# #         """ Update the centers and regions for each robot"""
# #         for (id, robot) in self.robots.items():
# #             print(f"Updating robot {id}")
# #             distances = np.linalg.norm(self.overall_centers - np.array(robot.position), axis=1)
# #             nearest_center = np.argmin(distances)
# #             robot.centers = nearest_center
# #             # robot.region = np.where(self.overall_lables == nearest_center)
# #             robot.region = np.column_stack(np.where(self.overall_lables == nearest_center))


# #     def plot_voronoi(self):
# #         """Plot Voronoi cells."""
# #         labels = self.overall_lables
# #         centers = self.overall_centers
# #         plt.figure(figsize=(6, 6))
# #         plt.imshow(labels, cmap='tab20', interpolation='nearest')
# #         plt.scatter(centers[:, 1], centers[:, 0], color='red', marker='x', s=100, label='Voronoi Centers')
# #         plt.colorbar(label='Region Label')
# #         plt.legend()
# #         plt.show()

# #     def display_robot_positions(self):
# #         # Display the positions of robots
# #         plt.imshow(self.occupancy_grid, cmap='gray')
# #         for (id, robot) in self.robots.items():
# #             pos = robot.position
# #             # print(pos)
# #             if pos is not None:
# #                 if robot.state == 'patrolling':
# #                     plt.scatter(pos[1], pos[0], color='blue', label=f'Robot {robot.id}')
# #                 else:
# #                     plt.scatter(pos[1], pos[0], color='red', label=f'Robot {robot.id}')
# #         plt.legend()
# #         plt.show()

# # class Robot:
# #     def __init__(self, id, robot_manager: RobotManager, initial_state='patrolling', initial_position=None):
# #         self.id = id
# #         self.state = initial_state  # 'patrolling' or 'working'
# #         self.robot_manager = robot_manager  # Reference to the RobotManager instance

# #         """ TO BE KEPT UPDATED EVERYTIME"""
# #         self.position = initial_position  # to be assigned based on clusters

# #         """ TO BE UPDATED WHENEVER THE ROBOT CHANGES ITS STATE"""
# #         self.region = None  # stores the region assigned to the robot
# #         self.centers = None  # list of cluster centers

# #         # self._assign_region() #TODO: DONT KNOW ABOUT THIS

# #     def _update_position(self, new_position=None):
# #         self.position = new_position
# #         self.robot_manager.robot_centers[self.id] = self.position

# #     def _change_state(self, new_state):
# #         """Change state from patrolling to working or vice versa"""
# #         if self.state != new_state:
# #             self.state = new_state
# #             if new_state == 'working':
# #                 self.robot_manager.n_clusters -= 1
# #             else:
# #                 self.robot_manager.n_clusters += 1
# #             self.robot_manager._divide_available_space()
        
# #     def _plot_region(self):
# #         """Plot the region assigned to the robot."""        
# #         grid_to_plot = np.copy(self.robot_manager.occupancy_grid)
        
# #         # Mark the robot's region (cells assigned to the robot) with a value that is easily visible
# #         for r, c in self.region:
# #             grid_to_plot[r, c] = -1  # Mark region cells with a value (e.g., -1 for robot region)
        
# #         # Plot the occupancy grid
# #         plt.figure(figsize=(8, 8))
# #         plt.imshow(grid_to_plot, cmap='coolwarm', interpolation='nearest')
        
# #         # Mark the robot's position on the grid (optional)
# #         plt.scatter(self.position[1], self.position[0], color='yellow', label=f'Robot {self.id} Position', zorder=5)
        
# #         # Add labels and title
# #         plt.title(f"Robot {self.id} Region on Occupancy Grid")
# #         plt.xlabel("X Coordinate")
# #         plt.ylabel("Y Coordinate")
        
# #         # Show a color bar to indicate the values in the grid
# #         plt.colorbar(label='Grid Value (2 indicates robot region)')
        
# #         # Display the plot
# #         plt.legend()
# #         plt.show()
    
# #     def _assign_graph_based_on_region(self, n_random_points):
# #         """Assign the graph based on the region assigned to the robot."""
# #         # Step 1: Randomly select n_random_points from region_cells
# #         if len(self.region) < n_random_points:
# #             raise ValueError("The number of points to select exceeds the available points in the region.")
# #         region_cells = [tuple(row) for row in self.region]
# #         random_points = random.sample(region_cells, n_random_points)
        
# #         # Step 2: Create a graph
# #         G = nx.Graph()
        
# #         # Add nodes to the graph (each random point is a node)
# #         for point in random_points:
# #             G.add_node(point)
        
# #         # Step 3: Add edges between consecutive points to form a cycle
# #         for i in range(n_random_points):
# #             G.add_edge(random_points[i], random_points[(i + 1) % n_random_points])  # Connect last node to the first
        
# #         return G, random_points

# #     def plot_graph(self, G, random_points, occupancy_grid=None):
# #         """Plot the cyclic graph using matplotlib and networkx"""
# #         # Step 1: Plot the occupancy grid
# #         plt.figure(figsize=(8, 8))
# #         plt.imshow(occupancy_grid, cmap='Greys', origin='lower')  # Plot occupancy grid in greyscale
# #         plt.title("Occupancy Grid with Cyclic Graph")

# #         # Step 2: Plot the random points from the graph
# #         random_points_set = set(random_points)  # Set for fast lookup
# #         for point in random_points:
# #             plt.plot(point[1], point[0], 'ro')  # Red points for random points in the cyclic graph
        
# #         # Step 3: Plot edges of the cyclic graph
# #         for i in range(len(random_points)):
# #             p1 = random_points[i]
# #             p2 = random_points[(i + 1) % len(random_points)]  # Connect last point to the first
# #             plt.plot([p1[1], p2[1]], [p1[0], p2[0]], 'r-', lw=2)  # Red lines for edges
        
# #         plt.grid(True)
# #         plt.show()
    
    


# # if __name__ == "__main__":
# #     pgm_file = 'map_name.pgm'
# #     occupancy_grid = pgm2occupancy(pgm_file)
# #     # occupancy_grid = expand_obstacles(occupancy_grid)

# #     # divide_available_space(occupancy_grid, 10)

# #     # random_coords = generate_random_coordinates(occupancy_grid, 10)
# #     # divide_available_space(occupancy_grid, 10, initial_centers=random_coords)

# #     robot_manager = RobotManager(occupancy_grid)

# #     robot_manager.add_robot(0, 'patrolling')
# #     robot_manager.robots[0]._update_position([600, 500])

# #     robot_manager.add_robot(1, 'patrolling')
# #     robot_manager.robots[1]._update_position([700, 650])
    
# #     robot_manager.add_robot(2, 'patrolling')
# #     robot_manager.robots[2]._update_position([800, 650])

# #     robot_manager.add_robot(3, 'working')
# #     robot_manager.robots[3]._update_position([800, 750])

# #     robot_manager.launch()
# #     # robot_manager.robots[0]._plot_region()
# #     # robot_manager.display_robot_positions()

# #     n_random_points = 10
# #     graph, random_points = robot_manager.robots[0]._assign_graph_based_on_region(n_random_points)
# #     robot_manager.robots[0].plot_graph(graph, random_points, occupancy_grid)

# #     # robot_manager.plot_voronoi()
# #     robot_manager.robots[0]._change_state('working')
# #     # robot_manager.plot_voronoi()
    
# #     # print(robot_manager.robots[0].region)
# #     # Display initial robot positions

# #     # Change state of a robot (e.g., Robot 2 starts patrolling)
# #     # robot_manager.change_robot_state(2, 'patrolling')
    
# #     # Display updated robot positions
# #     # robot_manager.display_robot_positions()

# patroling/robot_manager.py

import numpy as np
from sklearn.cluster import KMeans
import random
from icecream import ic

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

        # Map parameters for coordinate transformations
        self.resolution = 1.0  # Default value; to be set via set_map_parameters
        self.origin_x = 0.0
        self.origin_y = 0.0
        self.height = occupancy_grid.shape[0]

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
        self._divide_available_space()

    def get_robot_by_id(self, robot_id):
        return self.robots.get(robot_id, None)

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

        # Map region labels to robot IDs
        self.region_label_to_robot_id = {idx: robot_id for idx, robot_id in enumerate(self.robots) if self.robots[robot_id].state == 'patrolling'}

        # Assign monitoring points to robots based on regions
        self.assign_monitoring_points_to_robots()

    def _apply_kmeans_clustering(self, initial_centers):
        """
        Applies K-means clustering to divide the occupancy grid into regions.
        :param initial_centers: Array of initial cluster centers in grid coordinates
        :return: centers, labels_full
        """
        n_clusters = len(initial_centers)
        coords = np.argwhere(self.occupancy_grid == 0)  # Free cells
        if len(coords) < n_clusters:
            raise ValueError("Number of clusters exceeds available free cells.")

        # Convert initial centers to the closest free cell
        initial_centers_grid = []
        for center in initial_centers:
            # Find the nearest free cell
            distances = np.linalg.norm(coords - center, axis=1)
            nearest_idx = np.argmin(distances)
            initial_centers_grid.append(coords[nearest_idx])

        initial_centers_grid = np.array(initial_centers_grid)

        # Apply K-means
        kmeans = KMeans(n_clusters=n_clusters, init=initial_centers_grid, n_init=1)
        kmeans.fit(coords)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_

        # Assign labels to the full grid
        labels_full = np.full(self.occupancy_grid.shape, -1, dtype=int)
        for idx, (r, c) in enumerate(coords):
            labels_full[r, c] = labels[idx]

        return centers, labels_full

    def assign_monitoring_points_to_robots(self):
        """
        Asset_map_parameterssign monitoring points to robots based on their assigned regions.
        """
        for robot_id, robot in self.robots.items():
            if robot.state != 'patrolling':
                continue  # Skip non-patrolling robots

            # Find monitoring points within the robot's region
            region_label = self._get_robot_region_label(robot_id)
            if region_label is None:
                if self.logger:
                    self.logger.warning(f"No region assigned to robot {robot_id}")
                continue

            # Assign points that fall within the robot's region
            assigned_points = [point for point in self.monitoring_points if self._is_point_in_region(point, region_label)]
            robot.assigned_points = assigned_points.copy()

            if self.logger:
                self.logger.info(f"Assigned {len(assigned_points)} points to robot {robot_id}")

    def _get_robot_region_label(self, robot_id):
        """
        Retrieves the region label assigned to a robot.
        """
        for label, rid in self.region_label_to_robot_id.items():
            if rid == robot_id:
                return label
        return None

    def _is_point_in_region(self, point, region_label):
        """
        Checks if a given world coordinate point is within a specific region.
        :param point: Tuple (x, y) in world coordinates
        :param region_label: Integer label of the region
        :return: Boolean
        """
        x_world, y_world = point
        col = int((x_world - self.origin_x) / self.resolution)
        row = int((y_world - self.origin_y) / self.resolution)

        # Adjust row based on grid orientation
        row = self.height - row

        if 0 <= row < self.overall_labels.shape[0] and 0 <= col < self.overall_labels.shape[1]:
            return self.overall_labels[row, col] == region_label
        return False

    def set_map_parameters(self, resolution, origin, height):
        """
        Sets map parameters required for coordinate transformations.
        :param resolution: float, meters per grid cell
        :param origin: tuple (x_origin, y_origin, z_origin)
        :param height: int, number of rows in the grid
        """
        self.resolution = resolution
        self.origin_x = origin[0]
        self.origin_y = origin[1]
        self.height = height

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
            self.robot_manager._divide_available_space()

    # def assign_monitoring_points(self, points):
    #     """
    #     Assigns a list of monitoring points to the robot.
    #     :param points: List of tuples (x, y) in world coordinates
    #     """
    #     self.assigned_points = points.copy()

    def get_next_monitoring_point(self):
        """
        Retrieves the next monitoring point for the robot.
        :return: Tuple (x, y) or None
        """
        if not self.assigned_points:
            return None
        return self.assigned_points.pop(0)  # FIFO assignment
