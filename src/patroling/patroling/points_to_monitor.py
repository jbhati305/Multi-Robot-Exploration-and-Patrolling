# import matplotlib.pyplot as plt
# import numpy as np
# from PIL import Image
# from scipy.ndimage import distance_transform_edt
# import random
# import cv2

# def pgm2occupancy(pgm_file):
#     img = Image.open(pgm_file)
#     img = img.convert('L')  # 'L' mode is for grayscale
#     img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
#     occupancy_grid = np.zeros_like(img_array, dtype=int)
#     occupancy_grid[img_array > 0.9] = 0
#     occupancy_grid[img_array <= 0.9] = 1
#     return occupancy_grid


# # def get_random_points_to_monitor_near_obstacles(occupancy_grid, num_points=100, plot=False):

# #     # Erosion kernel
# #     kernel = np.ones((3, 3), np.uint8)

# #     # Dilation to add a buffer zone around the boundary
# #     dilated_grid = cv2.dilate(occupancy_grid.astype(np.uint8), kernel)

# #     # Extract boundary points with the buffer zone
# #     boundary_with_buffer = dilated_grid - occupancy_grid
# #     boundary_buffer_coords = np.column_stack(np.where(boundary_with_buffer))
# #     print(boundary_buffer_coords)

# #     # Sample n random points uniformly from the boundary points
# #     if len(boundary_buffer_coords) >= num_points:
# #         sampled_indices = np.random.choice(len(boundary_buffer_coords), size=num_points, replace=False)
# #         sampled_points = boundary_buffer_coords[sampled_indices]
# #     else:
# #         sampled_points = boundary_buffer_coords  # Return all points if n > available points

# #     # Plotting the grid and the boundary points
# #     if plot:
# #         plt.imshow(occupancy_grid, cmap='Greys', origin='upper')
# #         plt.scatter(sampled_points[:, 1], sampled_points[:, 0], color='red', s=10)
# #         plt.title("Boundary Points with Buffer Zone")
# #         plt.show()

# #     return sampled_points

# def dfs_boundary(coords, start_idx, visited, boundary_points, coords_set):
#     # Stack for DFS traversal
#     stack = [start_idx]
#     directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 4 possible movement directions (Up, Down, Left, Right)
#     boundary_points.append(coords[start_idx])
#     visited[start_idx] = True

#     while stack:
#         current_idx = stack.pop()
#         x, y = coords[current_idx]

#         # Check all 4 possible directions
#         for dx, dy in directions:
#             nx, ny = x + dx, y + dy
#             if (nx, ny) in coords_set and not visited[coords_set[(nx, ny)]]:
#                 new_idx = coords_set[(nx, ny)]
#                 visited[new_idx] = True
#                 stack.append(new_idx)
#                 boundary_points.append(coords[new_idx])
    
# def get_random_points_to_monitor_near_obstacles2(occupancy_grid, plot=False):
#     # Erosion kernel
#     kernel1 = np.ones((40, 40), np.uint8)
#     kernel2 = np.ones((20, 20), np.uint8)

#     # Dilation to add a buffer zone around the boundary
#     dilated_grid1 = cv2.dilate(occupancy_grid.astype(np.uint8), kernel1)
#     dilated_grid2 = cv2.dilate(occupancy_grid.astype(np.uint8), kernel2)

#     # Extract boundary points with the buffer zone
#     boundary_with_buffer1 = dilated_grid1 - occupancy_grid
#     boundary_with_buffer2 = dilated_grid2 - occupancy_grid

#     boundary_buffer_coords = np.column_stack(np.where(boundary_with_buffer1 - boundary_with_buffer2))
    
#     print(f"Found {len(boundary_buffer_coords)} boundary points")

#     # Create a dictionary to map coordinates to their index for easy lookup
#     coords_set = {tuple(point): idx for idx, point in enumerate(boundary_buffer_coords)}

#     visited = np.zeros(len(boundary_buffer_coords), dtype=bool)  # To keep track of visited points
#     all_sampled_points = []  # List to store the uniformly sampled points for all objects

#     # Start DFS for each unvisited point
#     for i in range(len(boundary_buffer_coords)):
#         if not visited[i]:
#             boundary_points = []
#             # Perform DFS from this unvisited point
#             dfs_boundary(boundary_buffer_coords, i, visited, boundary_points, coords_set)

#             # Sample 20% of the boundary points uniformly from the boundary_points list
#             num_to_sample = int(len(boundary_points) * 0.005)
#             sampled_indices = np.linspace(0, len(boundary_points) - 1, num=num_to_sample, dtype=int)
#             sampled_points = np.array([boundary_points[idx] for idx in sampled_indices])

#             # Add these sampled points to the final list
#             all_sampled_points.extend(sampled_points)

#     all_sampled_points = np.array(all_sampled_points)

#     # Plotting the grid and the sampled points
#     if plot:
#         plt.imshow(occupancy_grid, cmap='Greys', origin='upper')
#         plt.scatter(all_sampled_points[:, 1], all_sampled_points[:, 0], color='red', s=10)
#         plt.scatter(200, 600, color='blue', s=10)
#         plt.title("Boundary Points with Uniform Sampling")
#         plt.show()

#     return all_sampled_points

# if __name__ == "__main__":
#     pgm_file = "my_map.pgm"
#     occupancy_grid = pgm2occupancy(pgm_file)
#     # plt.imshow(occupancy_grid, cmap='gray')

#     sampled_points = get_random_points_to_monitor_near_obstacles2(occupancy_grid, plot=True)
#     print(len(sampled_points))

# # Tasks:
# # 1. To define the points where the teh robots should patrol
# # 2. Learn these points based on the amount of deviation from the previous in images.
# # 3. Basically remove the points that get no change in images and add the points near the points whose images have a change.
# # 4. How to define changes in the images?


# patroling/points_to_monitor.py

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import random
import cv2
from icecream import ic

def pgm2occupancy(pgm_file):
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

def dfs_boundary(coords, start_idx, visited, boundary_points, coords_set):
    """
    Depth-First Search to traverse connected boundary points.
    """
    stack = [start_idx]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 4-connected grid
    boundary_points.append(coords[start_idx])
    visited[start_idx] = True

    while stack:
        current_idx = stack.pop()
        x, y = coords[current_idx]

        # Explore all 4 directions
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            neighbor = (nx, ny)
            if neighbor in coords_set:
                neighbor_idx = coords_set[neighbor]
                if not visited[neighbor_idx]:
                    visited[neighbor_idx] = True
                    stack.append(neighbor_idx)
                    boundary_points.append(coords[neighbor_idx])

def Atransform(occu_points, shape, logger=None):
    height, width = shape
    occu_points = np.array(occu_points)  # Ensure input is a NumPy array
    
    if occu_points.size == 0:
        if logger:
            logger.warn("Empty occu_points array passed to Atransform.")
        return np.array([[0,0]])  # Return an empty array if input is empty
    
    if occu_points.ndim == 1:
        # Handle single point
        transformed_points = np.array([occu_points[1], height - occu_points[0]])
    elif occu_points.ndim == 2 and occu_points.shape[1] == 2:
        # Handle multiple points
        transformed_points = np.copy(occu_points)
        transformed_points[:, 1] = height - occu_points[:, 0]
        transformed_points[:, 0] = occu_points[:, 1]
    else:
        if logger:
            logger.error("Invalid shape for occu_points. Expected 1D or Nx2 array.")
        raise ValueError("Invalid shape for occu_points. Expected 1D or Nx2 array.")
    return transformed_points


def InverseAtransform(world_points, shape, OffsetX, OffsetY, resolution):
    height, width = shape
    occu_points = np.copy(world_points)
    if occu_points.ndim == 1:
        occu_points = occu_points.reshape(1, -1)
    occu_points[:, 1] = (world_points[:, 0] - OffsetX) / resolution
    occu_points[:, 0] = height - (world_points[:, 1] - OffsetY) / resolution
    return np.array(occu_points, dtype=int)


def get_random_points_to_monitor_near_obstacles2(occupancy_grid, plot=False):
    """
    Generates monitoring points near obstacles by sampling boundary points.
    """

    # Define erosion kernels to create buffer zones
    kernel1 = np.ones((70, 70), np.uint8)
    kernel2 = np.ones((40, 40), np.uint8)

    # Dilate the occupancy grid to create buffer zones around obstacles
    dilated_grid1 = cv2.dilate(occupancy_grid.astype(np.uint8), kernel1)
    dilated_grid2 = cv2.dilate(occupancy_grid.astype(np.uint8), kernel2)

    # Extract boundary points with buffer zones
    boundary_with_buffer1 = dilated_grid1 - occupancy_grid
    boundary_with_buffer2 = dilated_grid2 - occupancy_grid

    # Combine boundary buffers
    boundary_buffer = boundary_with_buffer1 - boundary_with_buffer2
    boundary_buffer_coords = np.column_stack(np.where(boundary_buffer))

    print(f"Found {len(boundary_buffer_coords)} boundary points")

    # Create a dictionary to map coordinates to their index for easy lookup
    coords_set = {tuple(point): idx for idx, point in enumerate(boundary_buffer_coords)}
    
    visited = np.zeros(len(boundary_buffer_coords), dtype=bool)  # Track visited points
    all_sampled_points = []  # Store sampled points

    # Traverse each unvisited boundary point using DFS and sample uniformly
    for i in range(len(boundary_buffer_coords)):
        if not visited[i]:
            boundary_points = []
            dfs_boundary(boundary_buffer_coords, i, visited, boundary_points, coords_set)

            # Sample 0.5% of the boundary points uniformly
            sampling_ratio = 0.004 # 0.5%
            num_to_sample = max(int(len(boundary_points) * sampling_ratio), 1)
            sampled_indices = np.linspace(0, len(boundary_points) - 1, num=num_to_sample, dtype=int)
            sampled_points = np.array([boundary_points[idx] for idx in sampled_indices])

            # Add sampled points to the final list
            all_sampled_points.extend(sampled_points)

    all_sampled_points = np.array(all_sampled_points)
    print(f"Found {all_sampled_points} boundary points")
    
    # Optional: Plot the monitoring points
    # if plot:
    #     plt.figure(figsize=(10, )their index for easy look
    return all_sampled_points

# if __name__ == "__main__":
#     pgm_file = "my_map.pgm"  # Replace with your map file path
#     occupancy_grid = pgm2occupancy(pgm_file)
#     sampled_points = get_random_points_to_monitor_near_obstacles2(occupancy_grid, plot=True)
#     print(f"Total sampled points: {len(sampled_points)}")

