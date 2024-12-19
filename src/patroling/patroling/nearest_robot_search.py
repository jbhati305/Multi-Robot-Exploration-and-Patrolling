import numpy as np
from sklearn.neighbors import KDTree
import heapq

# Define A* pathfinding algorithm
def a_star(grid, start, goal):
    """ A* search algorithm to find the shortest path while avoiding obstacles """
    
    # Directions: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    def heuristic(a, b):
        """ Manhattan distance heuristic """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    # Priority queue for A* (stores (cost, (x, y), g_cost))
    open_list = []
    heapq.heappush(open_list, (0 + heuristic(start, goal), 0, start))
    
    # Maps to store the shortest path and g_cost
    came_from = {}
    g_cost = {start: 0}
    
    while open_list:
        _, cost, current = heapq.heappop(open_list)
        
        # If we reached the goal
        if current == goal:
            # Reconstruct the path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]
        
        # Explore neighbors
        for direction in directions:
            neighbor = (current[0] + direction[0], current[1] + direction[1])
            
            # Check if within bounds and not an obstacle
            if (0 <= neighbor[0] < len(grid) and 0 <= neighbor[1] < len(grid[0]) and
                grid[neighbor[0]][neighbor[1]] == 0):  # free space
                new_cost = cost + 1
                if neighbor not in g_cost or new_cost < g_cost[neighbor]:
                    g_cost[neighbor] = new_cost
                    priority = new_cost + heuristic(neighbor, goal)
                    heapq.heappush(open_list, (priority, new_cost, neighbor))
                    came_from[neighbor] = current
    
    return None  # No path found

# KD-Tree-based approach to find closest robot
def closest_robot_with_obstacles(robots, point, grid, k=4):
    """ Find the closest robot to a given point while avoiding obstacles """
    # Convert robot positions to a numpy array
    robot_positions = np.array(robots)
    
    # Create a KD-Tree from the robot positions
    tree = KDTree(robot_positions)
    
    # Query the nearest robot (assuming we're looking for the closest robot to the point)
    dist, index = tree.query([point], k=k)  # Query the 10 closest robots for potential candidates
    
    closest_robot = None
    min_path_length = float('inf')
    
    for i in range(len(index[0])):
        robot = robots[index[0][i]]
        
        # Use A* to calculate the path from the robot to the point
        path = a_star(grid, robot, point)
        
        if path is not None:  # If a valid path is found
            path_length = len(path)
            if path_length < min_path_length:
                min_path_length = path_length
                closest_robot = robot
    
    return closest_robot

if __name__ == "__main__":
    # Example usage
    robots = [(1, 2), (3, 4), (5, 6), (7, 8)]  # Robot positions
    point = (2, 3)  # Query point
    grid = np.zeros((10, 10))  # 10x10 grid (0 = free, 1 = obstacle)
    grid[4][4] = 1  # Add an obstacle at (4, 4)
    grid[5][5] = 1  # Add an obstacle at (5, 5)

    nearest_robot = closest_robot_with_obstacles(robots, point, grid, 4)
    print(f"Closest robot: {nearest_robot}")