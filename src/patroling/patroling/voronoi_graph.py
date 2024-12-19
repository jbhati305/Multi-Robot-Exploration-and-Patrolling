import numpy as np
from PIL import Image
import yaml
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import cdist
from utils import *
import numpy
import sys
from scipy.ndimage import maximum_filter
import argparse

numpy.set_printoptions(threshold=sys.maxsize)

def pgm2occupancy(pgm_file, occupied_thresh=0.65, free_thresh=0.196):
    img = Image.open(pgm_file)
    img = img.convert('L')
    img_array = np.array(img) / 255.0
    occupancy_grid = np.zeros_like(img_array, dtype=int)
    occupancy_grid[img_array > 0.9] = 0
    occupancy_grid[img_array <= 0.9] = 1
    return occupancy_grid

def expand_obstacles(occupancy_grid, dilation_radius=20):
    """
    Expand obstacles using scipy's maximum_filter
    """
    kernel_size = 2 * dilation_radius + 1
    return maximum_filter(occupancy_grid, size=kernel_size)

class patrol_graph():
    def __init__(self, yaml_path, pgm_file) -> None:
        with open(yaml_path, 'r') as stream:
            self.map_yaml = yaml.safe_load(stream)
        self.pgm_file = pgm_file
        self.occupancy_grid = pgm2occupancy(self.pgm_file)

        self.resolution = self.map_yaml['resolution']
        self.origin = self.map_yaml['origin']
        self.height = self.occupancy_grid.shape[0]
        self.width = self.occupancy_grid.shape[1]

        self.astar = AStar(self.occupancy_grid)

    def gen_voronoi(self, distance_threshold=40, dilation_radius=20):
        self.expanded_grid = expand_obstacles(self.occupancy_grid, dilation_radius)
        free_spaces = np.array(np.where(self.expanded_grid == 0)).T
        print(f"Total free spaces in modified grid: {free_spaces.shape[0]}")

        if free_spaces.shape[0] == 0:
            print("No valid spaces found in the modified grid. Exiting...")
            return

        centers = []
        for i in range(free_spaces.shape[0]):
            if len(centers) == 0:
                centers.append(free_spaces[i])
            else:
                min_dist = np.min(np.linalg.norm(np.array(centers) - free_spaces[i], axis=1))
                if min_dist >= distance_threshold:
                    centers.append(free_spaces[i])

        centers = np.array(centers)
        print(f"Total Voronoi centers: {centers.shape[0]}")

        if centers.shape[0] == 0:
            print("No valid Voronoi centers found. Exiting...")
            return

        if centers.ndim == 1:
            centers = centers.reshape(-1, 2)

        mst_edges = self.get_mst(centers)
        generate_graph(centers=centers, mst_edges=mst_edges, 
                      filename=f"{self.pgm_file.split('.')[0]}.graph", 
                      resolution=self.resolution, origin=self.origin)

    def get_mst(self, centers):
        dist_matrix = cdist(centers, centers, 'euclidean')
        mst = minimum_spanning_tree(dist_matrix)
        print(f"Minimum Spanning Tree (MST) shape: {mst.shape}")
        mst = mst.toarray()

        mst_edges = []
        for i in range(mst.shape[0]):
            for j in range(i + 1, mst.shape[1]):
                if mst[i, j] > 0:
                    start = tuple(centers[i])
                    end = tuple(centers[j])
                    path = self.astar.a_star(start, end)
                    if path:
                        mst_edges.append((i, j, mst[i, j]))
        return mst_edges

def generate_graph(centers, mst_edges, filename, resolution, origin):
    """
    Save the graph data to a file
    """
    with open(filename, 'w') as f:
        # Write header (number of vertices)
        f.write(f"{len(centers)}\n")

        # Write vertices
        for i, center in enumerate(centers):
            x = center[1] * resolution + origin[0]
            y = center[0] * resolution + origin[1]
            f.write(f"{i} {x:.6f} {y:.6f}\n")

        # Write edges
        for edge in mst_edges:
            f.write(f"{edge[0]} {edge[1]} {edge[2]:.6f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate patrol graph from map files')
    parser.add_argument('yaml_path', type=str, help='Path to the YAML file')
    parser.add_argument('pgm_path', type=str, help='Path to the PGM file')
    parser.add_argument('--dilation_radius', type=int, default=2, 
                        help='Dilation radius (default: 2)')
    parser.add_argument('--distance_threshold', type=int, 
                        help='Distance threshold for Voronoi centers')

    args = parser.parse_args()

    G = patrol_graph(args.yaml_path, args.pgm_path)
    h = G.height
    w = G.width

    distance_threshold = args.distance_threshold if args.distance_threshold else (h*w)//10000

    G.gen_voronoi(distance_threshold=distance_threshold, 
                  dilation_radius=args.dilation_radius)