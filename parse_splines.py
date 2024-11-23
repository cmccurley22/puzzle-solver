import scipy.ndimage.filters as filters
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import edge_rotate
import networkx as nx
from scipy.spatial import distance
from scipy.interpolate import interp1d
import math



def parse_edges(edges, corners):
    """Parse edges into four sections based on corners."""
    # Get coordinates of all edge pixels
    spline_coordinates = np.argwhere(edges == 255)
    
    # Create an empty graph and add all edge points as nodes
    G = nx.Graph()
    for i, coord in enumerate(spline_coordinates):
        G.add_node(tuple(coord))
 
    # Add edges to the two nearest neighbors for each point
    for i, coord in enumerate(spline_coordinates):
        distances = [
            (distance.euclidean(coord, neighbor), neighbor)
            for j, neighbor in enumerate(spline_coordinates) if i != j
        ]
        # Sort by distance and get the two nearest neighbors
        nearest_neighbors = sorted(distances)[:2]
        
        # Add edges to the graph for the two nearest neighbors
        for dist, neighbor in nearest_neighbors:
            G.add_edge(tuple(coord), tuple(neighbor))
        
    # Find the closest edge points to each corner
    corner_points = [min(spline_coordinates, key=lambda p: np.linalg.norm(p - corner)) for corner in corners]
    
    #Calculate center point
    center_point_x = sum(corners[0] for corners in corner_points)/4
    center_point_y = sum(corners[0] for corners in corner_points)/4

    # Use shortest path algorithm to find paths between corners
    splines = []
    for i in range(len(corner_points)):
        start = corner_points[i]
        end = corner_points[(i + 1) % len(corner_points)]  # Connect sequentially in a loop
        spline = nx.shortest_path(G, source=tuple(start), target=tuple(end), weight=None, method='dijkstra')
        #gives me the nodes
        # if spline_distance>
        splines.append(spline)
    
    return splines#format as [[],[],[],[]]

def classify_outwardness():
   
    
