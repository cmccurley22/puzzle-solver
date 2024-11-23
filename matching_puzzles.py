
import scipy.ndimage.filters as filters
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import edge_rotate
import networkx as nx
from scipy.spatial import distance
from scipy.interpolate import interp1d
import math
from itertools import zip_longest
class PuzzlePiece:

    def __init__(self, splines=None, corners =None, rectangle = None):
        self.splines = splines
        self.corners = corners
        self.rectangle = rectangle



def calculate_match_splines(target_puzzle_spline, t_outedness, other_spline, o_outedness):
    # Calculate the average offset using the first and last points

    avg_x = ((target_puzzle_spline[0][0] - other_spline[0][0]) +
             (target_puzzle_spline[-1][0] - other_spline[-1][0])) / 2
    avg_y = ((target_puzzle_spline[0][1] - other_spline[0][1]) +
             (target_puzzle_spline[-1][1] - other_spline[-1][1])) / 2
    
    
    error = 0
    for t_point, o_point in zip_longest(target_puzzle_spline, other_spline, fillvalue=(0, 0)):
        error += (t_point[0] - o_point[0])**2 + (t_point[1] - o_point[1])**2
    
    return  

    
def detect_outer_edge(self, spline):
    threshold = 0.5
    index = self.splines.index(spline)
    line = self.retangle(index)  # Assuming retangle() returns a line representation for comparison
    
    error = 0
    for point in spline:
        # Calculate the minimum distance from point to line and accumulate error
        point_distance = min(distance.cdist([point], line, 'euclidean')[0])
        error += point_distance

    return error > threshold

