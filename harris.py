
import scipy.ndimage as ndimage
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import scipy
import math


img = cv2.imread("testpiece.png")

operatedImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 


operatedImage = np.float32(operatedImage) 
  
dest = cv2.cornerHarris(operatedImage, 5, 5, 0.04) 
  
dest = cv2.dilate(dest, None) 

# img[dest > 0.01 * dest.max()]=[0, 0, 255] 




def filter_corners(harris_res):
    threshold = .3

    data = harris_res.copy()

    data[data < threshold * harris_res.max()] = 0
    data_max_val = ndimage.maximum_filter(data, 5)
    data_maxima = (data == data_max_val)

    l_data, n_objs = ndimage.label(data_maxima)

    yx = np.array(ndimage.center_of_mass(data, l_data, range(1, n_objs + 1)))

    return yx[:, ::-1]


def fit_rect(xy):
    verbose = 0
    perp_angle_thresh = 20

    n = len(xy)

    dists = scipy.spatial.distance.cdist(xy, xy)

    dist_threshold = 30
    dists[dists < dist_threshold] = 0

    def find_angles(xy):
        angles = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                pos_i, pos_j = xy[i], xy[j]

                if pos_i[0] == pos_j[0]:
                    theta = 90
                else:
                    theta = math.atan2(pos_j[1] - pos_i[1], pos_j[0] - pos_i[0]) * 180 / math.pi
                
                angles[i, j] = theta
                angles[j, i] = theta
        
        return angles
    
    angles = find_angles(xy)
    possible_rectangles = []

    def search_for_possible_rectangle(idx, prev_points=[]):

        curr_point = xy[idx]
        depth = len(prev_points)

        if depth == 0:
            right_points_idx = np.nonzero(np.logical_and(xy[:, 0] > curr_point[0], dists[idx] > 0))[0]
            
            if verbose >= 2:
                print ('point', idx, curr_point)
                
            for right_point_idx in right_points_idx:
                search_for_possible_rectangle(right_point_idx, [idx])

            if verbose >= 2:
                print
                
            return


        last_angle = angles[idx, prev_points[-1]]
        perp_angle = last_angle - 90
        if perp_angle < 0:
            perp_angle += 180

        if depth in (1, 2):

            if verbose >= 2:
                print('\t' * depth, 'point', idx, '- last angle', last_angle, '- perp angle', perp_angle)

            diff0 = np.abs(angles[idx] - perp_angle) <= perp_angle_thresh
            diff180_0 = np.abs(angles[idx] - (perp_angle + 180)) <= perp_angle_thresh
            diff180_1 = np.abs(angles[idx] - (perp_angle - 180)) <= perp_angle_thresh
            all_diffs = np.logical_or(diff0, np.logical_or(diff180_0, diff180_1))
            
            diff_to_explore = np.nonzero(np.logical_and(all_diffs, dists[idx] > 0))[0]

            if verbose >= 2:
                print('\t' * depth, 'diff0:', np.nonzero(diff0)[0], 'diff180:', np.nonzero(diff180)[0], 'diff_to_explore:', diff_to_explore)

            for dte_idx in diff_to_explore:
                if dte_idx not in prev_points: # unlickly to happen but just to be certain
                    next_points = prev_points[::]
                    next_points.append(idx)

                    search_for_possible_rectangle(dte_idx, next_points)
                
        if depth == 3:
            angle41 = angles[idx, prev_points[0]]

            diff0 = np.abs(angle41 - perp_angle) <= perp_angle_thresh
            diff180_0 = np.abs(angle41 - (perp_angle + 180)) <= perp_angle_thresh
            diff180_1 = np.abs(angle41 - (perp_angle - 180)) <= perp_angle_thresh
            dist = dists[idx, prev_points[0]] > 0

            if dist and (diff0 or diff180_0 or diff180_1):
                rect_points = prev_points[::]
                rect_points.append(idx)
                
                if verbose == 2:
                    print('We have a rectangle:', rect_points)

                already_present = False
                for possible_rectangle in possible_rectangles:
                    if set(possible_rectangle) == set(rect_points):
                        already_present = True
                        break

                if not already_present:
                    possible_rectangles.append(rect_points)

    if verbose >= 2:
        print('Coords')
        print(xy)
        print()
        print('Distances')
        print(dists)
        print()
        print('Angles')
        print(angles)
        print()
    
    for i in range(n):
        search_for_possible_rectangle(i)                 
                    
    if len(possible_rectangles) == 0:
        return None

    def PolyArea(x,y):
        return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

    areas = []
    rectangularness = []
    diff_angles = []

    for r in possible_rectangles:
        points = xy[r]
        areas.append(PolyArea(points[:, 0], points[:, 1]))

        mse = 0
        da = []
        for i1, i2, i3 in [(0, 1, 2), (1, 2, 3), (2, 3, 0), (3, 0, 1)]:
            diff_angle = abs(angles[r[i1], r[i2]] - angles[r[i2], r[i3]])
            da.append(abs(diff_angle - 90))
            mse += (diff_angle - 90) ** 2

        diff_angles.append(da)
        rectangularness.append(mse)


    areas = np.array(areas)
    rectangularness = np.array(rectangularness)

    scores = areas * scipy.stats.norm(0, 150).pdf(rectangularness)
    best_fitting_idxs = possible_rectangles[np.argmax(scores)]
    return xy[best_fitting_idxs]


xy = filter_corners(dest)
xy = np.round(xy).astype(np.int_)

intersections = fit_rect(xy)

print(intersections)


for point in intersections:
    img = cv2.circle(img, point, radius = 5, color = (0, 0, 255), thickness = -1)

cv2.imshow('Image with Borders', img) 
cv2.waitKey(0)
