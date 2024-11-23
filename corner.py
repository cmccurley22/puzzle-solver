import scipy.ndimage as ndimage
import numpy as np
import cv2
import scipy
import math

def filter_corners(harris_res):
    threshold = 0.3
    data = harris_res.copy()
    data[data < threshold * harris_res.max()] = 0
    data_max_val = ndimage.maximum_filter(data, 5)
    data_maxima = (data == data_max_val)
    l_data, n_objs = ndimage.label(data_maxima)
    yx = np.array(ndimage.center_of_mass(data, l_data, range(1, n_objs + 1)))
    return yx[:, ::-1]


def fit_rect(xy):
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
                theta = math.atan2(pos_j[1] - pos_i[1], pos_j[0] - pos_i[0]) * 180 / math.pi if pos_i[0] != pos_j[0] else 90
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
            for right_point_idx in right_points_idx:
                search_for_possible_rectangle(right_point_idx, [idx])
            return

        last_angle = angles[idx, prev_points[-1]]
        perp_angle = last_angle - 90
        if perp_angle < 0:
            perp_angle += 180

        if depth in (1, 2):
            diff0 = np.abs(angles[idx] - perp_angle) <= perp_angle_thresh
            diff180_0 = np.abs(angles[idx] - (perp_angle + 180)) <= perp_angle_thresh
            diff180_1 = np.abs(angles[idx] - (perp_angle - 180)) <= perp_angle_thresh
            all_diffs = np.logical_or(diff0, np.logical_or(diff180_0, diff180_1))
            diff_to_explore = np.nonzero(np.logical_and(all_diffs, dists[idx] > 0))[0]
            for dte_idx in diff_to_explore:
                if dte_idx not in prev_points:
                    search_for_possible_rectangle(dte_idx, prev_points + [idx])

        if depth == 3:
            angle41 = angles[idx, prev_points[0]]
            diff0 = np.abs(angle41 - perp_angle) <= perp_angle_thresh
            diff180_0 = np.abs(angle41 - (perp_angle + 180)) <= perp_angle_thresh
            diff180_1 = np.abs(angle41 - (perp_angle - 180)) <= perp_angle_thresh
            dist = dists[idx, prev_points[0]] > 0
            if dist and (diff0 or diff180_0 or diff180_1):
                rect_points = prev_points + [idx]
                if not any(set(rect_points) == set(possible_rectangle) for possible_rectangle in possible_rectangles):
                    possible_rectangles.append(rect_points)

    for i in range(n):
        search_for_possible_rectangle(i)                 
                    
    if len(possible_rectangles) == 0:
        return None

    def PolyArea(x, y):
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    areas, rectangularness = [], []
    for r in possible_rectangles:
        points = xy[r]
        areas.append(PolyArea(points[:, 0], points[:, 1]))
        mse = 0
        for i1, i2, i3 in [(0, 1, 2), (1, 2, 3), (2, 3, 0), (3, 0, 1)]:
            diff_angle = abs(angles[r[i1], r[i2]] - angles[r[i2], r[i3]]) - 90
            mse += diff_angle ** 2
        rectangularness.append(mse)

    scores = np.array(areas) * scipy.stats.norm(0, 150).pdf(np.array(rectangularness))
    best_fitting_idxs = possible_rectangles[np.argmax(scores)]
    return xy[best_fitting_idxs]


def detect_corners_and_fit_rectangle(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dest = cv2.cornerHarris(gray, 5, 5, 0.04)
    dest = cv2.dilate(dest, None)
    xy = filter_corners(dest)
    xy = np.round(xy).astype(np.int_)
    intersections = fit_rect(xy)
    return intersections

def compute_side_lengths(rect_points):
    side_lengths = []
    for i in range(len(rect_points)):
        pt1 = rect_points[i]
        pt2 = rect_points[(i + 1) % len(rect_points)]
        length = np.linalg.norm(np.array(pt1) - np.array(pt2))
        side_lengths.append(length)
    return side_lengths

