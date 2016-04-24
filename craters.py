"""Problem Set 2: Edges and Lines."""

import numpy as np
import cv2

import os
from math import pi
import time

input_dir = "input"  # read images from os.path.join(input_dir, <filename>)
output_dir = "output"  # write images to os.path.join(output_dir, <filename>)


def hough_lines_acc(img_edges, rho_res=1, theta_res=pi/90, normalize=True):
    """Compute Hough Transform for lines on edge image.

    Parameters
    ----------
        img_edges: binary edge image
        rho_res: rho resolution (in pixels)
        theta_res: theta resolution (in radians)

    Returns
    -------
        H: Hough accumulator array
        rho: vector of rho values, one for each row of H
        theta: vector of theta values, one for each column of H
    """

    img_rows = img_edges.shape[0]
    img_cols = img_edges.shape[1]

    # Compute number of rows in array
    diagonal_length = np.sqrt(img_rows ** 2 + img_cols ** 2)

    rho_rows = int(round(diagonal_length * 2 / rho_res)) + 1

    # Computer number of columns
    theta_cols = int(round(np.pi / theta_res)) + 0

    # Initialize zero array as H
    H = np.zeros((rho_rows, theta_cols))

    for i in range(img_rows):
        for j in range(img_cols):
            if img_edges[i, j] > 0:
                current_theta = 0
                for t in range(theta_cols):
                    # compute rho as d
                    d = round(j * np.cos(current_theta) + i * np.sin(current_theta))

                    # normalize d to map to non-negative value for row index
                    d = int(d / rho_res + rho_rows / 2)

                    # check that we get proper theta index
                    theta_col = int(round(current_theta / theta_res))
                    H[d, theta_col] += 1

                    # Increment current_theta for next loop
                    current_theta += theta_res

    if normalize:
        H = normalize_accumulator(H)

    # Create rho vector
    rho = []
    for i in range(rho_rows):
        rho_val = - round(diagonal_length) + i * rho_res
        rho.append(rho_val)

    rho = np.array(rho)

    # Create theta vector
    theta = []
    for j in range(theta_cols):
        theta.append(theta_res * j)

    theta = np.array(theta)

    return H, rho, theta


def hough_peaks(H, Q, threshold=70):
    """Find peaks (local maxima) in accumulator array.

    Parameters
    ----------
        H: Hough accumulator array
        Q: number of peaks to find (max)
        threshold: value of number of votes should be between 0 and 255 for normalized H

    Returns
    -------
        peaks: Px2 matrix (P <= Q) where each row is a (rho_idx, theta_idx) pair
    """
    # TODO: Your code here
    peaks = []

    # Find peaks greater than threshold
    for rho_idx in range(H.shape[0]):
        for theta_idx in range(H.shape[1]):
            if H[rho_idx, theta_idx] >= threshold:
                peaks.append([rho_idx, theta_idx, H[rho_idx, theta_idx]])

    # Ensure only the Q highest peaks are on the list
    if len(peaks) > Q:
        peaks = sorted(peaks, reverse=True, key=lambda votes: votes[2])
        peaks = peaks[:Q]

    peaks = np.array(peaks, dtype='uint16')

    # Slice off column containing votes from peaks
    peaks = peaks[0:, 0:2]

    return peaks


def hough_lines_draw(img_out, peaks, rho, theta):
    """Draw lines on an image corresponding to accumulator peaks.

    Parameters
    ----------
        img_out: 3-channel (color) image
        peaks: Px2 matrix where each row is a (rho_idx, theta_idx) index pair
        rho: vector of rho values, such that rho[rho_idx] is a valid rho value
        theta: vector of theta values, such that theta[theta_idx] is a valid theta value
    """
    """
    For each (rho, theta) pair in peaks:

    x = rho[i] * cos(theta[i])

    y = rho[i] * sin(theta[i])

    That gives you one point on the line, now you need to calculate other points to be able
    to draw the line.  These points should be far enough apart that the line you draw on the
    image covers the entire image window.

    d_theta = x0*cos(theta) + y0*sin(theta)
    """

    for i in range(peaks.shape[0]):

        theta_idx = peaks[i, 1]
        rho_idx = peaks[i, 0]

        a = np.cos(theta[theta_idx])
        b = np.sin(theta[theta_idx])

        # TODO need to adjust rho[rho_idx] to map to actual pixel value not just negative
        x0 = a * rho[rho_idx]
        y0 = b * rho[rho_idx]

        x1 = int(round(x0 + 1000 * (-b)))
        y1 = int(round(y0 + 1000 * a))

        x2 = int(round(x0 - 1000 * (-b)))
        y2 = int(round(y0 - 1000 * a))


        cv2.line(img_out, (x1, y1), (x2, y2), (0, 255, 0), 1)


    # TODO: Your code here (nothing to return, just draw on img_out directly)


def normalize_accumulator(H):
    """
    Normalizes Hough accumulator array by mapping the minimum value to zero and the maximum
    value to 255 with array data type uint8
    :param H: Hough accumulator array
    :return: normalized_H
    """

    vote_range = H.max() - H.min()

    normalized_accumulator = H - H.min()
    normalized_accumulator = normalized_accumulator * 255 / vote_range
    normalized_accumulator = np.array(normalized_accumulator, 'uint8')

    return normalized_accumulator


def highlight_peaks(peaks, accumulator):
    for i in range(peaks.shape[0]):

        # Initialize local_peak tuple
        local_peak = (peaks[i, 1], peaks[i, 0],)

        # Draw green circle with radius 5 around center of peak
        cv2.circle(accumulator, local_peak, 3, (0, 255, 0), 1)

    return accumulator


def hough_circles_acc(img_edges, radius, theta_res=pi/45, normalize=True):
    """

    :param img_edges: Monochrome edge image array to detect circles from
    :param radius: radius of circle to detect
    :param normalize: normalizes the accumulator array if set to True which is default
    :return:
        H: Hough accumulator array which will be an monochrome array of the same size as the img array.
        H is normalized to map values from 0 to 255 in a dtype uint8 array if normalize is set to True
    """

    # Initialize zero array with the same size as img
    H = np.zeros(img_edges.shape, dtype='uint16')

    img_rows = img_edges.shape[0]
    img_cols = img_edges.shape[1]

    for i in range(img_rows):
        for j in range(img_cols):
            if img_edges[i, j] > 0:
                theta = 0
                draw_points = int((2 * pi)/theta_res)
                for t in range(0, draw_points):
                    a = int(round(j - radius * np.cos(theta)))
                    b = int(round(i - radius * np.sin(theta)))

                    #print('(a, b) is (' + str(a) + ', ' + str(b) + ')')

                    # Check for out of range indices
                    if a < img_cols and b < img_rows:
                        H[b, a] += 1

                    theta += theta_res

    if normalize:
        H = normalize_accumulator(H)

    return H


def hough_centers(H, Q, radius, threshold=100):
    """Find peaks (local maxima) in accumulator array.

    Parameters
    ----------
        H: Hough accumulator array
        Q: number of peaks to find (max)
        threshold: value of number of votes should be between 0 and 255 for normalized H

    Returns
    -------
        peaks: Px3 matrix (P <= Q) where each row is (radius, j, i, votes) tuple
    """
    # TODO: Your code here
    centers = []

    # Find peaks greater than threshold
    for b in range(H.shape[0]):
        for a in range(H.shape[1]):
            if H[b, a] >= threshold:
                centers.append([radius, b, a, H[b, a]])

    # Ensure only the Q highest peaks are on the list
    if len(centers) > Q:
        centers = sorted(centers, reverse=True, key=lambda votes: votes[2])
        centers = centers[:Q]

    return centers


def remove_close_overlapping_circles(centers, max_distance=3):
    """
    Remove duplicate centers by checking for very close center points at similar radius values
    :param centers: List of centers with each element containing [radius, b, a, votes]
    :return: centers after stripping closely overlapping circles
    """

    duplicate_centers = []
    centers_orig = list(centers)

    # Remove consecutive points in centers that have close euclidean distance, keeping the circle with the most votes
    if len(centers) > 1:
        i = 0
        while (i + 1) < len(centers):
            point_i = (centers[i][2], centers[i][1])
            point_j = (centers[i + 1][2], centers[i + 1][1])

            distance = euclidean_distance(point_i, point_j)

            if 0.1 < distance < max_distance:
                if centers[i + 1][3] > centers[i][3]:
                    centers.remove(centers[i])
                else:
                    centers.remove(centers[i + 1])
                i -= 1

            i += 1

    # Final sweep through centers with nested for loops to remove close or overlapping circles keeping the
    # circles with the larger radius values
    centers_orig = list(centers)
    stop_idx = len(centers_orig)
    if stop_idx > 1:
        for i in range(stop_idx):
            for j in range(stop_idx):
                point_i = (centers_orig[i][2], centers_orig[i][1])
                point_j = (centers_orig[j][2], centers_orig[j][1])

                distance = euclidean_distance(point_i, point_j)

                if 0.01 < distance < max_distance:
                    if centers_orig[i][0] > centers_orig[j][0]:
                        duplicate_centers.append(centers_orig[j])
                    else:
                        duplicate_centers.append(centers_orig[i])

    for i in range(len(duplicate_centers)):
        if centers.count(duplicate_centers[i]) > 0:
            centers.remove(duplicate_centers[i])

    return centers


def euclidean_distance(point1, point2):
    distance = np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    return distance


def hough_circles_draw_single_r(img_out, centers, radius):
    """
    Description:  Draws circles onto the original color image.
    :param img_out: Color image to have circles drawn on
    :param centers: 2xQ array of center point values where there are Q center points
    :param radius: radius of circles to be drawn
    """
    num_centers = centers.shape[0]

    for i in range(num_centers):
        row = centers[i, 0]
        column = centers[i, 1]
        # circle function takes center coordinate in (x, y) format or (j, i)
        cv2.circle(img_out, (column, row), radius, (0, 255, 0))


def find_circles(img_edges, min_max_radius, resolution=pi/45 ,Q=10, threshold=50):
    """

    :param img_edges: Monochrome edge image
    :param min_max_radius: tuple of minimum and maximum radius sizes to look for as (min, max)
    :return:
    """
    min_radius = min_max_radius[0]
    max_radius = min_max_radius[1]

    radius_range = max_radius - min_radius

    rows = img_edges.shape[0]
    columns = img_edges.shape[1]
    height = radius_range + 1

    # Create 3-D array of Hough accumulators with height representing the number of radius to check
    H = np.zeros((height, rows, columns))

    for k in range(height):
        H[k] = hough_circles_acc(img_edges, min_radius + k, theta_res=resolution, normalize=False)
        #print("Finished hough_circles_acc for radius " + str(min_radius + k))

    centers = []
    for i in range(height):
        centers_i = hough_centers(H[i], Q, min_radius + i, threshold=threshold)
        if len(centers_i) > 0:
            for j in range(len(centers_i)):
                centers.append(centers_i[j])

    # Create list of radii to return
    radii = []
    for i in range(min_radius, max_radius + 1):
        radii.append(i)

    return centers, radii


def hough_centers_draw_multi_r(img_out, centers):
    num_centers = len(centers)

    for i in range(num_centers):
        radius = centers[i][0]
        row = centers[i][1]
        column = centers[i][2]
        # circle function takes center coordinate in (x, y) format or (j, i)
        cv2.circle(img_out, (column, row), radius, (0, 255, 0))


def get_parallel_lines(peaks, min_distance, max_distance):
    """
    Removes all non-parallel lines from peaks that are not within the minimum or maximum distance thresholds
    :param peaks: peak array
    :param min_distance: minimum distance between two parallel lines
    :param max_distance: maximum distance between two parallel lines
    :return peaks: after processing for parallel lines
    """

    num_peaks = peaks.shape[0]

    peaks_list = []

    for i in range(num_peaks):
        for j in range(num_peaks):
            if i != j:
                # Check if the peaks have similar theta values indicating similar slope
                if np.abs(int(peaks[i, 1]) - int(peaks[j, 1])) < 1:
                    # Check how far apart similarly sloped lines are
                    if min_distance < np.abs(int(peaks[i, 0]) - int(peaks[j, 0])) < max_distance:
                        peak_i = [peaks[i, 0], peaks[i, 1]]
                        peak_j = [peaks[j, 0], peaks[j, 1]]
                        peaks_list.append(peak_i)
                        peaks_list.append(peak_j)

    peaks_list = remove_duplicates_peaks(peaks_list)

    # get unique theta values
    peaks_set = set()
    for i in range(len(peaks_list)):
        peaks_set.add(peaks_list[i][1])

    # initialize dictionary of the empty lists for each unique theta value
    peaks_dict = dict()
    for element in peaks_set:
        peaks_dict[element] = list()

    # Add lists of values to each dictionary entry based on theta values
    for key in peaks_dict:
        for i in range(len(peaks_list)):
            if peaks_list[i][1] == key:
                peaks_dict[key].append(peaks_list[i])

    # Sort dictionary list values
    for key in peaks_dict:
        peaks_dict[key] = sorted(peaks_dict[key], key=lambda elem: elem[0])

    # If more than two lists exist for each key in peaks_dict remove all middle lists
    for key in peaks_dict:
        if len(peaks_dict[key]) > 2:
            while len(peaks_dict[key]) > 2:
                peaks_dict[key].pop(1)

    # Put dictionary values back into peaks_list
    peaks_list = list()
    for key in peaks_dict:
        for i in range(len(peaks_dict[key])):
            peaks_list.append(peaks_dict[key][i])

    peaks_list = np.array(peaks_list)

    return peaks_list


def remove_duplicates_peaks(peaks_list):
    deduped_list = []
    for i in range(len(peaks_list)):
        if deduped_list.count(peaks_list[i]) == 0:
            deduped_list.append(peaks_list[i])

    return deduped_list


def add_warped_circles(img, circles):
    """
    Draws warped circles on to image on green channel
    :param img: Original image to have circles drawn on them
    :param circles: image array with only warped circles on green channel
    :return:
    """
    rows = img.shape[0]
    cols = img.shape[1]

    for i in range(rows):
        for j in range(cols):
            if circles[i, j, 1] > 0:
                img[i, j, 1] = 255


def main():
    start = time.time()
    # Crater detection
    ast = cv2.imread('images/PIA15351.jpg', 0)
    ast_blurred = cv2.GaussianBlur(ast, (7, 7), 1)
    cv2.imwrite('images/ast_gaussian_blur.png', ast_blurred)
    ast_blurred = cv2.medianBlur(ast, 7)
    cv2.imwrite('images/ast_median_blur.png', ast_blurred)

    ast_blurred_edge = cv2.Canny(ast_blurred, 50, 100)
    cv2.imwrite('images/ast_edge.png', ast_blurred_edge)
    centers, radii = find_circles(ast_blurred_edge, (5, 100), resolution=pi/24, Q=10, threshold=16)
    centers = remove_close_overlapping_circles(centers, max_distance=20)

    img_out = cv2.cvtColor(ast, cv2.COLOR_GRAY2BGR)
    hough_centers_draw_multi_r(img_out, centers)
    cv2.imwrite('images/ast_craters.png', img_out)
    #
    # circles = cv2.HoughCircles(ast, cv2.HOUGH_GRADIENT, 1, 20,
    #                            param1=50, param2=30, minRadius=0, maxRadius=0)
    #
    # circles = np.uint16(np.around(circles))
    # for i in circles[0, :]:
    #     # draw the outer circle
    #     cv2.circle(ast, (i[0], i[1]), i[2], (0, 255, 0), 2)
    #     # draw the center of the circle
    #     cv2.circle(ast, (i[0], i[1]), 2, (0, 0, 255), 3)
    #
    # cv2.imwrite('images/test_circles.png', ast)
    end = time.time()
    run_time = end - start
    print('Run time is ' + str(run_time) + ' seconds')

if __name__ == "__main__":
    main()
