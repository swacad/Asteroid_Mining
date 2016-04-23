import numpy as np
import cv2
from scipy import spatial
import matplotlib.pyplot as plt

import os
import time

# I/O directories
input_dir = "input"
output_dir = "output"


# Assignment code
def gradientX(image):
    """Compute image gradient in X direction.

    Parameters
    ----------
        image: grayscale floating-point image, values in [0.0, 1.0]

    Returns
    -------
        Ix: image gradient in X direction, values in [-1.0, 1.0]
    """

    # TODO: Your code here
    Ix = image.copy()

    Ix = cv2.Sobel(Ix, cv2.CV_64F, 1, 0, ksize=5)
    return Ix


def gradientY(image):
    """Compute image gradient in Y direction.

    Parameters
    ----------
        image: grayscale floating-point image, values in [0.0, 1.0]

    Returns
    -------
        Iy: image gradient in Y direction, values in [-1.0, 1.0]
    """

    # TODO: Your code here

    Iy = image.copy()

    Iy = cv2.Sobel(Iy, cv2.CV_64F, 0, 1, ksize=5)

    return Iy


def normalize_array(A, new_min, new_max):
    """
    Normalizes array, A, with values normalized in the range of new_min to new_max.
    :param A: array to normalize of type numpy.array
    :param new_min: minimum of normalized range
    :param new_max: maximum of normalized range
    :return normalized_A: normalized array
    """
    min_max_range = new_max - new_min

    normalized_A = A.copy()

    # Set lowest value to zero
    normalized_A = normalized_A - normalized_A.min()

    # Normalize from 0 to min_max_range
    normalized_A = normalized_A * min_max_range / normalized_A.max()

    # Set minimum A value to new_min
    normalized_A = normalized_A + new_min

    return normalized_A


def make_image_pair(image1, image2):
    """Adjoin two images side-by-side to make a single new image.

    Parameters
    ----------
        image1: first image, could be grayscale or color (BGR)
        image2: second image, same type as first

    Returns
    -------
        image_pair: combination of both images, side-by-side, same type
    """

    # TODO: Your code here
    # Get the image shape of the paired images as the same size except for double the columns
    pair_shape = list(image1.shape)
    pair_shape[1] = pair_shape[1] * 2
    pair_shape = tuple(pair_shape)

    image_pair = np.zeros(pair_shape)

    # Assign each half of the image_pair with image1 and image2
    image_pair[:, :image1.shape[1]] = image1
    image_pair[:, image2.shape[1]:] = image2

    return image_pair


def harris_response(Ix, Iy, kernel, alpha):
    """Compute Harris reponse map using given image gradients.

    Parameters
    ----------
        Ix: image gradient in X direction, values in [-1.0, 1.0]
        Iy: image gradient in Y direction, same size and type as Ix
        kernel: 2D windowing kernel with weights, typically square
        alpha: Harris detector parameter multiplied with square of trace.  Usually in range 0.04 to 0.06

    Returns
    -------
        R: Harris response map, same size as inputs, floating-point
    """

    # TODO: Your code here
    # Note: Define any other parameters you need locally or as keyword arguments
    num_rows = Ix.shape[0]
    num_cols = Ix.shape[1]
    R = np.zeros(Ix.shape)
    M = np.zeros((2, 2))

    # Create arrays of the summed elements for the second moment matrix, M
    Ix_squared = Ix * Ix
    Iy_squared = Iy * Iy
    IxIy = Ix * Iy

    # Create arrays of weighted sums using the kernel to weight values around individual pixels
    # Arrays are named M_{element index} in matrix M.  M_01 will also hold M_10 values.
    M_00 = cv2.filter2D(Ix_squared, -1, kernel)
    M_11 = cv2.filter2D(Iy_squared, -1, kernel)
    M_01 = cv2.filter2D(IxIy, -1, kernel)

    det_matrix = M_00 * M_11 - M_01 * M_01
    alpha_trace_matrix = alpha * ((M_00 + M_11) * (M_00 + M_11))

    R = det_matrix - alpha_trace_matrix

    return R


def find_corners(R, threshold, radius):
    """Find corners in given response map.

    Parameters
    ----------
        R: floating-point response map, e.g. output from the Harris detector
        threshold: response values less than this should not be considered plausible corners
        radius: radius of circular region for non-maximal suppression (could be half the side of square instead)

    Returns
    -------
        corners: peaks found in response map R, as a sequence (list) of (x, y) coordinates
    """

    # TODO: Your code here
    corners = list()
    num_rows = R.shape[0]
    num_cols = R.shape[1]

    # Normalize response map for more predictable thresholding behavior.
    # Adjusting the new_max parameter will increase or decrease the number of corners found
    normalized_R = normalize_array(R, -1, 1.6)
    normalized_R = np.array(normalized_R, dtype='float32')

    # remove points below threshold
    #ret, thresholded_R = cv2.threshold(normalized_R, threshold, 1.0, cv2.THRESH_TOZERO)
    thresholded_R = threshold_array(normalized_R, 0.0)

    nms_R = non_max_suppression(thresholded_R, radius)

    for i in range(num_rows):
        for j in range(num_cols):
            if nms_R[i, j] > 0:
                corners.append((j, i))

    print('thresholded_R average = ' + str(thresholded_R.sum() / (thresholded_R.shape[0] * thresholded_R.shape[1])))
    print('Number of corners is ' + str(len(corners)))

    return corners


def non_max_suppression(thresholded_R, radius):
    """
    Performs non-maxima suppression (NMS) on the response map to remove duplicate corners that are within a square box.
    :param thresholded_R: numpy.ndarray which has been thresholded
    :param radius: length of half the side of the square
    :return nms_R: non-maximally suppressed response map as np.ndarray.
    """
    # Type cast radius to int to prevent indexing errors during non-maxima suppression
    radius = int(radius)

    num_rows = thresholded_R.shape[0]
    num_cols = thresholded_R.shape[1]

    # Perform non-maximal suppression in a square which has sides double the length of the radius parameter.
    # Pad thresholded_R by twice the radius pixels on all four sides to prevent out of range indexing.
    padded_R = np.zeros((num_rows + radius * 2, num_cols + radius * 2))
    last_row_idx = padded_R.shape[0]
    last_col_idx = padded_R.shape[1]
    padded_R[radius:last_row_idx - radius, radius:last_col_idx - radius] = thresholded_R

    # Loop through rows and columns.  i matches to the y coordinate and j matches to x coordinate
    for i in range(last_row_idx):
        for j in range(last_col_idx):
            if padded_R[i, j] > 0:
                # Get square portion of image with the (i, j) pixel in the center
                window = padded_R[i - radius:i + radius + 1, j - radius:j + radius + 1]

                # Set non-maximum values in the window to zero and max value to 1
                max_idx = window.argmax()
                max_idx = np.unravel_index(max_idx, window.shape)
                window = window * 0
                window[max_idx] = 1

                # Assign window back to padded_R
                padded_R[i - radius:i + radius + 1, j - radius:j + radius + 1] = window

    # Create new array which is non-maximally suppressed (nms) version of R
    nms_R = padded_R[radius:last_row_idx - radius, radius:last_col_idx - radius]

    return nms_R


def threshold_array(A, threshold):
    """
    Computes a thresholded array where values less than the threshold are set to zero.
    This function uses vectorized operations to optimize performance.
    All other values remain the same.
    :param A: array to thresholded
    :param threshold: a number greater than or equal to zero
    :return: A, which is now thresholded
    """
    # Create array of threshold values of same size as A
    threshold_valued_array = np.ones(A.shape)
    threshold_valued_array = threshold_valued_array * threshold

    # Create boolean array by doing comparison b and a
    boolean_array = np.less(threshold_valued_array, A)

    # Multiply boolean array with A to zero out all values below the threshold
    A = boolean_array * A

    return A


def draw_corners(image, corners):
    """Draw corners on (a copy of) given image.

    Parameters
    ----------
        image: grayscale floating-point image, values in [0.0, 1.0]
        corners: peaks found in response map R, as a sequence (list) of (x, y) coordinates

    Returns
    -------
        image_out: copy of image with corners drawn on it, color (BGR), uint8, values in [0, 255]
    """
    # TODO: Your code here
    # Copy image to image_out
    image_out = np.array(image)

    # Normalize image_out from 0 to 255 range
    image_out = image_out * 255

    # Convert data type of image out to 'float32' to make it work with cv2.cvtColor function
    image_out = np.array(image_out, dtype='float32')

    # Convert image out to 3-channel color image
    image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)

    for corner in corners:
        cv2.circle(image_out, corner, 3, (0, 255, 0), 1)

    return image_out


def gradient_angle(Ix, Iy):
    """Compute angle (orientation) image given X and Y gradients.

    Parameters
    ----------
        Ix: image gradient in X direction
        Iy: image gradient in Y direction, same size and type as Ix

    Returns
    -------
        angle: gradient angle image, each value in degrees [0, 359)
    """

    # TODO: Your code here
    # Note: +ve X axis points to the right (0 degrees), +ve Y axis points down (90 degrees)
    num_rows = Ix.shape[0]
    num_cols = Iy.shape[1]

    #angle = cv2.phase(Ix, Iy, angleInDegrees=True)

    angle = np.zeros(Ix.shape)

    for i in range(num_rows):
        for j in range(num_cols):
            angle[i, j] = np.arctan2(Iy[i, j], Ix[i, j])
            if angle[i, j] < 0:
                angle[i, j] = angle[i, j] + np.pi * 2

    # Convert radians to degrees
    angle = angle * 180 / np.pi

    return angle


def get_keypoints(points, R, angle, _size, _octave=0):
    """Create OpenCV KeyPoint objects given interest points, response and angle images.

    Parameters
    ----------
        points: interest points (e.g. corners), as a sequence (list) of (x, y) coordinates
        R: floating-point response map, e.g. output from the Harris detector
        angle: gradient angle (orientation) image, each value in degrees [0, 359)
        _size: fixed _size parameter to pass to cv2.KeyPoint() for all points
        _octave: fixed _octave parameter to pass to cv2.KeyPoint() for all points

    Returns
    -------
        keypoints: a sequence (list) of cv2.KeyPoint objects
    """

    # TODO: Your code here
    # Note: You should be able to plot the keypoints using cv2.drawKeypoints() in OpenCV 2.4.9+

    keypoints = list()

    # Load keypoints from points list
    for point in points:
        # Get gradient angle for point
        i = point[1]
        j = point[0]
        kp_angle = angle[i, j]

        keypoint = cv2.KeyPoint(j, i, _size=_size, _angle=kp_angle, _octave=_octave)
        keypoints.append(keypoint)

    return keypoints


def get_descriptors(image, keypoints):
    """Extract feature descriptors from image at each keypoint.
    Reference: https://docs.google.com/document/d/1r0DCy2kXlc31Cl4qUQLGyATAKJgVBNZPuszPkmOxu3A/view?pref=2&pli=1

    Parameters
    ----------
        keypoints: a sequence (list) of cv2.KeyPoint objects

    Returns
    -------
        descriptors: 2D NumPy array of shape (len(keypoints), 128)
    """
    # TODO: Your code here
    # Note: You can use OpenCV's SIFT.compute() method to extract descriptors, or write your own!

    # Convert image to data type uint8 and normalize from 0 to 255
    image = normalize_array(image, 0, 255)
    image = np.array(image, dtype='uint8')

    sift = cv2.xfeatures2d.SIFT_create()

    keypoints, descriptors = sift.compute(image, keypoints)

    return descriptors


def match_descriptors(desc1, desc2):
    """Match feature descriptors obtained from two images.
    Reference: https://docs.google.com/document/d/1r0DCy2kXlc31Cl4qUQLGyATAKJgVBNZPuszPkmOxu3A/view?pref=2&pli=1

    Parameters
    ----------
        desc1: descriptors from image 1, as returned by SIFT.compute()
        desc2: descriptors from image 2, same format as desc1

    Returns
    -------
        matches: a sequence (list) of cv2.DMatch objects containing corresponding descriptor indices
    """

    # TODO: Your code here
    # Note: You can use OpenCV's descriptor matchers, or roll your own!
    # Definitely NOT rolling my own... :)

    # Instantiate BFMatcher (brute-force matcher) object
    bfm = cv2.BFMatcher()

    # Compute matches with OpenCV library.  matches is a cv2.DMatch object.
    matches = bfm.match(desc1, desc2)

    return matches


def draw_matches(image1, image2, kp1, kp2, matches):
    """Show matches by drawing lines connecting corresponding keypoints.
    Reference: http://docs.opencv.org/3.0.0/d2/d29/classcv_1_1KeyPoint.html
    http://docs.opencv.org/3.0.0/d4/de0/classcv_1_1DMatch.html

    Parameters
    ----------
        image1: first image
        image2: second image, same type as first
        kp1: list of keypoints (cv2.KeyPoint objects) found in image1
        kp2: list of keypoints (cv2.KeyPoint objects) found in image2
        matches: list of matching keypoint index pairs (as cv2.DMatch objects)

    Returns
    -------
        image_out: image1 and image2 joined side-by-side with matching lines; color image (BGR), uint8, values in [0, 255]
    """

    # TODO: Your code here
    # Note: DO NOT use OpenCV's match drawing function(s)! Write your own :)

    # Normalize image_out from 0 to 255 range
    image1 = normalize_array(image1, 0, 255)
    image2 = normalize_array(image2, 0, 255)

    # Convert data type of image out to 'uint8' to make it work with cv2.cvtColor function
    image1 = np.array(image1, dtype='uint8')
    image2 = np.array(image2, dtype='uint8')

    # Convert image out to 3-channel color image
    image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
    image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)

    num_cols = image1.shape[1]

    # Join images
    image_out = make_image_pair(image1, image2)

    # for i in range(len(kp1)):
    #     print('kp1[' + str(i) + '] = ' + str(kp1[i].pt))

    for match in matches:
        #print('queryIdx = ' + str(match.queryIdx))
        query_idx = match.queryIdx
        train_idx = match.trainIdx

        # Get individual points from keypoint lists
        pt1 = kp1[query_idx].pt
        pt2 = kp2[train_idx].pt

        # Convert float to int in point tuples. Modify pt2 to add num_cols to x coordinate
        pt1 = (int(pt1[0]), int(pt1[1]))
        pt2 = (int(pt2[0]) + num_cols, int(pt2[1]))

        # Generate random colors for the blue, green, red channels.
        b = np.random.randint(0, 256)
        g = np.random.randint(0, 256)
        r = np.random.randint(0, 256)

        # Draw circles on matching pair end points
        cv2.circle(image_out, pt1, 3, (b, g, r), 1)
        cv2.circle(image_out, pt2, 3, (b, g, r), 1)

        # Draw connecting line between match points
        cv2.line(image_out, pt1, pt2, (b, g, r), 1)

    return image_out


def compute_translation_RANSAC(kp1, kp2, matches):
    """Compute best translation vector using RANSAC given keypoint matches.

    Parameters
    ----------
        kp1: list of keypoints (cv2.KeyPoint objects) found in image1
        kp2: list of keypoints (cv2.KeyPoint objects) found in image2
        matches: list of matches (as cv2.DMatch objects)

    Returns
    -------
        translation: translation/offset vector <x, y>, NumPy array of shape (2, 1)
        good_matches: consensus set of matches that agree with this translation
    """

    # TODO: Your code here
    # Calculate number of samples, N.
    N = compute_N(p=0.99, e=0.6, s=2)

    num_matches = len(matches)
    threshold = 7

    translations = list()
    avg_distances = list()

    # Instantiate list of N empty lists
    consensus_sets = [[] for i in range(N)]

    for i in range(N):
        distance_accumulator = 0

        # Generate random pair of points from a match
        match_idx = np.random.randint(0, num_matches)
        match = matches[match_idx]
        pt1, pt2 = get_points_from_match(match, kp1, kp2)

        # Compute translation
        dx = pt2[0] - pt1[0]
        dy = pt2[1] - pt1[1]
        translations.append((dx, dy))

        # Build consensus set by adding all points to set that match the translation within threshold
        for j in range(num_matches):
            pt1, pt2 = get_points_from_match(matches[j], kp1, kp2)

            # Compute translation
            dx_m = pt2[0] - pt1[0]
            dy_m = pt2[1] - pt1[1]

            # Check if translation is within threshold by comparing euclidean distance
            d = spatial.distance.euclidean((dx, dy), (dx_m, dy_m))
            if d <= threshold:
                consensus_sets[i].append(matches[j])
                distance_accumulator += d

        avg_distance = distance_accumulator / len(consensus_sets[i])
        avg_distances.append(avg_distance)
        print('Average distance for consensus set ' + str(i) + ' is ' + str(avg_distance))

    best_set_idx = get_best_set_idx(consensus_sets, avg_distances)

    translation = translations[best_set_idx]

    good_matches = consensus_sets[best_set_idx].copy()

    dx_sum = 0
    dy_sum = 0

    for i in range(len(good_matches)):
        pt1_1, pt1_2 = get_points_from_match(good_matches[i], kp1, kp2)

        # Compute translation
        dx = pt1_2[0] - pt1_1[0]
        dy = pt1_2[1] - pt1_1[1]

        dx_sum += dx
        dy_sum += dy

    avg_dx = dx_sum / len(good_matches)
    avg_dy = dy_sum / len(good_matches)

    avg_dx = int(round(avg_dx))
    avg_dy = int(round(avg_dy))

    translation = (avg_dx, avg_dy)

    # Prints for debugging outputs
    for i in range(len(consensus_sets)):
        print('Number of matches in consensus set ' + str(i) + ' is ' + str(len(consensus_sets[i])))
    print('Best average translation is ' + str(translation))
    print('best_set_idx = ' + str(best_set_idx))
    print('Best translation is ' + str(translation))
    print('Number of matches in good_matches is ' + str(len(good_matches)))

    return translation, good_matches


def get_best_set_idx(consensus_sets, avg_distances):
    """
    Computes the best set index based on the largest set size and the breaks ties using lowest average distance from
    the match point.  The order of consensus_sets and avg_distances correspond to each other by index.
    :param consensus_sets: list of consensus sets which are lists of DMatch objects
    :param avg_distances: list of average distances for each set
    :return: integer index argument for the best set
    """
    best_sets_indices = list()
    most_matches = 0
    most_matches_changed = False

    # Pick the indices for the consensus sets which have the largest size (# of matches)
    while True:
        for i in range(len(consensus_sets)):
            num_matches_C_i = len(consensus_sets[i])
            #print('num_matches_C_i = ' + str(num_matches_C_i))

            # If the size of set C_i is greater than most_matches, update most_matches with the size of that set and
            # append i as the index for that set to best_sets_indices
            if num_matches_C_i > most_matches:
                most_matches = num_matches_C_i
                most_matches_changed = True
                best_sets_indices.clear()
                best_sets_indices.append(i)

            # Handle case where num_matches_C_i is equal to most_matches
            if num_matches_C_i == most_matches:
                # If that match index is not in best_sets_indices, then append it to the list
                if best_sets_indices.count(i) == 0:
                    best_sets_indices.append(i)

        if not most_matches_changed:
            break

        most_matches_changed = False

    #print('best_sets_indices = ' + str(best_sets_indices))

    # Sort best_sets_indices using on average distance as sort key
    sorted_indices = list()
    for idx in best_sets_indices:
        sorted_indices.append((idx, avg_distances[idx]))
    sorted_indices = sorted(sorted_indices, key=lambda avg_distance: avg_distance[1])

    #print('sorted_indices = ' + str(sorted_indices))

    # Return the top set index from sorted_indices
    return sorted_indices[0][0]


def get_points_from_match(match, kp1, kp2):
    """
    Gets the matching point coordinates vectors from the kp1 and kp2 as defined in the DMatch object
    :param match: DMatch object
    :param kp1: first list of keypoints
    :param kp2: second list of keypoints
    :return pt1: point coordinate from kp1
    :return pt2: point coordinate from kp2
    """
    query_idx = match.queryIdx
    train_idx = match.trainIdx
    pt1 = kp1[query_idx].pt
    pt2 = kp2[train_idx].pt
    return pt1, pt2


def print_transform_outputs(matches, consensus_sets, avg_distances, best_set_idx, transform_vectors, best_fit_vector,
                            transform):
    for i in range(len(avg_distances)):
        print('Average distance for consensus set ' + str(i) + ' is ' + str(avg_distances[i]))

    for i in range(len(consensus_sets)):
        print('Number of matches in consensus set ' + str(i) + ' is ' + str(len(consensus_sets[i])))

    match_percentage = (len(consensus_sets[best_set_idx]) / len(matches)) * 100
    print('Biggest consensus set match percentage is ' + str(match_percentage))
    print('best_set_idx = ' + str(best_set_idx))
    print('transform vector at best_set_idx is ' + str(transform_vectors[best_set_idx]))
    print('best_fit_vector = ' + str(best_fit_vector))
    print('best fit transform below:')
    print(transform)


def compute_similarity_RANSAC(kp1, kp2, matches):
    """Compute best similarity transform using RANSAC given keypoint matches.

    Parameters
    ----------
        kp1: list of keypoints (cv2.KeyPoint objects) found in image1
        kp2: list of keypoints (cv2.KeyPoint objects) found in image2
        matches: list of matches (as cv2.DMatch objects)

    Returns
    -------
        transform: similarity transform matrix, NumPy array of shape (2, 3)
        good_matches: consensus set of matches that agree with this transform
    """
    # TODO: Your code here
    # Calculate number of samples, N.
    N = compute_N(p=0.99, e=0.5, s=4)
    threshold = 4
    transform_vectors, avg_distances, consensus_sets = compute_consensus_sets(N, matches, kp1, kp2, threshold,
                                                                              transform_type='similarity')

    best_set_idx = get_best_set_idx(consensus_sets, avg_distances)

    good_matches = consensus_sets[best_set_idx]

    # Compute least squares solution for transform matrix using all points in the consensus set
    best_fit_vector = compute_transform_vector(good_matches, kp1, kp2)

    transform = get_transform_matrix(best_fit_vector, transform_type='similarity')

    # Prints for debugging outputs
    print_transform_outputs(matches, consensus_sets, avg_distances, best_set_idx, transform_vectors, best_fit_vector,
                            transform)

    return transform, good_matches


def compute_affine_RANSAC(kp1, kp2, matches):
    # Calculate number of samples, N.
    N = compute_N(p=0.99, e=0.5, s=6)
    threshold = 4
    transform_vectors, avg_distances, consensus_sets = compute_consensus_sets(N, matches, kp1, kp2, threshold,
                                                                              transform_type='affine')
    best_set_idx = get_best_set_idx(consensus_sets, avg_distances)
    good_matches = consensus_sets[best_set_idx]

    # Compute least squares solution for transform matrix using all points in the consensus set
    best_fit_vector = compute_transform_vector(good_matches, kp1, kp2, transform_type='affine')

    transform = get_transform_matrix(best_fit_vector, transform_type='affine')

    # Prints for debugging outputs
    print_transform_outputs(matches, consensus_sets, avg_distances, best_set_idx, transform_vectors, best_fit_vector,
                            transform)

    return transform, good_matches


def compute_consensus_sets(N, matches, kp1, kp2, threshold, transform_type='similarity'):
    num_matches = len(matches)
    transform_vectors = list()
    avg_distances = list()

    # Instantiate list of N empty lists
    consensus_sets = [[] for i in range(N)]

    for i in range(N):
        # Generate three random pairs of points from three matches
        match_indices = np.arange(num_matches)
        np.random.shuffle(match_indices)

        # Handle the number of samples pair points drawn for computations. 2 pairs for similarity and 3 for affine.
        if transform_type == 'similarity':
            sample_matches = [matches[match_indices[0]], matches[match_indices[1]]]
        elif transform_type == 'affine':
            sample_matches = [matches[match_indices[0]], matches[match_indices[1]], matches[match_indices[2]]]
        else:
            print('Error: missing transform_type parameter for compute_consensus_sets!')

        transform_vector = compute_transform_vector(sample_matches, kp1, kp2, transform_type=transform_type)
        transform_vectors.append(transform_vector)

        consensus_sets[i], distance_accumulator = compute_consensus_set(matches, kp1, kp2, threshold, transform_vector,
                                                                        transform_type=transform_type)
        # Handle possible division by zero errors
        if len(consensus_sets[i]) == 0:
            avg_distance = distance_accumulator
        else:
            avg_distance = distance_accumulator / len(consensus_sets[i])
        avg_distances.append(avg_distance)
        #print('Average distance for consensus set ' + str(i) + ' is ' + str(avg_distance))

    return transform_vectors, avg_distances, consensus_sets


def compute_consensus_set(matches, kp1, kp2, threshold, transform_vector, transform_type='similarity'):
    """
    Computes one consensus set based on the parameters provided
    :param matches: list of all matches
    :param kp1: first set of keypoints
    :param kp2: second set of keypoints
    :param threshold: the maximum Euclidean distance that a match point should be from the computed point based on
                      the transform computation
    :param transform_vector: the vector with the components to build the transform matrix
    :param transform_type: the type of transform to be computed
    :return: consensus_set: a list of of the DMatch objects which match within the threshold
    :return: distance_accumulator: the sum of the distances by which the computed points and match points differ
    """
    consensus_set = list()
    distance_accumulator = 0
    for i in range(len(matches)):
        transform_matrix = get_transform_matrix(transform_vector, transform_type=transform_type)

        pt1, pt2 = get_points_from_match(matches[i], kp1, kp2)

        x1 = pt1[0]
        y1 = pt1[1]

        x2 = pt2[0]
        y2 = pt2[1]

        # Assign homogeneous coordinate
        homogeneous_pt = np.array([x1, y1, 1])

        # Compute transform
        transformed_points = np.dot(transform_matrix, homogeneous_pt)

        # Check if translation is within threshold by comparing euclidean distance
        d = spatial.distance.euclidean((x2, y2), transformed_points)
        if d <= threshold:
            consensus_set.append(matches[i])
            distance_accumulator += d

    return consensus_set, distance_accumulator


def compute_transform_vector(matches, kp1, kp2, transform_type='similarity'):
    """
    Computes the transform vector for the similarity transform by doing a linear least-squares best fit.
    :param matches: List of matches on which to compute the transform vector.
        There must be at least two matches in the list for similarity and at least three for affine
    :param kp1: first set of keypoints from which to compute the transform vector
    :param kp2: second set of keypoints
    :param transform_type: The type of transform to compute.  Should be either 'similarity' or 'affine'
    :return: transform_vector: the vector which can be used to compute a similarity transform
    """
    num_matches = len(matches)
    A = [list() for i in range(num_matches * 2)]
    b = list()
    for i in range(num_matches):
        pt1, pt2 = get_points_from_match(matches[i], kp1, kp2)
        u = pt1[0]
        v = pt1[1]

        x = pt2[0]
        y = pt2[1]

        # Load next two rows in matrix A
        if transform_type == 'similarity':
            A[i * 2]     = [u, -v, 1, 0]
            A[i * 2 + 1] = [v,  u, 0, 1]
        elif transform_type == 'affine':
            A[i * 2]     = [u, v, 1, 0, 0, 0]
            A[i * 2 + 1] = [0, 0, 0, u, v, 1]
        else:
            print('Error: compute_transform_vector is missing transform_type parameter!')

        # Load next two rows in vector b
        b.append(x)
        b.append(y)

    transform_vector, residuals, rank, s = np.linalg.lstsq(A, b)
    return transform_vector


def get_transform_matrix(transform_vector, transform_type='similarity'):
    """
    This function rearranges the tranform vector in the transform matrix which can be used to compute the
    inhomogeneous coordinates from the similarity transform by multiplying the transform matrix with the
    homogeneous coordinates.
    :param transform_vector: type np.array as [a, b, c, d]
    :param transform_type: The type of transform matrix to create.  Should be either 'similarity' or 'affine'
    :return: transform_matrix: Array with components in matrix form which computes the transform on a homogeneous
        coordinate.
    """

    if transform_type == 'similarity':
        a = transform_vector[0]
        b = transform_vector[1]
        c = transform_vector[2]
        d = transform_vector[3]
        transform_matrix = np.array([[a, -b, c],
                                     [b,  a, d]])
    elif transform_type == 'affine':
        a = transform_vector[0]
        b = transform_vector[1]
        c = transform_vector[2]
        d = transform_vector[3]
        e = transform_vector[4]
        f = transform_vector[5]
        transform_matrix = np.array([[a, b, c],
                                     [d, e, f]])
    else:
        print('Error: get_transform_matrix is missing transform_type parameter!')

    return transform_matrix


def compute_N(p, e, s):
    """

    :param p: Probability of success (usually p = 0.99)
    :param e: Proportion of outliers
    :param s: Number of points needed to compute a solution
    :return:
    N: the number of samples to compute such that the probability of success is greater than the probability that
    all samples in the set have outliers.
    """
    N = np.log(1 - p) / np.log(1 - (1 - e) ** s)
    N = int(np.ceil(N))
    return N


def get_square_gaussian(ksize, sigma):
    """
    Computes a square (ksize by ksize) Gaussian kernel
    :param ksize: size of the kernel
    :param sigma: standard deviation of kernel
    :return: square_gaussian
    """
    k = cv2.getGaussianKernel(ksize, sigma)
    square_gaussian = np.dot(k, k.transpose())

    return square_gaussian


def warp_image(img_in, transform):
    """
    Warps the image with given affine transform matrix
    :param img_in: numpy.array which is single channel (grayscale)
    :param transform: 2x3 transform matrix as a numpy.array
    :return img_out: Normalized image with values from 0 to 255
    """
    img_out = np.zeros(img_in.shape)
    num_rows = img_out.shape[0]
    num_cols = img_out.shape[1]

    #start = time.time()

    # Perform inverse warp with transform matrix.  Use (x, y) for img_out and (u, v) for img_in.
    for x in range(num_cols):
        for y in range(num_rows):
            img_out_coordinate = (x, y, 1)
            image_coordinate = np.dot(transform, img_out_coordinate)
            u = image_coordinate[0]
            v = image_coordinate[1]

            # Handle out of bounds coordinates
            if 0 <= u < num_cols and 0 <= v < num_rows:
                #u = int(round(u))
                #v = int(round(v))
                #img_out[y, x] = img_in[v, u]
                img_out[y, x] = bilinear_interpolation(img_in, v, u)

    #end = time.time()
    #run_time = end - start
    #print('Run time for warp loop is ' + str(run_time) + 's')

    #img_out = cv2.warpAffine(img_in, t_inv, (num_cols, num_rows))

    # Normalize img_in array
    img_out = normalize_array(img_out, 0, 255)

    return img_out


def bilinear_interpolation(img, i, j):
    """
    Perform bilinear interpolation.  Reference 3D-L2 slide.  Modified to get correct i down orientation values and use
    (i, j) where i is for rows and j is for columns (i.e. exchange i and j from slides)
    :param image:
    :param i:
    :param j:
    :return:
    """
    num_rows = img.shape[0]
    num_cols = img.shape[1]

    # Compute the portion left after decimal point
    a = j - int(j)
    b = i - int(i)

    if i > num_rows - 1 or j > num_cols - 1:
        pixel = img[int(i), int(j)]
    else:
        # Compute bilinear interpolation with i increasing downwards
        pixel = img[i, j] * (1-a)*(1-b) + \
                img[i+1, j] * (1-a)*b + \
                img[i, j+1] * a*(1-b) + \
                img[i+1, j+1] * a*b

    return pixel


def overlay_images(img_1, img_2):
    """
    Overlays two images on top of each other.  Image 1 goes in red channel and image 2 goes in green channel
    :param img_1: numpy.array, single channel
    :param img_2: numpy.array, single channel
    :return img_out: overlaid image array
    """
    img_out = np.zeros((img_1.shape[0], img_1.shape[1], 3))
    img_1 = normalize_array(img_1, 0, 255)
    img_2 = normalize_array(img_2, 0, 255)

    img_out[:, :, 2] = img_1
    img_out[:, :, 1] = img_2
    return img_out


def main():
    # Driver code
    ida = cv2.imread("ida.jpg")
    patch = ida[500:580, 450:575]
    cv2.imshow('patch', patch)  # [debug]
    cv2.waitKey(0)

    patch_hsv = np.zeros((patch.shape))
    patch_hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    # print(patch_hsv)

    # Hues will be stored in OpenCV in 8-bit images as values ranging from 0 to 179
    hues = np.array(patch_hsv[:, :, 0], dtype='float32')

    # Need to normalize hues in the 0 to 360 range by multiplying outputs by 2
    hues = hues * 2

    # print(hues)
    # print(hues.max())

    # Estimating that the usable part of the visible spectrum is 450-620nm,
    # with wavelength (in microns) and hue value (in degrees)
    wavelengths = 620 - 170 / 270 * hues

    # Divide by 1000 to normalize to microns
    wavelengths = wavelengths / 1000
    # print(wavelengths)

    # Apply Gaussian blur to filter out noise on image data
    kernel_size = (5, 5)
    wavelengths = cv2.GaussianBlur(wavelengths, kernel_size, 0)
    print('wavelengths.min() = ' + str(wavelengths.min()))
    print('wavelengths.max() = ' + str(wavelengths.max()))
    print('wavelengths.mean() = ' + str(wavelengths.mean()))
    print('wavelengths.std() = ' + str(wavelengths.std()))

    # print(wavelengths)

    hist = cv2.calcHist([wavelengths], [0], None, [50], [0.585, 0.615]).flatten()
    plt.plot(hist)
    plt.title('Wavelengths Histogram')
    plt.ylabel('Count')
    plt.yscale('log')
    plt.xlabel('Bins where bin 0 is 550 microns')

    # ticks = np.arange(500, 700, 5)
    # labels = range(ticks.size)
    # print('ticks = ' + str(ticks))
    # plt.xticks(ticks, labels)
    plt.show()


if __name__ == "__main__":
    main()
