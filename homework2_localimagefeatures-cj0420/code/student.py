from os import W_OK
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, feature, img_as_int
from skimage.measure import regionprops

def plot_interest_points(image, x, y):
    '''
    Plot interest points for the input image. 
    
    Show the interest points given on the input image. Be sure to add the images you make to your writeup. 

    Useful functions: Some helpful (not necessarily required) functions may include
        - matplotlib.pyplot.imshow, matplotlib.pyplot.scatter, matplotlib.pyplot.show, matplotlib.pyplot.savefig
    
    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :x: np array of x coordinates of interest points
    :y: np array of y coordinates of interest points
    '''

    # TODO: Your implementation here! See block comments and the homework webpage for instruction
    plt.imshow(image)
    
    # to draw a point on co-ordinate (200,300)
    plt.scatter(x, y, c='g', s=10)
    plt.show()

def get_feature_points(image, feature_width):
    '''
    Returns interest points for the input image

    (Please note that we recommend implementing this function last and using cheat_interest_points()
    to test your implementation of get_feature_descriptors() and match_features())

    Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
    You do not need to worry about scale invariance or keypoint orientation estimation
    for your Harris corner detector.
    You can create additional interest point detector functions (e.g. MSER)
    for extra credit.

    If you're finding spurious (false/fake) interest point detections near the boundaries,
    it is safe to simply suppress the grad / corners near the edges of
    the image.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on Piazza with any questions

        - skimage.feature.peak_local_max (experiment with different min_distance values to get good results)
        - skimage.measure.regionprops
          
    Note: You may decide it is unnecessary to use feature_width in get_feature_points, or you may also decide to 
    use this parameter to exclude the points near image edges.

    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :feature_width: the width and height of each local feature in pixels

    :returns:
    :xs: an np array of the x coordinates of the interest points in the image
    :ys: an np array of the y coordinates of the interest points in the image

    :optional returns (may be useful for extra credit portions):
    :confidences: an np array indicating the confidence (strength) of each interest point
    :scale: an np array indicating the scale of each interest point
    :orientation: an np array indicating the orientation of each interest point

    '''

    # TODO: Your implementation here! See block comments and the homework webpage for instructions

    # These are placeholders - replace with the coordinates of your interest points!

    xs = np.array([])
    ys = np.array([])

    # Note that xs and ys represent the coordinates of the image. Thus, xs actually denote the columns
    # of the respective points and ys denote the rows of the respective points.

    # STEP 1: Calculate the gradient (partial derivatives on two directions).
    dy, dx = np.gradient(image)
    dxx = dx**2
    dxy = dy*dx
    dyy = dy**2

    # STEP 2: Apply Gaussian filter with appropriate sigma.
    Ixx = filters.gaussian(dxx, sigma=1)
    Ixy = filters.gaussian(dxy, sigma=1)
    Iyy = filters.gaussian(dyy, sigma=1)

    # STEP 3: Calculate Harris cornerness score for all pixels.
    alpha = 0.05
    det = Ixx * Iyy - Ixy ** 2
    trace = Ixx + Iyy
    harris_response = det - alpha * trace ** 2

    harris_response[harris_response < 0] = 0

    # STEP 4: Peak local max to eliminate clusters. (Try different parameters.)
    cords = feature.peak_local_max(harris_response, min_distance=feature_width//2, num_peaks=500)
    xs = cords[:,1]
    ys = cords[:,0]

    
    # BONUS: There are some ways to improve:
    # 1. Making interest point detection multi-scaled.
    # 2. Use adaptive non-maximum suppression.

    return xs, ys


def get_feature_descriptors(image, x_array, y_array, feature_width):
    '''
    Returns features for a given set of interest points.

    To start with, you might want to simply use normalized patches as your
    local feature descriptor. This is very simple to code and works OK. However, to get
    full credit you will need to implement the more effective SIFT-like feature descriptor
    (See Szeliski 4.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)

    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) feature descriptor should have:
    (1) a 4x4 grid of cells, each feature_width / 4 pixels square.
    (2) each cell should have a histogram of the local distribution of
        grad in 8 grad_proc_2. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length

    You do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions. This type
    of interpolation probably will help, though.

    You do not have to explicitly compute the gradient orientation at each
    pixel (although you are free to do so). You can instead filter with
    oriented filters (e.g. a filter that responds to edges with a specific
    orientation). All of your SIFT-like features can be constructed entirely
    from filtering fairly quickly in this way.

    You do not need to do the normalize -> threshold -> normalize again
    operation as detailed in Szeliski and the SIFT paper. It can help, though.

    Another simple trick which can help is to raise each element of the final
    feature vector to some power that is less than one.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on Piazza with any questions

        - skimage.filters (library)


    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :x: np array of x coordinates of interest points
    :y: np array of y coordinates of interest points
    :feature_width: in pixels, is the local feature width. You can assume
                    that feature_width will be a multiple of 4 (i.e. every cell of your
                    local SIFT-like feature will have an integer width and height).
    If you want to detect and describe features at multiple scales or
    particular grad_proc_2 you can add input arguments. Make sure input arguments 
    are optional or the autograder will break.

    :returns:
    :features: numpy array of computed features. It should be of size
            [num points * feature dimensionality]. For standard SIFT, `feature
            dimensionality` is 128. `num points` may be less than len(x) if
            some points are rejected, e.g., if out of bounds.

    '''

    # TODO: Your implementation here! See block comments and the homework webpage for instructions
    
    # STEP 1: Calculate the gradient (partial derivatives on two directions) on all pixels.


    # STEP 2: Decompose the gradient vectors to magnitude and direction.
    # STEP 3: For each interest point, calculate the local histogram based on related 4x4 grid cells.
    #         Each cell is a square with feature_width / 4 pixels length of side.
    #         For each cell, we assign these gradient vectors corresponding to these pixels to 8 bins
    #         based on the direction (angle) of the gradient vectors. 
    # STEP 4: Now for each cell, we have a 8-dimensional vector. Appending the vectors in the 4x4 cells,
    #         we have a 128-dimensional feature.
    # STEP 5: Don't forget to normalize your feature.
    
    # BONUS: There are some ways to improve:
    # 1. Use a multi-scaled feature descriptor.
    # 2. Borrow ideas from GLOH or other type of feature descriptors.


    #x = np.round(x_array).astype(int).flatten()
    #y = np.round(y_array).astype(int).flatten()
    bin_width = 2 * np.pi / 8
    bins = np.arange(-np.pi, np.pi, bin_width)
    print(bins)

    filtered_image = filters.gaussian(image, sigma=1.0)
    fw = feature_width//2

    grad = np.gradient(filtered_image)
    grad_proc = np.linalg.norm(grad, axis=0)
    grad_proc_2 = np.arctan2(grad[0], grad[1])
    feature_list = []
    features = np.zeros(1)

    for x, y in zip(x_array, y_array):
        w_m = grad_proc[y-fw+1 : y+fw+1, x-fw+1 : x+fw+1]
        if w_m.shape != (feature_width, feature_width):
            continue
        w_o = grad_proc_2[y-fw+1 : y+fw+1, x-fw+1 : x+fw+1]

        pwm = np.array(np.split(np.array(np.split(w_m, 4, axis=1)).reshape(4, -1), 4, axis=1,)).reshape(-1, feature_width)
        pwo = np.array(np.split(np.array(np.split(w_o, 4, axis=1)).reshape(4, -1), 4, axis=1,)).reshape(-1, feature_width)

        feature = np.zeros(int(feature_width * feature_width * 8 / 16))
        for subwindow_i in range(len(pwm)):
            ind=[]
            
            for val in pwo[subwindow_i]:
                f = True
                for idx, ck in enumerate(bins):
                    if val <= ck:
                        ind.append(idx)
                        f = False
                        break
                if f:
                    ind.append(len(bins))

            inds=np.array(ind)
            for inds_i in range(8):
                mask = np.array(inds == inds_i)
                feature[subwindow_i * 8 + inds_i] = np.sum(
                    pwm[subwindow_i].flatten()[mask]
                ) 

        feature = feature ** 0.6
        feature_norm = feature / np.linalg.norm(feature)
        threshold = np.percentile(feature_norm, [60.0])
        np.putmask(feature_norm, feature_norm < threshold, 0)
        feature_norm = feature_norm ** 0.7  
        feature_norm_2 = feature_norm / np.linalg.norm(feature_norm)
        feature_list.append(feature_norm_2)

        features = np.array(feature_list)

    return features


def match_features(im1_features, im2_features):
    '''
    Implements the Nearest Neighbor Distance Ratio Test to assign matches between interest points
    in two images.

    Please implement the "Nearest Neighbor Distance Ratio (NNDR) Test" ,
    Equation 4.18 in Section 4.1.3 of Szeliski.

    For extra credit you can implement spatial verification of matches.

    Please assign a confidence, else the evaluation function will not work. Remember that
    the NNDR test will return a number close to 1 for feature points with similar distances.
    Think about how confidence relates to NNDR.

    This function does not need to be symmetric (e.g., it can produce
    different numbers of matches depending on the order of the arguments).

    A match is between a feature in im1_features and a feature in im2_features. We can
    represent this match as a the index of the feature in im1_features and the index
    of the feature in im2_features

    :params:
    :im1_features: an np array of features returned from get_feature_descriptors() for interest points in image1
    :im2_features: an np array of features returned from get_feature_descriptors() for interest points in image2

    :returns:
    :matches: an np array of dimension k x 2 where k is the number of matches. The first
            column is an index into im1_features and the second column is an index into im2_features
    :confidences: an np array with a real valued confidence for each match
    '''

    # TODO: Your implementation here! See block comments and the homework webpage for instructions

    # These are placeholders - replace with your matches and confidences!
    
    # STEP 1: Calculate the distances between each pairs of features between im1_features and im2_features.
    #         HINT: https://browncsci1430.github.io/webpage/hw2_featurematching/efficient_sift/
    # I get D 2d array 

    # STEP 2: Sort and find closest features for each feature, then performs NNDR test.
    # np.sort(array, axis=1)
    
    # BONUS: Using PCA might help the speed (but maybe not the accuracy).

    f1 = np.sum(np.square(im1_features), axis=1, keepdims=True)
    f2 = np.sum(np.square(im2_features), axis=1, keepdims=True).transpose()

    A = f1 + f2
    B = 2 * np.matmul(im1_features, im2_features.transpose())
    D = np.sqrt(A - B)

    idx1 = np.sort(D, axis=1)[:, :2]
    nddr1 = idx1[:, 0] / idx1[:, 1]
    confidences1 = 1 - nddr1
    closest_ind1 = np.argmin(D, axis=1)
    match_1 = np.stack((np.arange(D.shape[0]), closest_ind1), axis=1)

    idx2 = np.sort(D.T, axis=1)[:, :2]
    nddr2 = idx2[:, 0] / idx2[:, 1]
    confidences2 = 1 - nddr2
    closest_ind2 = np.argmin(D.T, axis=1)
    match_2 = np.stack((closest_ind2, np.arange(D.T.shape[0])), axis=1)

    # cross check match pair
    idx_pair = match_1[match_2[:, 0]][:, 1] == match_2[:, 1]
    match = match_2[idx_pair]
    confidence = (confidences1[match_2[idx_pair, 0]] + confidences2[idx_pair]) / 2

    matches = np.array(match)
    confidences = np.array(confidence)

    return matches, confidences
