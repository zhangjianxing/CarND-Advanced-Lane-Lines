import numpy as np
import cv2


def find_lane_pixels_from_scratch(binary_warped, midpoint=None, nwindows=9, margin=100, minpix=50, draw_out_img=False):
    """

    :param binary_warped: input image
    :param nwindows: Choose the number of sliding windows
    :param margin: Set the width of the windows +/- margin
    :param minpix: Set minimum number of pixels found to recenter window
    :return:
    """
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) if draw_out_img else None
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = midpoint if midpoint else np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0] // nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        if draw_out_img:
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 3)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 3)

        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    ## Visualization ##
    # Colors in the left and right lane regions
    if draw_out_img:
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]

    return leftx, lefty, rightx, righty, out_img


def find_lane_pixels_around_old_fit(binary_warped, fit, margin=100):
    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    ### Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    lane_inds = ((nonzerox > (fit[0] * (nonzeroy ** 2) + fit[1] * nonzeroy + fit[2] - margin)) &
                 (nonzerox < (fit[0] * (nonzeroy ** 2) + fit[1] * nonzeroy + fit[2] + margin)))

    # Again, extract left and right line pixel positions
    xs = nonzerox[lane_inds]
    ys = nonzeroy[lane_inds]
    return xs, ys


def fit_poly_from_pixels(img_shape, xs, ys):
    fitx, fit = None, None
    ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])
    if len(xs) and len(xs) == len(ys):
        ### Fit a second order polynomial to each with np.polyfit() ###
        fit = np.polyfit(ys, xs, deg=2)
        ### Calc both polynomials using ploty, left_fit and right_fit ###
        fitx = fit[0] * ploty ** 2 + fit[1] * ploty + fit[2]
    return fitx, fit


def _fit_poly(img_shape, leftx, lefty, rightx, righty):
    ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])
    left_fitx, left_fit = fit_poly_from_pixels(img_shape, leftx, lefty)
    right_fitx, right_fit = fit_poly_from_pixels(img_shape, rightx, righty)

    return left_fitx, right_fitx, left_fit, right_fit, ploty


def search_poly_from_scratch(binary_warped, draw_out_img=False):
    leftx, lefty, rightx, righty, out_img = find_lane_pixels_from_scratch(binary_warped, draw_out_img=draw_out_img)
    left_fitx, right_fitx, left_fit, right_fit, ploty = _fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)

    return left_fitx, right_fitx, left_fit, right_fit, ploty, out_img


def search_poly_around_old(binary_warped, left_fit, right_fit, margin=100, draw_out_img=False):
    """
    :param binary_warped:
    :param left_fit:
    :param right_fit:
    :param margin: Choose the width of the margin around the previous polynomial to search
    :return: left_fitx, right_fitx, ploty
    """
    # Grab activated pixels
    leftx, lefty = find_lane_pixels_around_old_fit(binary_warped, left_fit, margin)
    rightx, righty = find_lane_pixels_around_old_fit(binary_warped, right_fit, margin)

    # Fit new polynomials
    left_fitx, right_fitx, left_fit, right_fit, ploty = _fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)

    ## Visualization ##
    if draw_out_img:
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        # Color in left and right line pixels
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]
        # Color in region
        window_img = visualize_poly(out_img, left_fitx, right_fitx, ploty, margin=100)
        out_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        # Plot the polynomial lines onto the image
        cv2.polylines(out_img, [np.dstack((ploty, left_fitx))], False, color='yellow')
        cv2.polylines(out_img, [np.dstack((ploty, right_fitx))], False, color='yellow')
        ## End visualization steps ##
    else:
        out_img = None

    return left_fitx, right_fitx, left_fit, right_fit, ploty, out_img


def visualize_poly(img, left_fitx, right_fitx, ploty, margin=None):
    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    window_img = np.zeros_like(img)
    if margin:
        left_line_window1 = np.dstack([left_fitx - margin, ploty])
        left_line_window2 = np.dstack([left_fitx + margin, ploty])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))

        right_line_window1 = np.dstack([right_fitx - margin, ploty])
        right_line_window2 = np.dstack([right_fitx + margin, ploty])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_(left_line_pts), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_(right_line_pts), (0, 255, 0))
    elif left_fitx is not None and right_fitx is not None:
        line_window1 = np.dstack([left_fitx, ploty])[0]
        line_window2 = np.dstack([right_fitx, ploty])[0][::-1]
        line_pts = np.int_([np.vstack([line_window1, line_window2])])
        cv2.fillPoly(window_img, line_pts, (0, 255, 0))

    return window_img
