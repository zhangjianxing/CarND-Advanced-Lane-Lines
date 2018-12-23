import numpy as np


def poly_to_polycr(line_fit, ploty, ym_per_pix, xm_per_pix):
    ### Calc both polynomials using ploty, left_fit and right_fit ###
    if line_fit is None:
        return None

    line_fitx = line_fit[0] * ploty ** 2 + line_fit[1] * ploty + line_fit[2]
    line_fit_cr = np.polyfit(ploty * ym_per_pix, line_fitx * xm_per_pix, 2)
    return line_fit_cr


def measure_real_world_curvature_for_line(line_fit, ploty, ym_per_pix=30 / 720, xm_per_pix=3.7 / 700):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)

    line_fit_cr = poly_to_polycr(line_fit, ploty, ym_per_pix, xm_per_pix)
    if line_fit_cr is None:
        return None
    curvature = (((1 + (2 * line_fit_cr[0] * y_eval * ym_per_pix + line_fit_cr[1]) ** 2) ** 1.5)
                 / np.absolute(2 * line_fit_cr[0]))
    return curvature


def avg_curvature(left_curvature, right_curvature):
    if left_curvature is not None and right_curvature is not None:
        return np.mean([left_curvature, right_curvature])
    elif left_curvature is not None and right_curvature is None:
        return left_curvature
    elif left_curvature is None and right_curvature is not None:
        return right_curvature
    else:
        # print('no curvature found')
        return None


def measure_real_world_curvature_for_road(left_fit, right_fit, ploty, ym_per_pix=30 / 720, xm_per_pix=3.7 / 700):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
    # Calculation of R_curve (radius of curvature)
    left_curvature, right_curvature = None, None
    if left_fit is not None:
        left_curvature = measure_real_world_curvature_for_line(left_fit, ploty, ym_per_pix, xm_per_pix)
    if right_fit is not None:
        right_curvature = measure_real_world_curvature_for_line(right_fit, ploty, ym_per_pix, xm_per_pix)

    return avg_curvature(left_curvature, right_curvature)


def measure_mid_position(left_fit, right_fit, ploty, std_mid=660, imgsize_x=1200, xm_per_pix=3.7 / 700):
    y_eval = np.max(ploty)
    # find mid point
    leftx = left_fit[0] * y_eval ** 2 + left_fit[1] * y_eval + left_fit[2] if left_fit is not None else 0
    rightx = right_fit[0] * y_eval ** 2 + right_fit[1] * y_eval + right_fit[2] if right_fit is not None else imgsize_x

    current_mid_pix = (leftx + rightx) / 2
    to_mid_of_line = (std_mid - current_mid_pix) * xm_per_pix
    return to_mid_of_line
