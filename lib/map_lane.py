from lib.camera_calibration import generate_undistort_matrix, undistort_img
from lib.color_and_gradient import img_to_binary, img_to_gray
from lib.find_line_poly import search_poly_from_scratch, find_lane_pixels_from_scratch, find_lane_pixels_around_old_fit, \
    visualize_poly, fit_poly_from_pixels
from lib.measure_curve import measure_real_world_curvature_for_road, measure_mid_position, \
    measure_real_world_curvature_for_line, avg_curvature
from lib.utils import add_text_to_img, get_perspective_transform_meta_data
import cv2
import numpy as np


class Line(object):
    def __init__(self, fit, ploty, allx, ally, is_left=None):
        assert allx is not None
        assert ally is not None
        assert is_left in [True, False]
        # x values of the last n fits of the line
        self.current_fitx = fit[0] * ploty ** 2 + fit[1] * ploty + fit[2] if fit is not None else np.array([])
        self.best_fitx = np.array([])
        # average x values of the fitted line over the last n iterations
        self.current_base_x = self.current_fitx[-1] if len(self.current_fitx) else None
        self.best_base_x = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = fit
        # radius of curvature of the line in some units
        self.current_curvature = measure_real_world_curvature_for_line(fit, ploty)
        self.best_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        # self.diffs = np.array([0,0,0], dtype='float')
        # x values for detected line pixels
        self.allx = allx
        # y values for detected line pixels
        self.ally = ally

        # was the line detected in the last iteration?
        if self.current_base_x is None:
            self.detected = False
        else:
            std_mid = RoadLineDetector.STD_MID
            self.detected = self.current_base_x < std_mid if is_left else std_mid < self.current_base_x
        self.ploty = ploty

    def set_best_fit(self, best_fit, is_left_line=None):
        assert isinstance(is_left_line, bool)
        if best_fit is None:
            return
        self.best_fit = best_fit
        self.best_fitx = best_fit[0] * self.ploty ** 2 + best_fit[1] * self.ploty + best_fit[2]
        self.best_base_x = self.best_fitx[-1]
        self.line_base_pos = (RoadLineDetector.STD_MID - self.best_base_x) * RoadLineDetector.XM_PER_PIX
        if not is_left_line:
            self.line_base_pos = -self.line_base_pos
        self.best_curvature = measure_real_world_curvature_for_line(best_fit, self.ploty)


class RoadLineDetector(object):
    NUM_RECENT_LINES_KEEP_IN_MEM = 100
    NUM_RECENT_LINES_PROCESS = 3
    RECENT_LINES_MULTIPLIER = 5
    STD_MID = 660
    YM_PER_PIX = 30 / 720
    XM_PER_PIX = 3.7 / 700

    def __init__(self, mtx, dist, out_img_gray_img=False, out_img_show_pixels=False):
        self.mtx = mtx
        self.dist = dist
        self.transform_matrix = None
        self.transform_inv_matrix = None
        # detected lines
        self.left_lines = []  # type: list[Line]
        self.right_lines = []  # type: list[Line]

        self.out_img_gray_img = out_img_gray_img
        self.out_img_show_pixels = out_img_show_pixels

    def generate_transform_matrix(self, img_size):
        img_size_x, img_size_y = img_size
        _, _, transform_matrix, transform_inv_matrix = get_perspective_transform_meta_data(img_size_x, img_size_y)
        self.transform_matrix = transform_matrix
        self.transform_inv_matrix = transform_inv_matrix

    def get_transform_matrix(self, img_size):
        if self.transform_matrix is None:
            self.generate_transform_matrix(img_size)
        return self.transform_matrix

    def get_transform_inv_matrix(self, img_size):
        if self.transform_inv_matrix is None:
            self.generate_transform_matrix(img_size)
        return self.transform_inv_matrix

    def _find_best_fit_from_lines(self, lines):
        best_fit = np.zeros(3)
        total_weight = 0
        for i in range(1, min(len(lines), self.NUM_RECENT_LINES_PROCESS) + 1):
            line = lines[-i]
            if line.detected:
                total_weight += (1. / self.RECENT_LINES_MULTIPLIER) ** i
                best_fit += np.array(line.current_fit).reshape(3) * (1. / self.RECENT_LINES_MULTIPLIER) ** i
        if not total_weight:
            return None

        return best_fit / total_weight

    def do_cross(self, left_line: Line, right_line: Line):
        if left_line.detected and right_line.detected:
            for lx, rx in zip(left_line.current_fitx, right_line.current_fitx):
                if lx + 30 > rx:
                    return True
        return False

    def find_line(self, bindary_transformed):
        img_shape = bindary_transformed.shape

        ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])
        old_best_left_fit = None if not self.left_lines else self.left_lines[-1].best_fit
        old_best_right_fit = None if not self.right_lines else self.right_lines[-1].best_fit
        left_line, right_line = None, None

        if old_best_left_fit is not None:
            leftx, lefty = find_lane_pixels_around_old_fit(bindary_transformed, old_best_left_fit)
            left_fitx, left_fit = fit_poly_from_pixels(img_shape, leftx, lefty)
            left_line = Line(left_fit, ploty, leftx, lefty, is_left=True)
        if old_best_right_fit is not None:
            rightx, righty = find_lane_pixels_around_old_fit(bindary_transformed, old_best_right_fit)
            right_fitx, right_fit = fit_poly_from_pixels(img_shape, rightx, righty)
            right_line = Line(right_fit, ploty, rightx, righty, is_left=False)
        # if line cross, reset both to be None
        if left_line is not None \
                and right_line is not None \
                and self.do_cross(left_line, right_line):
            left_line, right_line = None, None

        if left_line is None or right_line is None:
            leftx, lefty, rightx, righty, out_img = find_lane_pixels_from_scratch(bindary_transformed)
            left_fitx, left_fit = fit_poly_from_pixels(img_shape, leftx, lefty)
            right_fitx, right_fit = fit_poly_from_pixels(img_shape, rightx, righty)
            if left_line is None:
                left_line = Line(left_fit, ploty, leftx, lefty, is_left=True)
            if right_line is None:
                right_line = Line(right_fit, ploty, rightx, righty, is_left=False)
        # if line cross, invalidate both
        if left_line is not None \
                and right_line is not None \
                and self.do_cross(left_line, right_line):
            left_line.detected, right_line.detected = False, False

        self.left_lines.append(left_line)
        self.right_lines.append(right_line)
        left_line.set_best_fit(self._find_best_fit_from_lines(self.left_lines), True)
        right_line.set_best_fit(self._find_best_fit_from_lines(self.right_lines), False)

        return bindary_transformed

    def map_lane(self, img, out_img_gray_img=None, out_img_show_pixels=None):
        img_shape = img.shape
        unimg = undistort_img(img, self.mtx, self.dist)
        bindary_img = img_to_binary(unimg)

        ## transform img
        img_size = img.shape[1::-1]
        ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])

        transform_matrix = self.get_transform_matrix(img_size)
        transform_inv_matrix = self.get_transform_inv_matrix(img_size)

        bindary_transformed = cv2.warpPerspective(bindary_img, transform_matrix, img_size)

        ## find line ##
        self.find_line(bindary_transformed)

        ## Visualization ##
        # road_layer = np.dstack((bindary_transformed, bindary_transformed, bindary_transformed)) * 255
        road_layer = np.zeros_like(img)
        left_line = self.left_lines[-1]
        left_fitx = left_line.current_fitx if left_line.detected else None

        right_line = self.right_lines[-1]
        right_fitx = right_line.current_fitx if right_line.detected else None

        # Color in region
        if left_fitx is not None and right_fitx is not None:
            window_img = visualize_poly(road_layer, left_fitx, right_fitx, ploty)
            road_layer = cv2.addWeighted(road_layer, 1, window_img, 0.3, 0)

        # Plot the polynomial lines onto the image
        if left_fitx is not None:
            cv2.polylines(road_layer, np.int_(np.dstack((left_fitx, ploty))), False, color=(255, 255, 0), thickness=8)
        if right_fitx is not None:
            cv2.polylines(road_layer, np.int_(np.dstack((right_fitx, ploty))), False, color=(255, 255, 0), thickness=8)
        if out_img_show_pixels or (out_img_show_pixels is None and self.out_img_show_pixels):
            road_layer[self.left_lines[-1].ally, self.left_lines[-1].allx] = [0, 0, 255]
            road_layer[self.right_lines[-1].ally, self.right_lines[-1].allx] = [0, 0, 255]

        road_layer = cv2.warpPerspective(road_layer, transform_inv_matrix, img_size)

        out_img = unimg
        if out_img_gray_img or (out_img_gray_img is None and self.out_img_gray_img):
            gray_img = img_to_gray(unimg).astype(np.uint8)
            out_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)

        out_img = cv2.addWeighted(out_img, 1, road_layer, 1, 0)

        ## compute stats ##
        curvature = avg_curvature(self.left_lines[-1].best_curvature, self.right_lines[-1].best_curvature)
        out_img = add_text_to_img(
            out_img,
            'radius of curvature = %s (m)' %
            ('%.2f' % curvature) if curvature is not None and curvature != np.inf else 'inf',
            (10, 100)
        )
        if self.left_lines[-1].line_base_pos is None or self.right_lines[-1].line_base_pos is None:
            out_img = add_text_to_img(
                out_img,
                'vehicle position is unknown',
                (10, 200)
            )
        else:
            dist = (self.left_lines[-1].line_base_pos - self.right_lines[-1].line_base_pos) / 2
            pos = 'left' if dist < 0 else 'right'
            dist = abs(dist)
            out_img = add_text_to_img(
                out_img,
                'vehicle is %.2fm %s of center' % (dist, pos),
                (10, 200)
            )

        return out_img
