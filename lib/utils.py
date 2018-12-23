import numpy as np
import cv2


def add_text_to_img(img: np.ndarray, text, bottomLeftCornerOfText=None):
    # Create a black image
    out_img = img.copy()

    # Write some Text
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = bottomLeftCornerOfText if bottomLeftCornerOfText else (10,100)
    fontScale              = 2
    thickness              = 4
    fontColor              = (255,255,255)
    lineType               = 2

    cv2.putText(out_img, text, bottomLeftCornerOfText, font,
                fontScale, fontColor, thickness=thickness, lineType=lineType)
    return out_img


def get_perspective_transform_meta_data(img_size_x, img_size_y):
    src = np.float32(
        [[(img_size_x / 2) - 63, img_size_y / 2 + 100],
         [((img_size_x / 6) - 12), img_size_y],
         [(img_size_x * 5 / 6) + 90, img_size_y],
         [(img_size_x / 2 + 70), img_size_y / 2 + 100]])
    dst = np.float32(
        [[(img_size_x / 4), 0],
         [(img_size_x / 4), img_size_y],
         [(img_size_x * 3 / 4), img_size_y],
         [(img_size_x * 3 / 4), 0]])

    # Warp the image using OpenCV warpPerspective()
    transform_matrix = cv2.getPerspectiveTransform(src, dst)
    transform_inv_matrix = cv2.getPerspectiveTransform(dst, src)
    return src, dst, transform_matrix, transform_inv_matrix


def combine_img(img1, img2, alpha1, alpha2):
    return cv2.addWeighted(img1, alpha1, img2, alpha2, 0)