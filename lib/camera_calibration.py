import numpy as np
import cv2
import itertools


def _generate_objpoints_imgpoints(img, nx, ny, thresh=0.5):
    gray_tmp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.zeros_like(gray_tmp)
    gray[gray_tmp > thresh * np.max(gray_tmp)] = np.max(gray_tmp)
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    objpoints = np.array([[i % nx, i // nx, 0] for i in range(nx * ny)]).astype('float32')
    imgpoints = np.array(corners)
    return ret, objpoints, imgpoints


def generate_objpoints_imgpoints(img, nxs=None, nys=None):
    nxs = nxs if nxs else [9]
    nys = nys if nys else [6, 5]
    ret, objpoints, imgpoints = False, [], []
    nx, ny = 0, 0
    for nx, ny, thresh in itertools.product(nxs, nys, [0.3, .5, .7]):
        ret, objpoints, imgpoints = _generate_objpoints_imgpoints(img, nx, ny, thresh)
        if ret:
            break
    return ret, objpoints, imgpoints, nx, ny


def undistort_matrix(img, objpoints_list, imgpoints_list):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints_list, imgpoints_list, img.shape[1::-1], None, None
    )
    return ret, mtx, dist, rvecs, tvecs


def generate_undistort_matrix(image_files, wait_key=-1):
    """
    generate undistord matrix (ret, mtx, dist, rvecs, tvecs) for the input images
    :param image_files: directory of images
    :param log: bool
    :return: ret, mtx, dist, rvecs, tvecs
    """
    # Make a list of calibration images
    nxs = [9]
    nys = [6, 5]

    objpoints_list = []  # 3d points in real world space
    imgpoints_list = []  # 2d points in image plane.

    img = None
    for idx, fname in enumerate(image_files):
        if wait_key >= 0:
            print(fname)
        img = cv2.imread(fname)
        ret, objpoints, imgpoints, nx, ny = generate_objpoints_imgpoints(img, nxs, nys)
        if ret:
            objpoints_list.append(objpoints)
            imgpoints_list.append(imgpoints)
            if wait_key >= 0:
                img_cp = np.copy(img)
                cv2.drawChessboardCorners(img_cp, (nx, ny), imgpoints, ret)
                cv2.imshow(fname, img_cp)
                cv2.waitKey(wait_key)
        elif wait_key >= 0:
            print(ret)
    if wait_key >= 0:
        cv2.destroyAllWindows()
        cv2.waitKey(100)

    return undistort_matrix(img, objpoints_list, imgpoints_list)


def undistort_img(img, mtx, dist):
    return cv2.undistort(img, mtx, dist, None, mtx)
