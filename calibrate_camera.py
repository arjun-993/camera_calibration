import numpy as np
import cv2 as cv
import glob

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

objpoints = []
imgpoints = []
image_shape = None

images = glob.glob('images/*.jpg')

for fname in images:

    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    if image_shape is None:
        image_shape = gray.shape[::-1]

    ret, corners = cv.findChessboardCorners(gray, (7,6), None)

    if ret == True:

        objpoints.append(objp)

        corners2 = cv.cornerSubPix(
            gray, corners, (11,11), (-1,-1), criteria
        )

        imgpoints.append(corners2)

        cv.drawChessboardCorners(img, (7,6), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(500)

cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
    objpoints,
    imgpoints,
    image_shape,
    None,
    None
)

np.savez(
    "calibration_data.npz",
    mtx=mtx,
    dist=dist,
    rvecs=rvecs,
    tvecs=tvecs
)

print("Calibration data saved.")