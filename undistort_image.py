import numpy as np
import cv2 as cv

data = np.load("calibration_data.npz")

mtx = data["mtx"]
dist = data["dist"]

img = cv.imread("images/chess0.jpg")

h, w = img.shape[:2]

newcameramtx, roi = cv.getOptimalNewCameraMatrix(
    mtx,
    dist,
    (w,h),
    1,
    (w,h)
)

dst = cv.undistort(img, mtx, dist, None, newcameramtx)

x, y, w, h = roi
dst = dst[y:y+h, x:x+w]

cv.imshow("Undistorted Image", dst)
cv.waitKey(0)
cv.destroyAllWindows()

cv.imwrite("calibresult.png", dst)