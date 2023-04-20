import cv2
import numpy as np
import glob

def calib():
    
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 22, 0.001)

    cell_X=10
    cell_Y=7

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((cell_X*cell_Y,3), np.float32)
    objp[:,:2] = np.mgrid[0:cell_X,0:cell_Y].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    images = glob.glob('/home/pi/PROJECT/pictures-00/cam*.jpg')

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (cell_X,cell_Y),None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (cell_X,cell_Y), corners2,ret)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return mtx, dist

def undistortion(img, mtx, dist):

    return cv2.undistort(img, mtx, dist, None, mtx)
