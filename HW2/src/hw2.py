# -*- coding: utf-8 -*-

import sys
import math
import glob
import numpy as np
import matplotlib.pyplot as plt
from hw2_ui import Ui_MainWindow
import cv2
from PyQt5.QtWidgets import QMainWindow, QApplication


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.onBindingUI()

    def onBindingUI(self):
        self.btn1_1.clicked.connect(self.on_btn1_1_click)
        self.btn1_2.clicked.connect(self.on_btn1_2_click)
        self.btn2_1.clicked.connect(self.on_btn2_1_click)
        self.btn2_2.clicked.connect(self.on_btn2_2_click)
        self.btn2_3.clicked.connect(self.on_btn2_3_click)
        self.btn3_1.clicked.connect(self.on_btn3_1_click)
        self.btn3_2.clicked.connect(self.on_btn3_2_click)
        self.btn3_3.clicked.connect(self.on_btn3_3_click)
        self.btn3_4.clicked.connect(self.on_btn3_4_click)
        self.btn4_1.clicked.connect(self.on_btn4_1_click)

    def on_btn1_1_click(self):
        img = cv2.imread("../images/plant.jpg", 0)
        cv2.namedWindow("Image")
        cv2.imshow("Image", img)
        plt.hist(img.flatten(), 256, [0,256], color = 'r')
        plt.show()
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def on_btn1_2_click(self):
        img = cv2.imread("../images/plant.jpg", 0)
        equ = cv2.equalizeHist(img)
        cv2.namedWindow("Image")
        cv2.imshow("Image", equ)
        plt.hist(equ.flatten(), 256, [0,256], color = 'r')
        plt.show()
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def on_btn2_1_click(self):
        img = cv2.imread("../images/q2_train.jpg")
        cv2.namedWindow("Image")
        cv2.imshow("Image", img)
        img_g = cv2.imread("../images/q2_train.jpg", 0)
        img = cv2.medianBlur(img_g, 5)
        cimg = cv2.cvtColor(img_g, cv2.COLOR_GRAY2BGR)
        circles = cv2.HoughCircles(img_g, cv2.HOUGH_GRADIENT, 1, 30, param1=50, param2=17, minRadius=14, maxRadius=20)
        circles = np.uint16(np.around(circles))
        print(circles)
        # draw the circles
        for i in circles[0,:]:
            cv2.circle(cimg, (i[0],i[1]), i[2],(0,255,0), 2)
            cv2.circle(cimg, (i[0],i[1]), 2, (0,0,255), 3)
        cv2.imshow('circles', cimg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def on_btn2_2_click(self):
        img = cv2.imread("../images/q2_train.jpg")
        cv2.namedWindow("Image")
        cv2.imshow("Image", img)
        img_g = cv2.imread("../images/q2_train.jpg", 0)
        img = cv2.medianBlur(img_g, 5)
        cimg = cv2.cvtColor(img_g, cv2.COLOR_GRAY2BGR)
        circles = cv2.HoughCircles(img_g, cv2.HOUGH_GRADIENT, 1, 30, param1=50, param2=17, minRadius=14, maxRadius=20)
        circles = np.uint16(np.around(circles))

        from math import hypot
        mask = np.zeros(img.shape[:2], np.uint8)
        rows, cols = img.shape[0:2]
        for x, y, r in circles[0,:]:
            for i in range(cols):
                for j in range(rows):
                    if hypot(i-x, j-y) < r:
                        mask[j, i] = 255
        cv2.imshow("mask", mask)

        hist = cv2.calcHist([img], [0], mask, [256], [0,256])
        plt.plot(hist)
        plt.show()
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def on_btn2_3_click(self):
        img = cv2.imread("../images/q2_train.jpg")
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hue = np.zeros(hsv.shape, dtype=np.uint8)
        cv2.mixChannels([hsv], [hue], [0,0])
        hist = cv2.calcHist([img], [0], None, [256], [0,256])
        # hist = cv2.calcHist([hue], [0], None, [256], [0,256])
        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX, -1)
        backproj = cv2.calcBackProject([hue], [0,1], hist, [0,256], 1)
        cv2.imshow("backproj", backproj)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def on_btn3_1_click(self):
        comeraCalibrate("corner_detection", "all")
        
    def on_btn3_2_click(self):
        comeraCalibrate("intrinsic", "all")
        
    def on_btn3_3_click(self):
        selePic = str(self.comboBox.currentText())
        comeraCalibrate("extrinsic", selePic)  

    def on_btn3_4_click(self):
        comeraCalibrate("distortion", "all")

    def on_btn4_1_click(self):
        cube = np.float32([[2, 2, -2], [2, 0, -2], [0, 0, -2], [0, 2, -2], [2, 2, 0],[2, 0, 0],[0, 0, 0],[0, 2, 0]])    
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        w,h = 11,8
        objp = np.zeros((w*h,3), np.float32)
        objp[:,:2] = np.mgrid[0:w,0:h].T.reshape(-1,2)
        objpoints = []
        imgpoints = []
        images = glob.glob("../images/CameraCalibration/*.bmp")
        index = 1
        for index in range(5):
            img = cv2.imread("../images/CameraCalibration/%s.bmp" % str(index+1))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (w,h), None)
            if ret == True:
                corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
                objpoints.append(objp)
                imgpoints.append(corners)
                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
                # Find the rotation and translation vectors.
                _, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)
                # project 3D points to image plane
                imgpts, jac = cv2.projectPoints(cube, rvecs, tvecs, mtx, dist)
                img = drawCube(img, corners2, imgpts)
                cv2.namedWindow("Image", 0)
                cv2.resizeWindow("Image", 500, 500)
                cv2.imshow("Image", img)
                cv2.waitKey(500)
        cv2.destroyAllWindows()

def comeraCalibrate(method, selection):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    w,h = 11,8
    objp = np.zeros((w*h,3), np.float32)
    objp[:,:2] = np.mgrid[0:w,0:h].T.reshape(-1,2)
    objpoints = []
    imgpoints = []
    if selection == "all":
        images = glob.glob("../images/CameraCalibration/*.bmp")
    else:
        images = glob.glob("../images/CameraCalibration/%s.bmp" % selection)
    index = 1
    for image in images:
        img = cv2.imread(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (w,h), None)
        if ret == True:
            cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners)
            if method == "corner_detection":
                cv2.drawChessboardCorners(img, (w,h), corners, ret)
                cv2.namedWindow("Image" + str(index), 0)
                cv2.resizeWindow("Image" + str(index), 500, 500)
                cv2.imshow("Image" + str(index), img)
            elif method == "intrinsic":
                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
                print("Picture(%d): " % index)
                print(mtx)
            elif method == "extrinsic":
                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
                print("Picture(%s): " % selection)
                rotMat = cv2.Rodrigues(np.asarray(rvecs))
                extrinsicMat = np.concatenate((rotMat[0], tvecs[0]),axis=1)
                print(extrinsicMat)
            elif method == "distortion":
                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
                print("Picture(%d): " % index)
                print(dist)
            else:
                print("invalid parameter")
        index += 1
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def drawCube(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1, 2)
    img = cv2.drawContours(img, [imgpts[:4]], -1, (0,0,255), -3)
    img = cv2.drawContours(img, [imgpts[4:]], -1, (0,0,255), 3)
    for i,j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (0,0,255), 3)
    return img

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
