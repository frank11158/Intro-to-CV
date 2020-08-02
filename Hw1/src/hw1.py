# -*- coding: utf-8 -*-

import sys
import math
import numpy as np
from hw1_ui import Ui_MainWindow
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
        self.btn1_3.clicked.connect(self.on_btn1_3_click)
        self.btn1_4.clicked.connect(self.on_btn1_4_click)
        self.btn2_1.clicked.connect(self.on_btn2_1_click)
        self.btn3_1.clicked.connect(self.on_btn3_1_click)
        self.btn4_1.clicked.connect(self.on_btn4_1_click)
        self.btn4_2.clicked.connect(self.on_btn4_2_click)
        self.btn5_1.clicked.connect(self.on_btn5_1_click)
        self.btn5_2.clicked.connect(self.on_btn5_2_click)

    def on_btn1_1_click(self):
        img = cv2.imread("../images/dog.bmp")
        print(img)
        print("Height: ", img.shape[0])
        print("Width: ", img.shape[1])
        print("channels: ", img.shape)
        cv2.namedWindow("Image")
        cv2.imshow("Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def on_btn1_2_click(self):
        img = cv2.imread("../images/color.png")
        cv2.namedWindow("Image")
        cv2.imshow("Image", img)
        for row in range(len(img)):
            for col in range(len(img[row])):
                temp = img[row, col][0]
                img[row, col][0] = img[row, col][1]
                img[row, col][1] = img[row, col][2]
                img[row, col][2] = temp
        cv2.namedWindow("Image_converse")
        cv2.imshow("Image_converse", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def on_btn1_3_click(self):
        img = cv2.imread("../images/dog.bmp")
        img_flip = cv2.flip(img, 1)
        cv2.namedWindow("Image")
        cv2.imshow("Image", img)
        cv2.namedWindow("Image_flipping")
        cv2.imshow("Image_flipping", img_flip)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def on_btn1_4_click(self):
        img_origin = cv2.imread("../images/dog.bmp")
        img_flip = cv2.flip(img_origin, 1)
        cv2.namedWindow("Blending")
        cv2.createTrackbar('Blend', 'Blending', 0, 100, nothing)
        while(1):
            Blend_Val = cv2.getTrackbarPos('Blend', 'Blending') / 100
            overlapping = cv2.addWeighted(img_origin, Blend_Val, img_flip, 1-Blend_Val, 0)
            cv2.imshow("Blending", overlapping)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

    def on_btn2_1_click(self):
        img = cv2.imread("../images/M8.jpg")
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray_img, (3, 3), 0) # 3x3 Gaussian Smooth Filter
        cv2.namedWindow("GausBlur")
        cv2.imshow("GausBlur", blur)

        cv2.namedWindow("Sobel_M")
        cv2.namedWindow("Sobel_A")
        cv2.createTrackbar('Magnitude', 'Sobel_M', 0, 255, nothing)
        cv2.createTrackbar('Angle', 'Sobel_A', 0, 360, nothing)
        while(1):
            # magnitude
            threshold = cv2.getTrackbarPos('Magnitude', 'Sobel_M')
            sobel_M = sobel_filter(blur, threshold, -1)
            cv2.imshow("Sobel_M", sobel_M)
            # angel
            angel = cv2.getTrackbarPos('Angle', 'Sobel_A')
            sobel_A = sobel_filter(blur, 80, angel)
            cv2.imshow("Sobel_A", sobel_A)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def on_btn3_1_click(self):
        img = cv2.imread("../images/pyramids_Gray.jpg")
        # Gaussain pyramid - Down sampling
        gaussian_L1 = cv2.pyrDown(img) # 5x5 Gaussian filter?
        gaussian_L2 = cv2.pyrDown(gaussian_L1)
        # Up sampling
        gaussian_expanded_L0 = cv2.pyrUp(gaussian_L1)
        gaussian_expanded_L1 = cv2.pyrUp(gaussian_L2)
        # Laplacian pyramid - Subraction
        laplacian_L0 = cv2.subtract(img, gaussian_expanded_L0)
        laplacian_L1 = cv2.subtract(gaussian_L1, gaussian_expanded_L1)
        # Inverse pyramid
        inverse_L1 = cv2.add(gaussian_expanded_L1, laplacian_L1)
        inverse_L0 = cv2.add(gaussian_expanded_L0, laplacian_L0)
        # cv2.imshow("Image", img)
        cv2.imshow("Gaussian", gaussian_L1)
        cv2.imshow("Laplacian", laplacian_L0)
        cv2.imshow("Inverse_L1", inverse_L1)
        cv2.imshow("Inverse_L0", inverse_L0)

    def on_btn4_1_click(self):
        img = cv2.imread("../images/QR.png", 0)
        ret,thresh_global = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)
        cv2.imshow("Global Threshold", thresh_global)

    def on_btn4_2_click(self):
        img = cv2.imread("../images/QR.png", 0)
        thresh_adaptive = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,\
        cv2.THRESH_BINARY, 19, -1)
        cv2.imshow("Local Threshold", thresh_adaptive)

    def on_btn5_1_click(self):
        centerX = 130
        centerY = 125
        Angel = float(self.edtAngle.text())
        Scale = float(self.edtScale.text())
        Tx = float(self.edtTx.text())
        Ty = float(self.edtTy.text())

        img = cv2.imread("../images/OriginalTransform.png")
        col,row = img.shape[:2]
        movMat = np.float32([[1, 0, Tx], [0, 1, Ty]])
        rotateMat = cv2.getRotationMatrix2D((centerX, centerY), Angel, Scale)
        result = cv2.warpAffine(cv2.warpAffine(img, rotateMat, (row, col)), movMat, (row, col))
        cv2.imshow("Image", img)
        cv2.imshow("OriginalTransform", result)
        

    def on_btn5_2_click(self):
        img = cv2.imread("../images/OriginalPerspective.png")
        cv2.namedWindow("Image")
        cv2.imshow("Image", img)
        cv2.setMouseCallback('Image', appendPoint)
        pts = np.float32([[0, 0], [0, 0], [0, 0], [0, 0]]) 
        while(1): # read 4 points
            global clickNum
            if clickNum == 4:
                for i in range (0,4):
                    pts[i] = pts1[i]
                    print(pts[i])
                break
            if cv2.waitKey(20) & 0xFF == 27:
                break
        pts2 = np.float32([[0, 0], [450, 0], [450, 450], [0, 450]])
        transMat = cv2.getPerspectiveTransform(pts, pts2)
        result = cv2.warpPerspective(img, transMat, (450, 450))
        cv2.imshow("Perspective", result)

    ### ### ###
    
def nothing(x):
    pass

def sobel_filter(img, thres, angel):
    row = img.shape[0] #row 
    col = img.shape[1] #column
    result = np.zeros((row,col))
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype = np.float)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype = np.float)
    # img padding with 0
    img = np.pad(img, (1,1), 'edge')

    for i in range(1, row-1):
        for j in range(1, col-1):
            G_x = (sobel_x[0][0] * img[i-1][j-1]) + (sobel_x[0][1] * img[i-1][j]) + \
                (sobel_x[0][2] * img[i-1][j+1]) + (sobel_x[1][0] * img[i][j-1]) + \
                (sobel_x[1][1] * img[i][j]) + (sobel_x[1][2] * img[i][j+1]) + \
                (sobel_x[2][0] * img[i+1][j-1]) + (sobel_x[2][1] * img[i+1][j]) + \
                (sobel_x[2][2] * img[i+1][j+1])
            G_y = (sobel_y[0][0] * img[i-1][j-1]) + (sobel_y[0][1] * img[i-1][j]) + \
                (sobel_y[0][2] * img[i-1][j+1]) + (sobel_y[1][0] * img[i][j-1]) + \
                (sobel_y[1][1] * img[i][j]) + (sobel_y[1][2] * img[i][j+1]) + \
                (sobel_y[2][0] * img[i+1][j-1]) + (sobel_y[2][1] * img[i+1][j]) + \
                (sobel_y[2][2] * img[i+1][j+1])
            G_mag = np.sqrt(G_x*G_x + G_y*G_y)
            if angel != -1:
                theta = np.arctan2(G_x, G_y)*180/math.pi + 180 # 0 < theta < 360
                if abs(theta - angel) > 10: # set to 0 if angel diff > 10
                    G_mag = 0
            if G_mag < thres:
                G_mag = 0
            result[i-1,j-1] = G_mag
    result *= 255.0 / np.max(result)
    return result


# def sobel_filter(src, threshold, angel):
#     sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
#     sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
#     G_x = np.absolute(cv2.filter2D(src, cv2.CV_64F, sobel_x))
#     G_y = np.absolute(cv2.filter2D(src, cv2.CV_64F, sobel_y))
#     G_mag = np.sqrt(G_x*G_x + G_y*G_y)
#     G_mag *= 255.0 / np.max(G_mag) # Normalize
#     if angel != -1:
#         theta = np.arctan2(G_x, G_y)*180/math.pi # -180 < theta < 180
#         theta += 180
#         for i in range(len(theta)):
#             for j in range(len(theta[i])):
#                 if abs(theta[i][j] - angel) > 10: # set to 0 when angel diff > 10
#                     G_mag[i][j] = 0 
#     ret,result = cv2.threshold(G_mag, threshold, 255, cv2.THRESH_TOZERO)
#     return np.uint8(result)

pts1 = np.zeros(shape=(4, 2))
clickNum = 0
def appendPoint(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global clickNum
        if clickNum <= 3:
            pts1[clickNum] = [int(x), int(y)]
        print(pts1)
        clickNum += 1

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
