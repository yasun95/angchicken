import cv2
import numpy as np
import smbus as sm
#import RPi.GPIO as gp
from utils.setFrame import set_img, set_rect, set_roi, warp_img, filter_img, hough_img, hough_yimg, result_yline, result_edge
from utils.calibration import undistortion
from utils.connecter import writeNumber, readNumber, sendData
#from multiCam import setGPIO

# SET I2C
bus = sm.SMBus(1)
address = 0x04

# CAM MATRIX & DISTORTION COEFFICIENT
mtx = np.float32([[150.34752656, 0, 159.20976247],
       [0, 149.56439711, 95.08680565], [0, 0, 1]])
dist = np.float32([[-0.29374967, 0.0809826, -0.00233821, -0.00363307, -0.01106136]])

# SET CAM PROP
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 200)

if __name__ == '__main__':
    while cap.isOpened():
        _, frame = cap.read()
        if _ :
            undist_img = undistortion(frame, mtx, dist)
            height, width = undist_img.shape[:2]

            halfControl_img = set_img(undist_img, 3)                     

            rect = set_rect(undist_img, 0, 0)
            vertices = np.array([[rect[0],rect[1], rect[2], rect[3]]], dtype=np.int32)

            # set_roi(img,value1,value2=0 or value2=value1)
            ROI = set_roi(halfControl_img,vertices)                  
                               
            # psv = warp_img(canny,rect)
            psv_BGR = warp_img(ROI,rect)
            
            # filter_img(img,kernel,low_threshold,high_threshold)
            psv_canny = filter_img(psv_BGR,3,20,50)                  
            
            try:
                # hough_img(img,height,width,threshold,min_length,max_gap)
                # hough = hough_img(psv_canny,height,width,20,10,10)  
                houghy, data = hough_yimg(psv_canny,20,10,10)       

                pixel_data = sendData(data)
            
                # result_img(img, ximg, weightness)
                # res_edge = result_edge(hough,psv_BGR,0.8)         
                res_yimg = result_yline(houghy,psv_BGR,1)
                   
                # cv2.imshow('result edge detective', res_edge)
                # cv2.imshow('result', undist_img)
                cv2.imshow('result yline detective', res_yimg)
            except:
                pass

            if cv2.waitKey(1) == ord('q'):
                break
        else:
            print("NO FRAME")

    cap.release()
    cv2.destroyAllWindows()
