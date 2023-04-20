import cv2
import numpy as np

# img = frame
# ROI , PERSPECTIVE VIEW AND WARP

# SPLIT LEFT & RIGHT FRAME
def set_img(img,num):
    
    kernel = np.ones((num,num),np.uint8)

    # SET LEFT FRAME(WHITE)
    mask_L = np.zeros_like(img)
        
    MORPH_L = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    DILATE = cv2.dilate(MORPH_L, kernel, iterations=2)

    vertices_L = np.array([[(0,0),(200,0),(200,200),(0,200)]], dtype=np.int32)
    cv2.fillPoly(mask_L,vertices_L,(255,255,255))
    HALF_L = cv2.bitwise_and(DILATE,mask_L)

    HSV_L = cv2.cvtColor(HALF_L,cv2.COLOR_BGR2HSV)

    lower_white = np.array([0,0,200], dtype=np.uint8)
    upper_white = np.array([180,50,255], dtype=np.uint8)
    WHITE = cv2.inRange(HSV_L, lower_white, upper_white)    
    
    # SET RIGHT FRAME(BLACK)
    mask_R = np.zeros_like(img)

    MORPH_R = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    ERODE = cv2.erode(MORPH_R, kernel, iterations=2)

    vertices_R = np.array([[(200,0),(320,0),(320,200),(200,200)]], dtype=np.int32)
    cv2.fillPoly(mask_R,vertices_R,(255,255,255))
    HALF_R = cv2.bitwise_and(ERODE,mask_R)
    
    HSV_R = cv2.cvtColor(HALF_R,cv2.COLOR_BGR2HSV)

    lower_black = np.array([0,0,0], dtype=np.uint8)
    upper_black = np.array([180,255,100], dtype=np.uint8)
    BLACK = cv2.inRange(HSV_R, lower_black, upper_black)

    mix_L = cv2.bitwise_and(HSV_L,HSV_L,mask=WHITE)
    mix_R = cv2.bitwise_and(HSV_R,HSV_R,mask=BLACK)

    BGR_L = cv2.cvtColor(mix_L, cv2.COLOR_HSV2BGR)
    BGR_R = cv2.cvtColor(mix_R, cv2.COLOR_HLS2BGR)

    mix = cv2.add(BGR_L,BGR_R)

    return mix

def set_rect(img, value1, value2):

    height,width = img.shape[:2]
    
    rect = np.zeros((4,2),dtype=np.float32)

    rect[0] = (value1, height*2/3-value1)           # TOP LEFT
    rect[1] = (width-value1, height*2/3-value1)     # TOP RIGHT

    rect[2] = (width-value2,height-value1)          # BOTTOM RIGHT
    rect[3] = (value2,height-value1)                # BOTTOM LEFT

    return rect
    
def set_roi(img, vertices, color3 = (255,255,255), color1 = 255):

    mask = np.zeros_like(img)

    if len(img.shape) > 2:
        color = color3
    else:
        color = color1
    
    cv2.fillPoly(mask, vertices, color)
    ROI_img = cv2.bitwise_and(img, mask)

    return ROI_img

# PERSPECTIVE VIEW & WARP    
def warp_img(img,rect):

    # CALCULATE MAX,MIN OF WIDTH AND HEIGHT
    widthA = np.sqrt((pow((rect[0][0] - rect[1][0]),2)) + (pow((rect[0][1] - rect[1][1]),2)))
    widthB = np.sqrt((pow((rect[2][0] - rect[3][0]),2)) + (pow((rect[2][1] - rect[3][1]),2)))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt((pow((rect[0][0] - rect[3][0]),2)) + (pow((rect[0][1] - rect[3][1]),2)))
    heightB = np.sqrt((pow((rect[1][0] - rect[2][0]),2)) + (pow((rect[1][1] - rect[2][1]),2)))
    maxHeight = max(int(heightA), int(heightB))
    
    distance = np.array([[0,0],[maxWidth - 1, 0],[maxWidth - 1, maxHeight - 1],
                         [0, maxHeight - 1]], dtype = 'float32')
    
    M = cv2.getPerspectiveTransform(rect, distance)
    warped = cv2.warpPerspective(img,M,(maxWidth,maxHeight))
    warped = cv2.resize(warped,(320,200),interpolation=cv2.INTER_AREA) 

    return warped

# FILTERING IMG
def filter_img(img,kernel,low_threshold,high_threshold):

    # img = NOISE
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (kernel,kernel),0)
    blur2 = cv2.bilateralFilter(gray,kernel,50,50)
    canny = cv2.Canny(blur2,low_threshold,high_threshold)

    return canny

# HOUGH TRANSFORM AND DRAW LINES
def hough_img(img,height,width,threshold,min_length,max_gap):

    # SET LINE IMG
    lines = cv2.HoughLinesP(img.astype(np.uint8),1,np.pi/180,threshold,min_length,max_gap)
    line_img = np.zeros((img.shape[0],img.shape[1],3), dtype=np.float32)
    yline_img = np.zeros((img.shape[0],img.shape[1],3), dtype=np.float32)

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    xedge_rmv = 50
    yedge_rmv = 20

    #(xedge_rmv < x1 < width-xedge_rmv) & 
    
    for line in lines:
            for x1,y1,x2,y2 in line:                             
                if 100 - yedge_rmv < y1 < 100 + yedge_rmv :
                    if 10 < x1 < 160 - xedge_rmv or 160 + xedge_rmv < x1 < 310:
                        cv2.line(line_img,(x1,y1),(x2,y2),(0,0,255),2)

    return line_img

# DRAW CENTER LINES
def hough_yimg(img,threshold,min_length,max_gap):

    # SET LINE IMG
    lines = cv2.HoughLinesP(img.astype(np.uint8),1,np.pi/180,threshold,min_length,max_gap)
    line_img = np.zeros((img.shape[0],img.shape[1],3), dtype=np.float32)
    yline_img = np.zeros((img.shape[0],img.shape[1],3), dtype=np.float32)
     
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    xedge_rmv = 50
    yedge_rmv = 50
    
    X1,Y1,X2,Y2 = [0,0,0,0]
    data = (0,0)
    data2 = (0,0)
    
    for line in lines:
            for x1,y1,x2,y2 in line:                             
                if 100 - yedge_rmv < y1 < 100 + yedge_rmv:
                    #if 8 < x1 < 160 - xedge_rmv or 160 + xedge_rmv < x1 < 312:
                    if 10 < x1 < 150:
                        if 70*np.pi/180 < np.absolute(np.arctan((np.absolute(y1-y2)/np.absolute(x1-x2))*180/np.pi)) < 110*np.pi/180:            
                            X1 = x1
                            Y1 = y1
                            if x2 < 150:
                                data = (50,0)
                                data = np.uint8(data)
                                #print('turn right')        
                    if 150 < x1 < 310:
                        if 70*np.pi/180 < np.absolute(np.arctan((np.absolute(y1-y2)/np.absolute(x1-x2))*180/np.pi)) < 110*np.pi/180:                   
                            X2 = x2
                            Y2 = y2
                            if 170 < x2:
                                data = (290,0)                    
                                #print('turn left') 

                    if 10 < X1 & X2 < 310:                          
                        if 180 < np.absolute(X1-X2) < 220: 
                            data = (int((X1+X2)/2),int(np.absolute(Y1+Y2)/2))
                            data = np.uint8(data)
                          
                            cv2.line(yline_img,(X1,int(np.absolute(Y1+Y2)/2)),(X2,int(np.absolute(Y1+Y2)/2)),(255,0,255),3)
                            cv2.line(yline_img,(int(np.absolute(X1+X2)/2),int(np.absolute(Y1+Y2)/2)-5),(int(np.absolute(X1+X2)/2),int(np.absolute(Y1+Y2)/2)+5),(0,255,255),2)
                            #print('keep center')

                    if  20 < np.absolute(x2-X1) & 140 < x2 < 200:
                        if 165*np.pi/180 < np.absolute(np.arctan((np.absolute(y1-y2)/np.absolute(x1-x2))*180/np.pi)) < 195*np.pi/180:
                            data = (0,0)
                            #print(x1,x2)   

    return yline_img, data

# RESUTL EDGE
def result_edge(line_img, original_img, weightness):

    return cv2.addWeighted(line_img,1,original_img,weightness,0,dtype=cv2.CV_8UC1)

# RESULT IMG
def result_yline(yline_img, original_img, weightness):
    #added = cv2.bitwise_and(line_img2,line_img) 

    return cv2.addWeighted(yline_img,1,original_img,weightness,0,dtype=cv2.CV_8UC1)
