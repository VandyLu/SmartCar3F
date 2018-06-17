import cv2
import numpy as np

def detect_parksign(image, kernel_size=(5,5), r_threshold = 100):   # r_threshold 半径大于某值认为与停车标志距离足够近
    ''' FXX
        
        return True if there is park sign in the image 
    '''

    # detect blue and crop the image

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 100, 50])
    upper_blue = np.array([130, 255, 255])

    mask1 = cv2.inRange(hsv, lower_blue, upper_blue)
    output = cv2.bitwise_and(image, image, mask1)
    out_binary, contours, hierarchy = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        rect = cv2.minAreaRect(contours[i])
        box = np.int0(cv2.boxPoints(rect))
        cv2.drawContours(mask1, [box], 0, (100, 0, 0), 5)

        x0 = int(rect[0][0] - rect[1][0] / 2)
        y0 = int(rect[0][1] - rect[1][1] / 2)
        x1 = int(rect[0][0] + rect[1][0] / 2)
        y1 = int(rect[0][1] + rect[1][1] / 2)

        mask1 = image[y0:y1, x0:x1]      # cropped image
    # circle detection

    gray = cv2.cvtColor(mask1, cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(gray, kernel_size)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 200, param1=100, param2=30, minRadius=50, maxRadius=300)
    for circle in circles[0]:
        x = int(circle[0])
        y = int(circle[1])
        r = int(circle[2])
        image = cv2.circle(image, (x, y), r, (0, 0, 255), 3)
    if r > r_threshold:
        return True
    else:
        return False


# test

image = cv2.imread('T1.jpg')
x = detect_parksign(image, (5,5), 100)
print(x)
