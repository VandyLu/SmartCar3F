import cv2
import numpy as np

def crop(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([100, 100, 50])
    upper_blue = np.array([130, 255, 255])

    mask1 = cv2.inRange(hsv, lower_blue, upper_blue)
    output = cv2.bitwise_and(img, img, mask1)
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

        mask1 = mask1[y0:y1, x0:x1]
    return mask1


img = cv2.imread('T1.jpg')
mask = crop(img)
cv2.imwrite('T4.1.jpg',mask)
cv2.imshow('1', mask)
cv2.waitKey()
