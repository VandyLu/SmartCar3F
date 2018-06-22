# This computer vision part of smartcar library
# Author: LFB

from __future__ import absolute_import, division, print_function
import cv2
import numpy as np
from skimage import measure

def blackline_detection(image, threshold=64, erode_iters=2):
    ''' Detect black line, do GaussianBlur or Closed operation to avoid errors
    '''
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 1.5)
    _, binary = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY_INV)

    erode = cv2.erode(binary, np.ones((5,5), np.uint8), iterations=erode_iters)

    mask = erode
    return mask

                  
def calculate_center(image):
    ''' Calculate center of a binary image
    '''
    height, width = image.shape
    valid_mask = image > 0.5
    xrange = np.arange(width).reshape((1, width))
    yrange = np.arange(height).reshape((height, 1))
    xmap = np.tile(xrange, (height, 1))
    ymap = np.tile(yrange, (1, width))

    valid_x = xmap[valid_mask]
    valid_y = ymap[valid_mask]

    if np.sum(valid_x) == 0 or np.sum(valid_y) == 0:
        return width/2, height

    mean_x = np.mean(valid_x)
    mean_y = np.mean(valid_y)
    return mean_x, mean_y

def target_points_detection(line_mask, num=1):
    ''' Detect targets given mask of blackline, at most 11 targets
    '''
    assert num > 0 and num < 12
    height, width = line_mask.shape
    hstep = height // num

    tmp = line_mask.copy()
    tars = np.zeros((num, 2), dtype=np.float32)
    for i in range(num):
        rect_mask = line_mask[height-(i+1)*hstep: height-i*hstep, :]
        mean_x, mean_y = calculate_center(rect_mask)
        mean_y = height - (i+1)*hstep + mean_y

        tars[i, 0] = mean_x
        tars[i, 1] = mean_y

        tmp[int(mean_y-3): int(mean_y+3), int(mean_x-3): int(mean_x+3)] = 128

    #cv2.imshow('tmp', tmp)
    #cv2.waitKey()
    return tars, tmp



if __name__ == '__main__':
    image = cv2.imread('./cross.jpeg', -1)
    mask = blackline_detection(image, 100.0)

    for t in range(10):
        target_points_detection(mask, t+1)

    cv2.imshow('img', image)
    cv2.imshow('mask', mask)
    cv2.waitKey()
