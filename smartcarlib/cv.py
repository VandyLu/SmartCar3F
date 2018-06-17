# This computer vision part of smartcar library
# Author: LFB

from __future__ import absolute_import, division, print_function
import cv2
import numpy as np
from skimage import measure

def blackline_detection(image):
    ''' Detect black line, do GaussianBlur or Closed operation to avoid errors
    '''
    height = image.shape(0)
    width = image.shape(1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 110)
    lines = np.squeeze(lines, 1)
    i_v = 0
    i_h = 0
    k_v = 0  # k of the vertical line
    b_v = 0  # b of the vertical line
    k_h = 0  # k of the horizontal line
    b_h = 0  # b of the horizontal line
    for r, theta in lines:
        if (theta < (np.pi / 4)) or (theta > (3 * np.pi / 4)):  # vertical line
            k_v = np.pi / 2 - theta + k_v
            b_v = int(r / np.sin(theta)) + b_v
            i_v = i_v + 1
        else:  # horizontal line
            k_h = np.pi / 2 - theta + k_h
            b_h = int(r / np.sin(theta)) + b_h
            i_h = i_h + 1
    k_v = np.tan(k_v / i_v)  # average
    b_v = b_v / i_v
    k_h = np.tan(k_h / i_h)
    b_h = b_h / i_h
    pt1 = (int(b_v / k_v), 0)
    pt2 = (int((b_v + height) / k_v), height)
    mask = np.zeros((height, width, 3), dtype=np.uint8)
    mask = cv2.line(mask, pt1, pt2, (255, 255, 255))
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

    cv2.imshow('tmp', tmp)
    cv2.waitKey()
    return tars, tmp



if __name__ == '__main__':
    image = cv2.imread('./cross.jpeg', -1)
    mask = blackline_detection(image, 100.0)

    for t in range(10):
        target_points_detection(mask, t+1)

    cv2.imshow('img', image)
    cv2.imshow('mask', mask)
    cv2.waitKey()
