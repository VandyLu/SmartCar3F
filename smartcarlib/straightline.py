import cv2
import numpy as np

def straightline(img):
    edges = cv2.Canny(img, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 110)
    lines = np.squeeze(lines, 1)
    i_v = 0
    i_h = 0
    k_v = 0
    b_v = 0
    k_h = 0
    b_h = 0
    for r, theta in lines:
        if (theta < (np.pi / 4)) or (theta > (3 * np.pi / 4)):  # 垂直直线
            k_v = np.pi / 2 - theta + k_v
            b_v = int(r / np.sin(theta)) + b_v
            i_v = i_v + 1
        else:
            k_h = np.pi / 2 - theta + k_h
            b_h = int(r / np.sin(theta)) + b_h
            i_h = i_h + 1
    k_v = k_v / i_v
    b_v = b_v / i_v
    k_h = k_h / i_h
    b_h = b_h / i_h

    return k_v, b_v, k_h, b_h

img = cv2.imread("straightline.jpg", 0)
k_v, b_v, k_h, b_h = straightline(img)
print(k_h)

