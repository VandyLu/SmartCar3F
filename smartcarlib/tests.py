import cv2
from .cv import * 
from .utils import *

import itertools

def test_blackline_detection(params, caps, driver):
    n = 0
    
    image = cv2.imread('4.png', -1)
    image = crop(image, 0.5, 0.4)

    print(image.shape)
    line_mask = blackline_detection(image, threshold=params.threshold)
    targets, targets_image = target_points_detection(line_mask, 1)

    cv2.imshow('frame', image)
    cv2.imshow('line', line_mask)
    cv2.imshow('target_image', targets_image)
    c = cv2.waitKey()

    return 


    print('Press C to take a photo, restored in test_images/')
    for i in itertools.count():
        frame = query_camera(caps[0], flip=True)
        line_mask = blackline_detection(frame, threshold=params.threshold)
        target_points, target_image = target_points_detection(line_mask, 1)

        cv2.imshow('frame', frame)
        cv2.imshow('target_image', target_image)
        c = cv2.waitKey(30)

        if c == ord('c'):
            print('test_{:03d}.jpeg'.format(n))
            cv2.imwrite('images/test_{:03d}.jpeg'.format(n), frame)
            n += 1

        if c == 27 or c == ord('q'):
            break
