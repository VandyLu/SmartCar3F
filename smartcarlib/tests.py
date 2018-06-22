import cv2
from .cv import * 
from .utils import *

import itertools

def test_blackline_detection(params, caps, driver):
    n = 0
    print 'Press C to take a photo, restored in test_images/'
    for i in itertools.count():
        frame = query_camera(caps[0], flip=True)
        line_mask = blackline_detection(frame, threshold=params.threshold)
        target_points, target_image = target_points_detection(line_mask, 3)

        cv2.imshow('frame', frame)
        cv2.imshow('target_image', target_image)
        c = cv2.waitKey(30)

        if c == ord('c'):
            print 'test_{:03d}.jpeg'.format(n)
            cv2.imwrite('images/test_{:03d}.jpeg'.format(n), frame)
            n += 1

        if c == 27 or c == ord('q'):
            break
