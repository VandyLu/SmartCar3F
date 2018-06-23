import cv2
from .cv import * 
from .utils import *
from .parksign import *

import itertools

def test_blackline_detection(params, caps, driver):
    print('Testing blackline')
    n = 0
    
    image = cv2.imread('./lines/4.png', -1)
    image = crop(image, 0.1, 0.4)

    print(image.shape)
    line_mask = blackline_detection(image, threshold=params.cruise_params.threshold)
    targets, targets_image = target_points_detection(line_mask, 1)

    cv2.imshow('frame', image)
    cv2.imshow('line', line_mask)
    cv2.imshow('target_image', targets_image)
    print('Test sign image -- OK')
    c = cv2.waitKey()

    print('Press C to take a photo, restored in test_images/')
    for i in itertools.count():
        frame = query_camera(caps[0], flip=True)
        image = crop(frame, 0.1, 0.4)

        line_mask = blackline_detection(image, threshold=params.cruise_params.threshold)
        target_points, target_image = target_points_detection(line_mask, 1)

        cv2.imshow('frame', frame)
        cv2.imshow('roi', image)
        cv2.imshow('target_image', target_image)
        c = cv2.waitKey(30)

        if c == ord('c'):
            print('test_{:03d}.jpeg'.format(n))
            cv2.imwrite('test_{:03d}.jpeg'.format(n), frame)
            n += 1

        if c == 27 or c == ord('q'):
            break
    print('Test video -- OK')

def test_parksign_detection(params, caps, driver):
    print('Testing parksign')

    image = cv2.imread('./parksigns/1.png', -1)
    print(image.shape)
    print('Parksign: {}'.format(detect_parksign(image, params)))
    print('Test sign image -- OK')

    cv2.imshow('image', image)
    cv2.waitKey()

    for i in itertools.count():
        frame = query_camera(caps[0], flip=True)

        print('Parksign: {}'.format(detect_parksign(frame, params)))

        cv2.imshow('frame', frame)
        c = cv2.waitKey(30)
        if ord('q') == c or 27 == c:
            break
    print('Test video -- OK')

