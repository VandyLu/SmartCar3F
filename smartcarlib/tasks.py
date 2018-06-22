import cv
import cv2
import driver
import parksign
import PID
import utils
import time

import numpy as np

import itertools

def cruise(params, caps, driver):
    ''' params: namedtuple
        caps: a list of cv2.VideoCapture
        driver: motion control
    '''
    stop = False

    steer_pid = PID.PID(params.cruise_params.steer_kp,
                        params.cruise_params.steer_ki,
                        params.cruise_params.steer_kd)

    image = utils.query_camera(caps[0], flip=True)


    for t in itertools.count():
        start_time = time.time()

        image = utils.query_camera(caps[0], flip=True)
        
        cruise_roi = utils.crop(image, 0.2, 0.7)
        height, width, channels = cruise_roi.shape

        line_mask = cv.blackline_detection(cruise_roi, params.cruise_params.threshold)
        points, points_image = cv.target_points_detection(line_mask, 1)

        # Take the middle point as target
        middle_x = width / 2
        cur_x, cur_y = points[0, 0], points[0, 1]

        error_x = (middle_x - cur_x) / width * 2 # [0,1]
        steer_value = steer_pid.update(error_x) # positive means turning right
        steer_value = 1.5 * np.clip(steer_value, -1.0, 1.0)

        # Motor speed
        motor_value = 0.04

        driver.setStatus(motor = motor_value, servo = steer_value, mode = 'speed')

        cv2.imshow('frame', image)
        cv2.imshow('roi', cruise_roi)
        cv2.imshow('line', points_image)
        c = cv2.waitKey(params.control_interval)

        if ord('q') == c or 27 == c:
            break

        duration = time.time() - start_time
        print('Time: {:.3f} | Steer: {:.3f}| FPS: {:.3f}'.format(time.time(), steer_value, 1.0 / duration))

        # Stop Flags
        # 1. detect park sign

        #park_sign = parksign.detect_parksign(image, params)
        park_sign = False

        if park_sign:
            print('Time: {} | Parksign detected.'.format(time.time()))
            break

    driver.setStatus(motor = 0., servo = 0., mode = 'speed')
    driver.setStatus(motor = 0., servo = 0., mode = 'speed')
    cv2.waitKey(200)


def park(params, caps, driver):
    pass
