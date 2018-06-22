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
        
        cruise_roi = utils.crop(image, 0.5, 0.4)
        height, width, channels = cruise_roi.shape

        line_mask = cv.blackline_detection(cruise_roi, params.cruise_params.threshold)
        points, points_image = cv.target_points_detection(line_mask, 1)

        # Take the middle point as target
        middle_x = width / 2
        cur_x, cur_y = points[0, 0], points[0, 1]

        error_x = (middle_x - cur_x) / width * 2 # [0,1]
        steer_value = steer_pid.update(error_x) # positive means turning right
        steer_value = np.clip(steer_value, -1.0, 1.0)

        # Motor speed
        motor_value = 0.4

        #driver.setStatus(motor = motor_value, servo = -steer_value, mode = 'speed')

        #while 1000 * (time.time() - start_time) < params.cruise_params.control_interval:
        #    pass
        cv2.imshow('frame', image)
        cv2.imshow('line', points_image)
        c = cv2.waitKey(params.control_interval)

        if ord('q') == c or 27 == c:
            break

        duration = time.time() - start_time
        print('Time: {} | FPS: {}'.format(time.time(),1.0 / duration))
        print(steer_value)

        # Stop Flags
        # 1. detect park sign

        park_sign = parksign.detect_parksign(image, params)

        if park_sign:
            print('Time: {} | Parksign detected.'.format(time.time()))
            break


def park(params, caps, driver):
    pass
