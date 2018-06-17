import cv
import cv2
import driver
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

    steer_pid = PID.PID(params.steer_kp,
                        params.steer_ki,
                        params.steer_kd)

    image = utils.query_camera(caps[0], flip=True)
    height, width, channels = image.shape


    for t in itertools.count() and not stop:
        start_time = time.time()

        image = utils.query_camera(caps[0], flip=True)

        line_mask = cv.blackline_detection(image, params.threshold, method='close')
        points, points_image = cv.target_points_detection(line_mask, 3)
        # (N, 2)

        # Take the middle point as target
        middle_x = width / 2
        cur_x, cur_y = points[1, 0], points[1, 1]

        error_x = (middle_x - cur_x) / width * 2 # [0,1]
        steer_value = steer_pid.update(error_x) # positive means turning right
        steer_value = np.clip(steer_value, -1.0, 1.0)

        # Motor speed
        motor_value = 0.4

        driver.setStatus(motor = motor_value, servo = -steer_value, mode = 'speed')

        while 1000 * (time.time() - start_time) < params.control_interval:
            pass

        duration = time.time() - start_time
        print('FPS: {}'.format(1.0 / duration))

        # Stop Flags
        # 1. detect park sign

        park_sign = False

        if park_sign:
            stop = True
            


def park(params, caps, driver):
    pass
