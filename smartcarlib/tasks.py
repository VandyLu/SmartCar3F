import cv
import cv2
import driver
import parksign
import PID
import utils
import time
import parklot

import numpy as np

import itertools

def cruise(params, caps, driver):
    ''' params: namedtuple
        caps: a list of cv2.VideoCapture
        driver: motion control
    '''
    stop = False
    park_cnt = 0

    steer_pid = PID.PID(params.cruise_params.steer_kp,
                        params.cruise_params.steer_ki,
                        params.cruise_params.steer_kd)

    image = utils.query_camera(caps[0], flip=True)

    servo_last = 0
    signal_last = time.time()


    for t in itertools.count():
        start_time = time.time()

        image = utils.query_camera(caps[0], flip=True)
        
        cruise_roi = utils.crop(image, 0.1, 0.9)
        height, width, channels = cruise_roi.shape

        line_mask = cv.blackline_detection(cruise_roi, params.cruise_params.threshold)
        points, points_image = cv.target_points_detection(line_mask, 3)

        if points.shape[0] != 3:
            if time.time() - signal_last > 0.5:
                driver.setStatus(motor = 0.05, servo = servo_last, mode = 'speed')
                servo_last = np.clip(1.1 * servo_last, -1.0, 1.0)
                signal_last = time.time()
            #c = cv2.waitKey(params.control_interval)
            c = cv2.waitKey(100)
            continue

        # Take the middle point as target
        middle_x = width / 2
        cur_x, cur_y = points[0, 0], points[0, 1]

        error_x = (middle_x - cur_x) / width * 2 # [0,1]
       	print(error_x)
        steer_value = steer_pid.update(error_x) # positive means turning right
        steer_value = np.clip(steer_value, -1.0, 1.0)

        # Motor speed
        motor_value = 0.06

        if time.time() - signal_last > 0.5:
            driver.setStatus(motor = motor_value, servo = steer_value, mode = 'speed')
            signal_last = time.time()
            servo_last = steer_value

        cv2.imshow('frame', image)
        cv2.imshow('roi', cruise_roi)
        cv2.imshow('line', points_image)
        #c = cv2.waitKey(params.control_interval)
        c = cv2.waitKey(100)

        if ord('q') == c or 27 == c:
            break

        duration = time.time() - start_time
        print('Time: {:.3f} | Steer: {:.3f}| FPS: {:.3f}'.format(time.time(), steer_value, 1.0 / duration))

        # Stop Flags
        # 1. detect park sign

        park_sign = parksign.detect_parksign(image, params)
        #park_sign = False

        if park_sign:
            driver.setStatus(motor = 0.0, servo = 0.0, mode='speed')
            park_cnt += 1

        if park_cnt > 5:
            print('Time: {} | Parksign detected.'.format(time.time()))
            break

    driver.setStatus(motor = 0., servo = 0., mode = 'speed')
    driver.setStatus(motor = 0., servo = 0., mode = 'speed')
    cv2.waitKey(200)

class Clock():

    def __init__(self, interval):
        self.last = time.time()
        self.interval = interval

    def update(self):
        if time.time() - self.last > self.interval:
            self.last = time.time()
            return True
        else:
            return False


def park(params, caps, driver):
    clock = Clock(0.5)

    steer_pid = PID.PID(params.park_params.steer_kp,
                        params.park_params.steer_ki,
                        params.park_params.steer_kd)
    cv2.waitKey(1000)

    stop_cnt = 0

    for t in itertools.count():
        img = utils.query_camera(caps[1], flip=False)
        #img = cv2.imread('./parklot/27.png', -1)

        idx = params.park_params.parklot_idx
        img = parklot.park_preprocess(img)

        mask = parklot.park_color_detection(img, idx)

        cv2.imshow('bin', mask)
        cv2.imshow('img', img)

        mask_close = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.float32))

        if mask_close.any():
            meanx, meany = calculate_center(mask_close)
        else:
            print('color failed')
            cv2.waitKey(200)
            continue
            
        #edge = cv2.Canny(img, 50, 150)
        #edge = cv2.dilate(edge, np.ones((3,3)))
        #edge  = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, np.ones((5,5), np.float32))
        #edge *= np.logical_not(mask_close>1).astype(np.uint8) # no edges if color detected

        #origin = edge.copy()
        #mask = np.zeros((edge.shape[0]+2, edge.shape[1]+2), dtype=np.uint8)
        #cv2.floodFill(edge, mask, (int(meanx), int(meany)), 255)#, flags=cv2.FLOODFILL_MASK_ONLY|cv2.FLOODFILL_FIXED_RANGE)

        #valid = True
        #while True:
        #    filled = edge - origin
        #    num_filled = np.sum(filled) / 255.0
        #    if num_filled < 1000:
        #        for i in range(40):
        #            if edge[int(meany), int(meanx+i)] < 1:
        #                print(i)
        #                mask = np.zeros((edge.shape[0]+2, edge.shape[1]+2), dtype=np.uint8)
        #                cv2.floodFill(edge, mask, (int(meanx), int(meany)), 255)#, flags=cv2.FLOODFILL_MASK_ONLY|cv2.FLOODFILL_FIXED_RANGE)
        #                meanx = meanx+i
        #                break
        #        valid = False
        #    else: 
        #        break
            
        #filled

        cv2.circle(img, (int(meanx), int(meany)), 3, [255, 0, 0], -1) 

        src = parklot.park_contour_process(mask_close, img)
        pt_left, pt_right = src[0], src[1]
        if pt_left[1] < 80 and pt_right[1] < 80:
            stop_cnt += 1
            # in lot

            if stop_cnt > 5:
                break

        else:
            # target
            error = (img.shape[1]/2 - mean_x) / img.shape[1] 
            servo = np.clip(steer_pid.update(error), -1.0, 1.0)
            if clock.update():
                driver.setStatus(motor=-0.02, servo =servo, mode='speed')

        
        if type(src) == type(None) or not valid or src.shape[0] !=4 :
            print('n_pt: {}'.format(src))
            cv2.waitKey(200)
            continue
        m = parklot.getm(img, src)

        if type(m) == type(None):
            print(src)
            print('Fail')
            cv2.waitKey(200)
            continue

        #result = cv2.warpPerspective(img, m, (img.shape[1], img.shape[0]))
        #pt, theta = parklot.getCar(img.shape[1], img.shape[0], m)
        #for i in range(src.shape[0]):
        #    cv2.circle(img, tuple(src[i].astype(np.int32)), 3, color=[0,255,0], thickness=-1)

        #x0, y0 = pt[0], pt[1]
        #x1, y1 = result.shape[1]//2, result.shape[0]//2
        #dy = 10
        #alpha = (x1-x0)/(y1-y0)**2 
        #grad_dydx = -1.0/(2*alpha*(y0-y1)**2)
        #fi = np.arctan(grad_dydx)
        
        #error = (fi - theta) / np.pi # if +, should turn left ,steer +
        #steer_value = steer_pid.update(error)
        #steer_value = np.clip(steer_value, -1.0, 1.0)
        #if clock.update():
        #    driver.setStatus(motor = -0.02, servo = steer_value, mode='speed')

        #print('n_pt: {} | steer: {:.3f}'.format(src.shape[0], steer_value))

        #cv2.circle(result, tuple(pt.astype(np.int32)), 3, color=[255, 0, 255], thickness=-1)
        #cv2.imshow('filled', filled)
        #cv2.imshow('edge', edge)
        #cv2.imshow('img', img)
        #cv2.imshow('result', result)
        c = cv2.waitKey(100)
        if ord('q') == c or 27 == c:
            break

    driver.setStatus(motor = 0., servo = 0., mode = 'speed')
    driver.setStatus(motor = 0., servo = 0., mode = 'speed')
    cv2.waitKey(200)

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
    


