
import smartcarlib
import smartcarlib.cv as cv
import smartcarlib.driver as driver
import smartcarlib.PID as PID

import cv2
from collections import namedtuple
import itertools

PicarParams = namedtuple('PicarParams',
                         ['threshold',
                          'steer_kp',
                          'steer_ki',
                          'steer_kd',
                          'control_interval' # one interval=50ms
                         ])

class Picar():

    def __init__(self, params):
        self.params = params

        self.front = 0
        self.back = 1
        self.cap = [cv2.VideoCapture(i) for i in range(2)]
        self.driver = smartcarlib.driver.driver()

    def cruise(self):
        ''' Cruise black line
        '''
        stop = False
        steer_pid = PID.PID(self.params.steer_kp,
                            self.params.steer_ki,
                            self.params,steer_kd)

        for t in itertools.count():
            image = self.query_camera(self.front, flip=True)
            height, width, _ = image.shape

            line_mask = cv.blackline_detection(image, self.params.threshold, method='close')
            points, points_image = cv.target_points_detection(line_mask, 3)
            print(points.shape)
        
            # Take the middle point as target
            middle_x = width / 2
            target_x, target_y = points[1, 0], points[1, 1]

            # Limit the frequency of control signal
            if t % self.params.control_interval == 0:
                error_x = (target_x - middle_x) / width # [0,1)
                steer_value = steer_pid.update(error_x) # positive -> turn right
                self.driver.setStatus(motor=0.3, servo=steer_value, mode='speed')

            cv2.imshow('frame', image)
            cv2.imshow('points', points_image)

            if 27 == cv2.waitKey(50): # Manually stop
                stop = True
        
            if stop:
                self.driver.setStatus(motor=0, servo=0, mode='speed')
                break
        
        print('Picar-cruise stopped, time: {}s'.format(t*50/1000))


    def park(self):
        ''' Park to specified plot
        '''
        pass

    def query_camera(self, cam_id, flip=True):
        ''' Get a frame from front/back camera
            Input:
                cam_id (int) Camera index
                flip (bool) Flip image upside down or not
        '''
        assert cam_id in (0, 1)

        ret, frame = self.cap[cam_id]
        if flip:
            frame = cv2.flip(frame, 0)
        return frame

    def test(self):
        ''' Test to run on picar
        '''
        self.__test_blackline_detection()

    def __test_blackline_detection(self):
        while True:
            frame = self.query_camera(self.front, flip=True)
            line_mask = cv.blackline_detection(frame, threshold=self.threshold)
            target_points, target_image = cv.target_points_detection(line_mask, 3)

            cv2.imshow('frame', frame)
            cv2.imshow('target_image', target_image)
            c = cv2.waitKey(30)
            if c == 27 or c == ord('q'):
                break



if __name__ == '__main__':
    # Test Picar
    params = PicarParams(threshold=100.0,
                         steer_kp=1.0,
                         steer_ki=0.0,
                         steer_kd=0.0,
                         control_interval=10) 
    picar = Picar(params)
    picar.test()


