
import smartcarlib
import smartcarlib.cv
import smartcarlib.driver
import smartcarlib.PID

import cv2
import collections.namedtuple

PicarParams = namedtuple('PicarParams',
                         ['threshold'])

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
        
        image = self.query_camera(self.front)

        # Segment the black line to points list nx2
        points = smartcarlib.cv.line_segment(image, self.params.threshold)

        # Get target 


    def park(self):
        ''' Park to specified plot
        '''
        pass

    def query_camera(self, cam_id):
        ''' Get a frame from front/back camera
        '''
        assert cam_id in (0, 1)

        ret, frame = self.cap[cam_id]
        return frame


if __name__ == '__main__':
    print smartcarlib.cv
