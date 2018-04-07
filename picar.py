
import smartcarlib
import smartcarlib.cv as cv
import smartcarlib.driver as driver
import smartcarlib.PID as PID

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
    params = PicarParams(threshold=100.0)
    picar = Picar(params)
    picar.test()


