
from smartcarlib import *

import cv2
from collections import namedtuple



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
        self.driver = driver.driver()

    def cruise(self):
        tasks.cruise(self.params, self.cap, self.driver)

    def park(self):
        ''' Park to specified plot
        '''
        pass

    def test(self):
        ''' Test to run on picar
        '''
        tests.__test_blackline_detection(self.params, self.cap, self.driver)
        print('Test blackline: OK')

    
if __name__ == '__main__':
    # Test Picar
    params = PicarParams(threshold=100.0,
                         steer_kp=1.0,
                         steer_ki=0.0,
                         steer_kd=0.0,
                         control_interval=10) 

    picar = Picar(params)
    picar.test()

    print('Exit done!')
    picar.driver.close()

