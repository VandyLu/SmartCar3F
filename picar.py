
from smartcarlib import *

import cv2
from collections import namedtuple



PicarParams = namedtuple('PicarParams',
                         ['control_interval', # one interval=50ms
                          'cruise_params',
                          'park_params'
                         ])

ParkParams = namedtuple('ParkParams',
                        ['lower_blue',
                         'upper_blue',
                         'min_R',
                         'max_R',
                         'steer_kp',
                         'steer_ki',
                         'steer_kd',
                         'parklot_idx'
                        ])

CruiseParams = namedtuple('CruiseParams',
                          ['threshold',
                           'steer_kp',
                           'steer_ki',
                           'steer_kd'])

class Picar():

    def __init__(self, params):
        self.params = params

        self.front = 0
        self.back = 1
        self.cap = [cv2.VideoCapture(i) for i in [1,0]]
        #self.cap = []
        self.driver = None #
        self.driver = driver.driver()

    def cruise(self):
        tasks.cruise(self.params, self.cap, self.driver)

    def park(self):
        ''' Park to specified plot
        '''
        tasks.park(self.params, self.cap, self.driver)

    def test(self):
        ''' Test to run on picar
        '''
        #tests.test_blackline_detection(self.params, self.cap, self.driver)
        tests.test_parksign_detection(self.params, self.cap, self.driver)
        print('Test blackline: OK')

    
if __name__ == '__main__':
    # Test Picar
    cruise_params = CruiseParams(threshold=100,
                                 steer_kp=1.2,
                                 steer_ki=0.01,
                                 steer_kd=0.0)

    park_params = ParkParams(lower_blue = [100, 100, 50],
                             upper_blue = [130, 255, 255],
                             min_R = 10,
                             max_R = 300,
                             steer_kp = 1.0,
                             steer_ki = 0.01,
                             steer_kd = 0.0,
                             parklot_idx = 1)

    params = PicarParams(control_interval=600,
                         cruise_params = cruise_params,
                         park_params = park_params) 

    picar = Picar(params)
    #picar.cruise()
    #picar.test()
    picar.park()

    print('Exit done!')
    picar.driver.close()
