
import cv2
import numpy as np

def detect_parksign(image, params):
    ''' WZK
        
        return True if there is park sign in the image 
    '''
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = cv2.blur(hsv, kernel_size)
    #lower_blue = np.array([100, 100, 50])
    #upper_blue = np.array([130, 255, 255])
    lower_blue = params.park_params.lower_blue
    upper_blue = params.park_params.upper_blue
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, 
                               1, 200, param1=100, param2=50, 
                               minRadius = params.park_params.min_R, 
                               maxRadius = params.park_params.max_R)

    if isinstance(circles, type(None)):
        return False
    else:
        return True


def test_parksign():
    ''' A test function
        Reading an image and run your code, see the outputs
    '''

    image = cv2.imread('', -1)

    has_parksign = detect_parksign(image, None)
    print('Park: {}'.format(has_parksign))

    return

