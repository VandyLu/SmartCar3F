
import cv2
import numpy as np

def detect_parksign(image, params):
    ''' WZK
        
        return True if there is park sign in the image 
    '''

    return False


def test_parksign():
    ''' A test function
        Reading an image and run your code, see the outputs
    '''

    image = cv2.imread('', -1)

    has_parksign = detect_parksign(image, None)
    print('Park: {}'.format(has_parksign))

    return

