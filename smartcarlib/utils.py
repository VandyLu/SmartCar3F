import cv2
import numpy as np


def crop(image, hratio, wratio):
    ''' crop the downside of the image
    '''
    height, width = image.shape[0], image.shape[1]

    hstart = int(height * (1.0 - hratio))
    wstart = int(width * (1.0 - wratio) / 2.0)
    wend = wstart + int(width * wratio)
    return image[hstart: , wstart:wend]


def query_camera(capture, flip=True):
    ''' Get a frame from front/back camera
        Input:
            cam_id (int) Camera index
            flip (bool) Flip image upside down or not
    '''
    ret, frame = capture.read()
    if flip:
        frame = frame[::-1, ::-1]

    return frame

def loadparam():
    mtx = np.loadtxt('./params/mtx.txt')
    dist = np.loadtxt('./params/dist.txt')
    return mtx, dist


def undistort(img, mtx, dist):
    return cv2.undistort(img,mtx,dist,None,mtx)  
