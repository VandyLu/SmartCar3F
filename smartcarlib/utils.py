import cv2


def query_camera(capture, flip=True):
    ''' Get a frame from front/back camera
        Input:
            cam_id (int) Camera index
            flip (bool) Flip image upside down or not
    '''
    ret, frame = capture.read()
    if flip:
        frame = cv2.flip(frame, 0)
    return frame
