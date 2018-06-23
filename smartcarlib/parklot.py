
import cv2
import numpy as np

hsv = False
mouse_ok = False
mouse_x=0
mouse_y=0
mouse_img = None
def getCord(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONUP:
        mouse_ok = True
        mouse_x,mouse_y = x,y
        print((mouse_x,mouse_y),mouse_img[y,x])


def park_preprocess(img):
    img = cv2.blur(img, (3,3))

    if hsv:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        #channels = cv2.split(img)
        #merge = [cv2.equalizeHist(i, i)for i in channels]
        #img = cv2.merge(merge)

        #kernel = np.ones((5, 5), np.uint8)
        #img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        #img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    return img

def park_color_detection(img, color):

    #  1~4 represent the color of parkplot number 
    if hsv:
        color_dic_lower = {1: np.array([100, 100, 100]), 2: np.array([150, 100, 100]), 3: np.array([25, 100, 100]), 4: np.array([100, 100, 100])}
        color_dic_upper = {1: np.array([150, 255, 255]), 2: np.array([200, 255, 255]), 3: np.array([50, 255, 255]), 4: np.array([110, 255, 255])}
        lower = color_dic_lower[color]
        upper = color_dic_upper[color]

        mask = cv2.inRange(img, lower, upper)
    else:
        color_dic_lower = np.array([[70, 70, 50], [00,0,155], [0,150,140],[60,105,0]])
        color_dic_upper = np.array([[95, 95, 75], [80,55,220],[30,180,180],[110,160,60]])
        lower = color_dic_lower[color]
        upper = color_dic_upper[color]
        print(img.shape)

        mask = cv2.inRange(img, lower, upper)

    return mask


def park_contour_process(img, img_prv):

    emptyImage = np.zeros(img.shape, np.uint8)

    _, contours, hierarchy = cv2.findContours(img,
                                              cv2.RETR_TREE,
                                              cv2.CHAIN_APPROX_NONE)
    cnt = contours[0]
    max_area = cv2.contourArea(cnt)

    for cont in contours:
        if cv2.contourArea(cont) > max_area:
            cnt = cont
            max_area = cv2.contourArea(cont)

    #  k important paremeter!!!
    k = 0.02
    epsilon = k*cv2.arcLength(cnt,True)
    approx = cv2.approxPolyDP(cnt,epsilon,True)

    result = np.squeeze(resort(approx),1)

    cv2.circle(img_prv, tuple(result[0]), 3, (0,255,0), -1)
    cv2.circle(img_prv, tuple(result[1]), 3, (255,0,0), -1)
    cv2.circle(img_prv, tuple(result[2]), 3, (0,0,255), -1)
    cv2.circle(img_prv, tuple(result[3]), 3, (255,255,255), -1)

    print(approx)

    cv2.drawContours(img_prv, [approx], -1, (255, 255, 255), 1)
    cv2.imshow('prv', img_prv)
    return result[0], result[3]

def resort(approx):
    x0 = approx[0][0][0]
    y0 = approx[0][0][1]
    x1 = approx[1][0][0]
    y1 = approx[1][0][1]
    x2 = approx[2][0][0]
    y2 = approx[2][0][1]
    x3 = approx[3][0][0]
    y3 = approx[3][0][1]

    x_array =[x0, x1, x2, x3]
    y_array =[y0, y1, y2, y3]

    sum_array = [x0+y0, x1+y1, x2+y2, x3+y3]

    index_prm = sum_array.index(min(sum_array))

    # index_array = [index0, index1, index2, index3]

    result = approx.copy()

    result[0][0][0] = x_array[index_prm]
    result[0][0][1] = y_array[index_prm]
    result[1][0][0] = x_array[(index_prm+1)%4]
    result[1][0][1] = y_array[(index_prm+1)%4]
    result[2][0][0] = x_array[(index_prm+2)%4]
    result[2][0][1] = y_array[(index_prm+2)%4]
    result[3][0][0] = x_array[(index_prm+3)%4]
    result[3][0][1] = y_array[(index_prm+3)%4]

    return result

def getm(img, src):
    height, width = img.shape[:2]
    pcenter = width//2, height//2
    # 24:40 = 3:5
    boxx, boxy = 100*0.3, 100*0.5
    dst = [(width//2 - boxx, height//2 - boxy), 
       (width//2 - boxx, height//2 + boxy), 
       (width//2 + boxx, height//2 + boxy), 
       (width//2 + boxx, height//2 - boxy)]

    dst = np.array(dst, dtype=np.float32)

    print(src)
    if not src.shape[0] == 4:
        print('Error in parklot detection')
        return None

    m = cv2.getPerspectiveTransform(src.astype(np.float32), dst)
    return m

def warpPoints(src, m):
    ''' src: (2, n)
        dst: (2, n)
    '''
    n = src.shape[1]
    src = np.concatenate([src, np.ones([1, n])], axis=0)
    dst = np.matmul(m, src)
    
    dst = np.array([dst[i]/dst[2] for i in range(2)])
    return dst


def getCar(width, height, m):
    ''' middle_pt: (x,y)
        p: (b,k)
    '''
    src = np.array([[0, width/2, width], 
                    [height, height, height]], np.float32)

    dst = warpPoints(src, m)
    middle_pt = dst[:, 1]

    lstart = dst[:, 0]
    lend = dst[:, 2]

    x0 = lstart[0]
    x1 = lend[0]
    y0 = lstart[1]
    y1 = lend[1]

    #p = np.mat([[1.0, x0], [1.0, x1]], np.float32).I * np.array([[y0], [y1]])
    p = np.arctan2(y1-y0, x1-x0) - np.pi/2
    return middle_pt, p


if __name__ == '__main__':
    img = cv2.imread('./parklot/14.png', -1)
    img = park_preprocess(img)
    mask = park_color_detection(img,3)
    
    src = park_contour_process(mask, img)
    m = getm(img, src)
    
    if not type(m) == type(None):
        exit()
        
    
    result = cv2.warpPerspective(img, m, (img.shape[1], img.shape[0]))
    pt, p = getCar(img.shape[1], img.shape[0], m)
    
    cv2.circle(img, tuple(pt.astype(np.int32)), 3, color=[255, 0, 255], thickness=-1)
    
    mouse_img = img
    cv2.imshow('img', img)
    cv2.imshow('mask', mask)
    cv2.imshow('result', result)
    cv2.setMouseCallback('img',getCord)
    cv2.waitKey()
