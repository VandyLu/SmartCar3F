import cv2

def circle_detect(img, kernel_size=(5,5)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(gray, kernel_size)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 200, param1=100, param2=30, minRadius=50, maxRadius=300)
    for circle in circles[0]:
        x = int(circle[0])
        y = int(circle[1])
        r = int(circle[2])
        img = cv2.circle(img, (x, y), r, (0, 0, 255), 3)
    cv2.imshow('1', img)
    cv2.waitKey()
    return x, y, r

img = cv2.imread('T4.jpg')
x, y, r = circle_detect(img, (5,5))
print(x)
print(y)
print(r)
