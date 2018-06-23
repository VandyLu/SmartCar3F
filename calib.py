#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2  
import numpy as np  
import glob  
  
# 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001  
criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)  
  
# 获取标定板角点的位置  
nx, ny = 9, 6
objp = np.zeros((nx*ny,3), np.float32)  
objp[:,:2] = np.mgrid[0:ny,0:nx].T.reshape(-1,2)  # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y  
  
obj_points = []    # 存储3D点  
img_points = []    # 存储2D点  
  
images = glob.glob("./calib/*.png")
#images = glob.glob("/home/lfb/testing/*.jpg")
for fname in images:  
    img = cv2.imread(fname)  
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    size = gray.shape[::-1]  
    ret, corners = cv2.findChessboardCorners(gray, (ny,nx), None)  
  
    if ret:  
        obj_points.append(objp)  
  
        corners2 = cv2.cornerSubPix(gray, corners, (5,5), (-1,-1), criteria)  # 在原角点的基础上寻找亚像素角点  
        if not isinstance(corners2, type(None)):  
            img_points.append(corners2)  
        else:  
            img_points.append(corners)  
  
        cv2.drawChessboardCorners(img, (ny,nx), corners, ret)   # 记住，OpenCV的绘制函数一般无返回值  
        cv2.imshow('img', img)  
        cv2.waitKey(50)  
  
print len(img_points)  
cv2.destroyAllWindows()  
  
# 标定  
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points,size, None, None)  
  
print "ret:",ret  
print "mtx:\n",mtx        # 内参数矩阵  
print "dist:\n",dist      # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)  
print "rvecs:\n",rvecs    # 旋转向量  # 外参数  
print "tvecs:\n",tvecs    # 平移向量  # 外参数  

#np.savetxt('params/mtx.txt', mtx)
#np.savetxt('params/dist.txt', dist)
  
print("-----------------------------------------------------")  
# 畸变校正  
img = cv2.imread(images[10])  
h, w = img.shape[:2]  
print("------------------使用undistort函数-------------------")  
#dst = cv2.undistort(img,mtx,dist,None,newcameramtx)  
dst = cv2.undistort(img,mtx,dist,None,mtx)  

out = np.concatenate([img, dst], axis=1) 
cv2.imshow('out', out)
cv2.imshow('img',img)
cv2.imwrite('calibresult11.png', out)  
cv2.imshow('result', dst)
cv2.waitKey()
#print "方法一:dst的大小为:", dst1.shape  

exit()
  
# undistort方法二  
print("-------------------使用重映射的方式-----------------------")  
mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, np.diag([1.0, 1.0, 1.0]), newcameramtx, (w,h), cv2.CV_32FC1)  # 获取映射方程  
print(mapx, mapy)
#dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)      # 重映射  
dst = cv2.remap(img,mapx,mapy,cv2.INTER_CUBIC)        # 重映射后，图像变小了  
x,y,w,h = roi  
#dst2 = dst[y:y+h,x:x+w]  
dst2 = dst
cv2.imwrite('calibresult11_2.jpg', dst2)  
print "方法二:dst的大小为:", dst2.shape        # 图像比方法一的小  
  
print("-------------------计算反向投影误差-----------------------")  
tot_error = 0  
for i in xrange(len(obj_points)):  
    img_points2, _ = cv2.projectPoints(obj_points[i],rvecs[i],tvecs[i],mtx,dist)  
    error = cv2.norm(img_points[i],img_points2, cv2.NORM_L2)/len(img_points2)  
    tot_error += error  
  
mean_error = tot_error/len(obj_points)  
print "total error: ", tot_error  
print "mean error: ", mean_error  

