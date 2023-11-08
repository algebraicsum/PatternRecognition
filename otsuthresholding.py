import numpy as np 
import cv2 as cv

# Reference  https://www.youtube.com/watch?v=9XxJ9j5oIrk (Within Class Variance)
# Reference https://www.youtube.com/watch?v=SU9Xyhdq9Zc (Between Class Variance)

image = cv.imread("Nyoba.png")
image = cv.resize(image,(800,600))
image_gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
# cv.imshow('gray image', image_gray)
# cv.waitKey(0)

histogram = cv.calcHist([image_gray],[0],None,[255],[0,255])
# print(histogram)
within = []
between = []
d = 0
for i in range(len(histogram)):
    x,y = np.split(histogram,[i])
    x1 = np.sum(x)/(image.shape[0]*image.shape[1])
    y1 = np.sum(y)/(image.shape[0]*image.shape[1])
    x2 = np.sum ([j*t for j,t in enumerate(x)])/np.sum(x)
    x2 = np.nan_to_num(x2)
    y2 = np.sum ([(j+d)*(t) for j,t in enumerate(y)])/np.sum(y)
    x3 = np.sum ([(j-x2)**2*t for j,t in enumerate(x)])/np.sum(x)
    x3 = np.nan_to_num(x3)
    y3 = np.sum ([(j+d-y2)**2*t for j,t in enumerate(y)])/np.sum(y)
    d = d+1
    within.append(x1*x3 + y1*y3)
    between.append(x1*y1*(x2-y2)*(x2-y2))
m = np.argmin(within)
n = np.argmax(between)
print(m)
print(n)
(thresh, Bin) = cv.threshold(image_gray,m,255,cv.THRESH_BINARY)
cv.imshow("Binary", Bin)
cv.waitKey(0)