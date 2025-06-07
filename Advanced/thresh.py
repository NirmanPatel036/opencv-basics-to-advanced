import cv2 as cv
import numpy as np

img = cv.imread('/Users/nirmanpatel36/Documents/OpenCV/Photos/Cat.jpeg')
cv.imshow('Cat', img)

#Grayscaling
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

#simple thresholding
threshold, thresh = cv.threshold(gray, 150, 255, cv.THRESH_BINARY)
cv.imshow('Simple Thresholded', thresh)

#inverse thresholding
threshold, thresh_inv = cv.threshold(gray, 150, 255, cv.THRESH_BINARY_INV)
cv.imshow('Inverse Thresholded', thresh_inv)

#adaptive thresholding (mean or gaussian)
adaptive_thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 1)
cv.imshow('Adaptive Thresholded', adaptive_thresh)

cv.waitKey(0)