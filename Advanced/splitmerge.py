import cv2 as cv
import numpy as np

img = cv.imread('/Users/nirmanpatel36/Documents/OpenCV/Photos/Cat.jpeg')
cv.imshow('Cat', img)

blank = np.zeros(img.shape[:2], dtype='uint8')

#Splitting an image
b, g, r = cv.split(img)

blue = cv.merge([b, blank, blank])
green = cv.merge([blank, g, blank])
red = cv.merge([blank, blank, r])

cv.imshow('Blue', blue)
cv.imshow('Green', green)
cv.imshow('Red', red)

print(img.shape)
print(b.shape)
print(g.shape)
print(r.shape)

#Merging an image
merged = cv.merge([b, g, r])
cv.imshow('Merged', merged)

cv.waitKey(0)