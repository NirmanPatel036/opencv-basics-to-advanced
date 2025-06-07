import cv2 as cv
import numpy as np

blank = np.zeros((400,400), dtype='uint8')

rectangle = cv.rectangle(blank.copy(), (30,30), (370,370), 255, -1)
circle = cv.circle(blank.copy(), (200,200), 200, 255, -1)

cv.imshow('Rectangle', rectangle)
cv.imshow('Circle', circle)

#bitwise and --> returns the intersection of two images
bitwiseAnd = cv.bitwise_and(rectangle, circle)
cv.imshow('AND', bitwiseAnd)

#bitwise or --> returns the union of two images
bitwiseOr = cv.bitwise_or(rectangle, circle)
cv.imshow('OR', bitwiseOr)

#bitwise xor --> returns the exclusive or of two images
bitwiseXor = cv.bitwise_xor(rectangle, circle)
cv.imshow('XOR', bitwiseXor)

#bitwise not --> returns the inverse of the image
bitwiseNot = cv.bitwise_not(rectangle)
cv.imshow('NOT', bitwiseNot)

cv.waitKey(0)