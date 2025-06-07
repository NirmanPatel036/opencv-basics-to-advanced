import cv2 as cv
import numpy as np

img = cv.imread('/Users/nirmanpatel36/Documents/OpenCV/Photos/Black&White.jpeg')
cv.imshow('Two Cats', img)

# Translation
def translate(img, x, y):
	transMatrix = np.float32([[1,0,x], [0,1,y]])
	dimensions = (img.shape[1], img.shape[0])
	return cv.warpAffine(img, transMatrix, dimensions)

'''
-x --> Left
-y --> Up
x --> Right
y --> Down '''

translated = translate(img, 100, 100)
cv.imshow('Translated', translated)

# Rotation
def rotate(img, angle, rotPoint=None):
	(height, width) = img.shape[:2]

	if rotPoint is None:
		rotPoint = (width//2, height//2)

	'''The below function has three params:
	1. Pivot for rotation
	2. Angle of rotation
	3. Scale of the image'''	
	rotMatrix = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
	dimensions = (width, height)

	return cv.warpAffine(img, rotMatrix, dimensions)

'''The function creates a transformation matrix (transMatrix) using np.float32() which is a NumPy function
to create an array of floating-point numbers. This matrix represents the translation transformation.
The transformation matrix is a 2x3 matrix where the first row corresponds to the transformation of x-coordinate
and the second row corresponds to the transformation of y-coordinate.
cv.warpAffine() is then used to apply the translation transformation to the image (img). This function takes the image,
the transformation matrix, and the dimensions of the output image as input and returns the transformed image.'''

rotated = rotate(img, 45)                # a negative value for the angle will rotate it clock-wise
cv.imshow('Rotated', rotated)

# Resizing
resized = cv.resize(img, (500,700), interpolation=cv.INTER_CUBIC)
cv.imshow('Resized', resized)

# Flipping
'''The flip function has two params:
1. Source
2. Flip Code
		--> '0' will flip it about x-axis
		--> '1' will flip it abuot y-axis
		--> '-1' will flip it about x- as well as y-axis'''
flip = cv.flip(img, -1)
cv.imshow('Flipped', flip)


































cv.waitKey(0)