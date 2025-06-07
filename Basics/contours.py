import cv2 as cv
import numpy as np

img = cv.imread('/Users/nirmanpatel36/Documents/OpenCV/Photos/Cat.jpeg')
cv.imshow('Cat', img)

blank = np.zeros(img.shape, dtype='uint8')

#Grayscaling
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

#Cascading
edges = cv.Canny(gray, 125, 255)
cv.imshow('Canny Edges', edges)

#Blurring
blur = cv.GaussianBlur(gray, (5,5), cv.BORDER_DEFAULT)

#Threshold i.e., to binarize an image
'''here, if
--> the pixel density is below threshold value_01, it is set to 0
--> the pixel density is above threshold value_01, it is set to 255'''
ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)
cv.imshow('Thresh', thresh)

'''The contour method has 3 params:
1. A structuring element
2. A mode of finding the contours
3. Contour Approximation Method i.e., how do we want to approximate our contours

contour is a Python list of all the coordinates of the contours that were found in the image..
hierarchies refer to the hierarchical representation of the contours, for eg, square inside a rectangle inside a circle..'''
contours, hierarchies = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
print(f'{len(contours)} contour(s) found!')

#Drawing the contours, i.e., tracing the detected contours using the blank image
'''it has 5 params:
1. Blank image
2. Contours found
3. How many contours do we want to see
4. Color of the contours
5. Thickness'''
cv.drawContours(blank, contours, -1, (0,0,255), 1)
cv.imshow('Contours', blank)

cv.waitKey(0)