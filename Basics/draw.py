import cv2 as cv
import numpy as np

'''we will create a blank image using the zeros() method where it conists of two parameters:
1. Dimensions - WIDTH, HEIGHT, NO. OF COLOUR CHANNELS
2. Data Type of the image, i.e, uint8

VVVIMP NOTE: The co-ordinates are used in the form of (column, row) of the given image matrix
'''
blank = np.zeros((500,500,3), dtype='uint8')    # blank is a 2D-array of dimns: 500x500 consisting of all 0's
cv.imshow('Blank', blank)

# 1. Paint the image with a certain colour
blank[200:300, 300:400] = 223, 208, 184         # we can spcify the range of pixels for the colour(rgb values) to be applied
cv.imshow('Beige', blank)

blank[:] = 123, 201, 255                        # a colon means the colour(rgb values) will be applied to the whole image
cv.imshow('Brown', blank)

# 2. Draw a rectangle
newImg = np.zeros((500,500,3), dtype='uint8')
'''rectangle function has five parameters:
1. Image file name
2. Point_01
3. Point_02
4. RGB value
5. Thickness'''
cv.rectangle(newImg, (0,0), (250,250), (0, 255, 0), thickness=cv.FILLED)
#alternatively, cv.rectangle(newImg, (0,0), (blank.shape[1]//2, blank.shape[0]//2), (0, 255, 0), thickness=cv.FILLED)
cv.rectangle(newImg, (251,251), (498,498), (0, 0, 255), thickness=2)
#cv.imshow('Rectangle', newImg)

# 3. Draw a circle
'''circle function has five parameters:
1. Image file name
2. Centre Point
3. Radius in Pixels
4. RGB value
5. Thickness'''
cv.circle(newImg, (250, 250), 100, (0, 0, 255), thickness=5)
cv.circle(newImg, (250, 250), 100, (255, 0, 0), thickness=-1)    # thickness of -1 will fill in the provided colour
#cv.imshow('Circle', newImg)

# Draw a line
'''line function has five parameters:
1. Image file name
2. Point_01
3. Point_02
4. RGB value
5. Thickness'''
cv.line(newImg, (0,0), (500, 500), (255, 255, 255), thickness=4)
#cv.imshow('Line', newImg)

# Write Text
'''putText function has five parameters:
1. Image file name
2. Text String
3. Point of origin
4. Font Face
5. Font Scale
6. RGB Value of the font
7. Thickness'''
cv.putText(newImg, 'Lorem Ipsum', (160, 255), cv.FONT_HERSHEY_TRIPLEX, 0.75, (0,0,0), thickness=2)
cv.imshow('Draw', newImg)

cv.waitKey(0)