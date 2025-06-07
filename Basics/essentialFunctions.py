import cv2 as cv

img = cv.imread('/Users/nirmanpatel36/Documents/OpenCV/Photos/Cat.jpeg')
cv.imshow('Cat', img)

# 1. Converting to grayscale i.e., just displaying the intensity distribution and not the RBG values
'''Convert Color function has two params:
1. Source
2. Color Code'''
converted = cv.cvtColor(img, cv.COLOR_BGR2GRAY)    # RGB and BGR are the same
cv.imshow('Gray', converted)

# 2. Blur an Image
imgblur = cv.imread('/Users/nirmanpatel36/Documents/OpenCV/Photos/Cat2.jpeg')
cv.imshow('Other Cat', imgblur)
'''Gaussian Blur has three params:
1. Source
2. Kernel size (has to be odd)
3. Border Type'''
blur = cv.GaussianBlur(imgblur, (7,7), cv.BORDER_DEFAULT)       # increase the kernel size to increase the intensity
cv.imshow('Blur', blur)

# 3. Edge Cascade i.e., finding all the edges present in an image using the 'Canny' detector
img3 = cv.imread('/Users/nirmanpatel36/Documents/OpenCV/Photos/Sleep Cat.jpeg')
cv.imshow('Sleepy Cat', img3)
'''Canny Function has three main params:
1. Source
2. Threshold_01
3. Threshold_02

NOTE: For more prominent results, blur an image and then cascade it'''
imgEdge = cv.Canny(img3, 125, 175)                             # replace img3 with blur to observe more clear results
cv.imshow('Cascade', imgEdge)

# 4. Dilating an image with a specific structuring element
dilated = cv.dilate(imgEdge, (3,3), iterations=1)
cv.imshow('Dilated', dilated)

# 5. Eroding i.e., remove dilating from an image
eroded = cv.erode(dilated, (3,3), iterations=1)
cv.imshow('Eroded', eroded)

# 6. Resize
'''In the interpolation parameter, if we are downsacling the image, then INTER_AREA is used
if the image is upscaled, we prefer INTER_LINEAR or INTER_CUBIC'''
resized = cv.resize(img, (500,500), interpolation=cv.INTER_CUBIC)
cv.imshow('Resized', resized)

# Cropping
cropped = img[200:300, 300:400]                               # selecting a specific pixel area
cv.imshow('Cropped', cropped)

cv.waitKey(0)