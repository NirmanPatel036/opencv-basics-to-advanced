import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('/Users/nirmanpatel36/Documents/OpenCV/Photos/Cat.jpeg')
cv.imshow('Cat', img)

blank = np.zeros(img.shape, dtype='uint8')

#Grayscaling
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

#Histogram
histogram = cv.calcHist([gray], [0], None, [256], [0,256])
plt.figure()
plt.title('Grayscale Histogram')
plt.xlabel('Bins')
plt.ylabel('# of pixels')
plt.plot(histogram)
plt.xlim([0,256])
plt.show()

#colour histogram
plt.figure()
plt.title('Colour Histogram')
plt.xlabel('Bins')
plt.ylabel('# of pixels')
colors = ('b', 'g', 'r')
for i, col in enumerate(colors):
    hist = cv.calcHist([img], [i], None, [256], [0,256])
    plt.plot(hist, color=col)
    plt.xlim([0,256])
plt.show()

cv.waitKey(0)