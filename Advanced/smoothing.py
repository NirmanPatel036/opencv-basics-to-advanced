import cv2 as cv

img = cv.imread('/Users/nirmanpatel36/Documents/OpenCV/Photos/Cat.jpeg')
cv.imshow('Cat', img)

#Averaging (the average of all the pixels under the kernel)
average = cv.blur(img, (3,3)) #the second argument represents the kernel size, increase the kernel size to increase the intensity
cv.imshow('Average Blur', average)

#Gaussian Blur (average of all the pixel weights under the kernel)
gaussian = cv.GaussianBlur(img, (3,3), 0) #the third argument represents the standard deviation(weight on the pixels)
cv.imshow('Gaussian Blur', gaussian)

#Median Blur (the median of all the pixels under the kernel)
median = cv.medianBlur(img, 3) #the second argument represents the kernel size
cv.imshow('Median Blur', median)

#Bilateral Blur (the weighted average of all the pixels under the kernel, it retains the edges)
bilateral = cv.bilateralFilter(img, 5, 15, 15) #the second is the kernel diameter, the third argument represents
                                               #the sigmaColour of the kernel, and fourth argument represents the sigmaSpace
cv.imshow('Bilateral Blur', bilateral)

cv.waitKey(0)