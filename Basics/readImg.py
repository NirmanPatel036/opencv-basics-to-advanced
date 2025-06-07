import cv2 as cv

img = cv.imread('/Users/nirmanpatel36/Documents/OpenCV/Photos/Cat.jpeg') #takes in the path of the image and returns it as matrix pixels

'''Below function displays the image in a new window
it takes in 2 parameters: name of the window, and the file as the pixel matrix'''
cv.imshow('Cat', img)

cv.waitKey(0) #waits for a pressed key. If we pass in zero, it will wait for a key event infinitely