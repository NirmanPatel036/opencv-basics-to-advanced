import cv2 as cv

'''Working of shape() Function in OpenCV
The dimensions of a given image like the height of the image, width of the image and number of channels in the image are called the shape of the image.
The shape of the image is stored in numpy.ndarray.
In order to find the shape of a given image, we make use of a function in OpenCV called shape() function.
The shape() function can provide the dimension of a given image.
The shape() function stores each of the dimension of the image like the height of the image, width of the image and number of channels in the image at different indices.
The height of the image is stored at the index 0.
The width of the image is stored at index 1.
The number of channels in the image is stored at index 2.

Example:
#using shape() function to get the dimensions of the image
dimensions = imageread.shape
#using the indices 0, 1 and 2 in shape function to get the height, width and number of channels in the image which are nothing but the dimensions
height = imageread.shape[0]
width = imageread.shape[1]
channels = imageread.shape[2]
#displaying the dimensions of the image as the output on the screen
print('The dimension of the input image is : ', dimensions)
print('The height of the input image is : ', height)
print('The width of the input image is : ', width)
print('The Number of Channels in the input image are : ', channels)
'''

def rescaleFrame(frame, scale):
	#works for Images, Videos, and Live Videos
	width = int(frame.shape[1]*scale)
	height = int(frame.shape[0]*scale)
	dimensions = (width, height)            #storing the new dimensions as a tuple

	return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)
	# visit: https://learnopencv.com/image-resizing-with-opencv/#resize-with-interpolation
	# for detailed explaination of resize function

#rescale an image
img = cv.imread('/Users/nirmanpatel36/Documents/OpenCV/Photos/cat.jpeg')
resized_image = rescaleFrame(img, 0.75)
cv.imshow('Image', img)
cv.imshow('Image Resized', resized_image)

#rescale a video
capture = cv.VideoCapture('/Users/nirmanpatel36/Documents/OpenCV/Videos/IMG_8390.MOV') #variable to store the video file

'''Just FYI
def changeResolution(width, height):
	# works only for Live Videos
	capture.set(3, width)
	capture.set(4, height)
	# refer: https://docs.opencv.org/3.4/d8/dfe/classcv_1_1VideoCapture.html#a473055e77dd7faa4d26d686226b292c1
	# for detailed explaination on set function'''

#This starts an infinite loop which will continuously read frames from the video until a break condition is met.
while True:
	isTrue, frame = capture.read()           #method for reading video files or capturing data from decode and returns the just grabbed frame

	frame_resize = rescaleFrame(frame, 0.75) #rescalling each frame after reading it
	cv.imshow('Video', frame)
	cv.imshow('Video Resized', frame_resize)

	if cv.waitKey(20) & 0xFF == ord('d'):    #it waits for minimum 20 milliseconds after the video starts and if the letter d is pressed,
		break                                #it breaks out of the loop

capture.release()                            #release the capture device

cv.destroyAllWindows()