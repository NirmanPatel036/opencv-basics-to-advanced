import cv2 as cv

capture = cv.VideoCapture('/Users/nirmanpatel36/Documents/OpenCV/Videos/IMG_8390.MOV') #variable to store the video file

while True:                                 #This starts an infinite loop which will continuously read frames from the video until a break condition is met.
	isTrue, frame = capture.read()          #grab the video frame by frame using read function
	cv.imshow('Video', frame)

	if cv.waitKey(20) & 0xFF == ord('d'):   #it waits for minimum 20 milliseconds after the video starts and if the letter d is pressed, it breaks out of the loop
		break

capture.release()                           #release the capture device

cv.destroyAllWindows()