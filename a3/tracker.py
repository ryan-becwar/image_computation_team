import cv2 
import sys

cap = cv2.VideoCapture(sys.argv[1])
if not cap.isOpened():
	cap.open()

while (cap.isOpened()):
	ret, frame = cap.read()
	cv2.imshow('frame',frame)
	if cv2.waitKey(25) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()

