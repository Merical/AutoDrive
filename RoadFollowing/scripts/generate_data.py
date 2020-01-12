import numpy as np
import os
from uuid import uuid1
import cv2
from imutils.video import WebcamVideoStream

def xy_image_name(x, y):
    '''
    Generate the captured image name
    '''
    return "xy_{0:03d}_{1:03d}_{2}.jpg".format(x, y, uuid1())

# mouse callback function
def onMouse(event,x,y,flags,param):
    global frame
    global src
    global x_coord
    global y_coord

    if event == cv2.EVENT_MOUSEMOVE:
        x_coord = x
        y_coord = y

    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.imwrite(os.path.join(dataset_dir, xy_image_name(x_coord, y_coord)), src)
        print("LCH: Image captured")

# frame = cv2.circle(frame, (x, y), 8, (0, 255, 0), 3)
        # frame = cv2.circle(frame, (112, 224), 8, (0, 0,255), 3)
        # frame = cv2.line(frame, (x,y), (112,224), (255,0,0), 3)

dataset_dir = "../dataset"
CAMERA_IP_ADDR = "rtsp://192.168.10.10:554/user=admin&password=&channel=1&stream=0.sdp?"
cap = WebcamVideoStream(CAMERA_IP_ADDR).start()
# frame = cap.read()
# frame = cv2.resize(frame, (224, 224))
cv2.namedWindow('image')
cv2.setMouseCallback('image', onMouse)
x_coord = 112
y_coord = 224

count = 0
while True:
    frame = cap.read()
    frame = cv2.resize(frame, (224, 224))
    src = frame.copy()
    frame = cv2.circle(frame, (x_coord, y_coord), 8, (0, 255, 0), 3)
    frame = cv2.circle(frame, (112, 224), 8, (0, 0,255), 3)
    frame = cv2.line(frame, (x_coord,y_coord), (112,224), (255,0,0), 3)
    key = cv2.waitKey(30)
    if key & 0xFF == 27:
        break
    # elif key == ord('s'):
    #     cv2.imwrite(os.path.join(dataset_dir, xy_image_name(x_coord, y_coord)), src)
    #     print("LCH: Image captured")

    cv2.imshow('image',frame)
cv2.destroyAllWindows()