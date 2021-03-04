import cv2
import numpy as np
import imutils
# from easyocr import Reader
# reader = Reader(['en'], -1)


def nothing(x):
    frame = cv2.imread('images/osobna1.jpg')
    frame = imutils.resize(frame, width=600)
    frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)

    # get current positions of the trackbars
    ilowH = cv2.getTrackbarPos('lowH', 'image')
    ihighH = cv2.getTrackbarPos('highH', 'image')
    ilowS = cv2.getTrackbarPos('lowS', 'image')
    ihighS = cv2.getTrackbarPos('highS', 'image')
    ilowV = cv2.getTrackbarPos('lowV', 'image')
    ihighV = cv2.getTrackbarPos('highV', 'image')

    # convert color to hsv because it is easy to track colors in this color model
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array([ilowH, ilowS, ilowV])
    higher_hsv = np.array([ihighH, ihighS, ihighV])
    # Apply the cv2.inrange method to create a mask
    mask = cv2.inRange(hsv, lower_hsv, higher_hsv)
    # Apply the mask on the image to extract the original color
    frame = cv2.bitwise_and(frame, frame, mask=mask)
    cv2.imshow('image', frame)
    # print(reader.readtext(frame))

# Open the camera

# Create a window
cv2.namedWindow('image')

# create trackbars for color change
cv2.createTrackbar('lowH', 'image', 0, 179, nothing)
cv2.createTrackbar('highH', 'image', 179, 179, nothing)

cv2.createTrackbar('lowS', 'image', 0, 255, nothing)
cv2.createTrackbar('highS', 'image', 255, 255, nothing)

cv2.createTrackbar('lowV', 'image', 0, 255, nothing)
cv2.createTrackbar('highV', 'image', 80, 255, nothing)

nothing('asd')

cv2.waitKey(0)
cv2.destroyAllWindows()