from imutils import contours
import numpy as np
import imutils
import matplotlib.pyplot as plt
import cv2
from easyocr import Reader

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'


FIRST_NUMBER = {
    "3": "American Express",
    "4": "Visa",
    "5": "MasterCard",
    "6": "Discover Card"
}


#/////////////////////////////////////////////////////////////////////
#        Create Ref digit images
#/////////////////////////////////////////////////////////////////////

ref = cv2.imread('./OCR-A_ref_font.bmp')

ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
ref = cv2.threshold(ref, 200, 60, cv2.THRESH_BINARY_INV)[1]

refCnts = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL,
                           cv2.CHAIN_APPROX_SIMPLE)
refCnts = imutils.grab_contours(refCnts)
refCnts = contours.sort_contours(refCnts, method="left-to-right")[0]
digits = {}

for (i, c) in enumerate(refCnts):
    # compute the bounding box for the digit, extract it, and resize
    # it to a fixed size
    (x, y, w, h) = cv2.boundingRect(c)
    roi = ref[y:y + h, x:x + w]
    roi = cv2.resize(roi, (57, 88))
    # update the digits dictionary, mapping the digit name to the ROI
    digits[i] = roi


#/////////////////////////////////////////////////////////////////////



image = cv2.imread('./images/6.jpg')

# ROTATE
image = cv2.rotate(image, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)

image_res = imutils.resize(image, width=300)
gray = cv2.cvtColor(image_res, cv2.COLOR_BGR2GRAY)
gray = cv2.threshold(gray, 100, 80, cv2.THRESH_BINARY)[1]
plt.imshow(gray,cmap='gray')
# plt.show()
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)

gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
gradX = np.absolute(gradX)
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
gradX = gradX.astype("uint8")

gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)

#//////////////////// CNTS ///////////////////////////


cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
locs = []

# get BB
for (i, c) in enumerate(cnts):
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w / float(h)
    if ar > 2.5 and ar < 4.0:
        if (w > 40 and w < 55) and (h > 10 and h < 20):
            locs.append((x, y, w, h))

# for BB in locs:
#     cv2.rectangle(image_res, BB, (0, 255, 0), 2)

reader = Reader(['en'], -1)
readerRes = []

locs = sorted(locs, key=lambda x:x[0])
output = []

for (i, (gX, gY, gW, gH)) in enumerate(locs):
    # initialize the list of group digits
    groupOutput = []
    # extract the group ROI of 4 digits from the grayscale image,
    # then apply thresholding to segment the digits from the
    # background of the credit card
    group = gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
    group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # detect the contours of each individual digit in the group,
    # then sort the digit contours from left to right
    digitCnts = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digitCnts = imutils.grab_contours(digitCnts)
    digitCnts = contours.sort_contours(digitCnts, method="left-to-right")[0]
    # loop over the digit contours
    testGrp = constant= cv2.copyMakeBorder(group,40,40,40,40,cv2.BORDER_CONSTANT)
    readerRes.append(reader.readtext(testGrp, allowlist ='0123456789'))
    plt.imshow(testGrp)
    plt.show()

    for c in digitCnts:
        # compute the bounding box of the individual digit, extract
        # the digit, and resize it to have the same fixed size as
        # the reference OCR-A images
        (x, y, w, h) = cv2.boundingRect(c)
        roi = group[y:y + h, x:x + w]
        roi = cv2.resize(roi, (57, 88))
        # initialize a list of template matching scores


        scores = []

        # loop over the reference digit name and digit ROI
        for (digit, digitROI) in digits.items():
            # apply correlation-based template matching, take the
            # score, and update the scores list
            result = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF)
            (_, score, _, _) = cv2.minMaxLoc(result)
            scores.append(score)
        # the classification for the digit ROI will be the reference
        # digit name with the *largest* template matching score
        groupOutput.append(str(np.argmax(scores)))
    cv2.rectangle(image_res, (gX - 5, gY - 5), (gX + gW + 5, gY + gH + 5), (0, 0, 255), 2)
    # cv2.putText(image_res, "".join(groupOutput), (gX, gY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    cv2.putText(image_res, "".join(readerRes[i][0][1]), (gX, gY - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)

for i in readerRes:
    print(i[0][1])
# print(readerRes)
plt.imshow(image_res)
plt.show()

