import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier('model\keras_model.h5', 'model\labels.txt')

offset = 20
imageSize = 300
lable = ['A','B','C']

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    
    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox']

        imgWhite = np.ones((imageSize,imageSize,3),np.uint8)*255
        imgCrop = img[y-offset : y+h+offset, x-offset : x+w+offset]
        
        aspectRatio = h/w
        if aspectRatio > 1:
            k = imageSize/h
            wCal = math.ceil(k*w)
            try:
                imgResize = cv2.resize(imgCrop, (wCal, imageSize))
                wGap = math.ceil((imageSize - wCal)/2)
                imgWhite[:, wGap:wCal+wGap] = imgResize
                pred, index = classifier.getPrediction(img)
                print(pred, index)
            except:
                pass
        else:
            k = imageSize/w
            hCal = math.ceil(k*h)
            try:
                imgResize = cv2.resize(imgCrop, (imageSize, hCal))
                hGap = math.ceil((imageSize - hCal)/2)
                imgWhite[hGap:hCal+hGap ,:] = imgResize
                pred, index = classifier.getPrediction(img)
                print(pred, index)
            except:
                pass

        cv2.imshow("White Image", imgWhite)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()