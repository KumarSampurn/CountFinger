import cv2
import time
import os
import handTrackingModule as htm

############################
wCam, hCam = 640, 480
############################

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)


folderpath="CountFinger/images"
myList=os.listdir(folderpath)
myList.sort()


overlayList=[]
for imgPath in myList:
    image= cv2.imread(f"{folderpath}/{imgPath}")
    overlayList.append(image)
    

pTime = 0
cTime = 0

detector=htm.handDetector(detectionCon=0.75)


while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    
    img=detector.findHands(img)
    
    
    h,w,c=overlayList[0].shape
    img[0:h,0:w]=overlayList[0]
    
    
    
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    
    
    cv2.putText(img, str((int)(fps)), (400, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
