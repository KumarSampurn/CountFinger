import cv2
import time
import os
import handTrackingModule as htm
import math

############################
wCam, hCam = 640, 480
############################




def fingerIsopen(fingerArray, flag=1):
    count=0
    if(flag==1):
        if((fingerArray[0][2]>fingerArray[1][2])and 
            fingerArray[1][2]>fingerArray[2][2] and
            fingerArray[2][2]>fingerArray[3][2]):
                count=1
    else:
        if(fingerArray[0][1]>fingerArray[1][1]and 
            fingerArray[1][1]>fingerArray[2][1] and
            fingerArray[2][1]>fingerArray[3][1]):
                count=1
    return count

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
    
    imgNumber=0
    #distance between points
    lmlist= detector.findPosition(img, draw=False)
    if (len(lmlist)!=0):
        indexFinger=[lmlist[5],lmlist[6],lmlist[7],lmlist[8]]
        middleFinger=[lmlist[9],lmlist[10],lmlist[11],lmlist[12]]
        ringFinger=[lmlist[13],lmlist[14],lmlist[15],lmlist[16]]
        pinkyFinger=[lmlist[17],lmlist[18],lmlist[19],lmlist[20]]
        thumb=[lmlist[1],lmlist[2],lmlist[3],lmlist[4]]
        print(thumb[0][1])
        
        count =[fingerIsopen(indexFinger), fingerIsopen(middleFinger),fingerIsopen(ringFinger),
                fingerIsopen(pinkyFinger),fingerIsopen(thumb,0)]
        print(count)
        imgNumber=sum(count)
    
    
    h,w,c=overlayList[imgNumber].shape
    img[0:h,0:w]=overlayList[imgNumber]
    
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    
    
    cv2.putText(img, str((int)(fps)), (400, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
