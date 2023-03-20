import cv2
import os
import handTrackingModule as htm

class FingerCounter:
    def __init__(self,  detectionCon=0.75, images_folder="CountFinger/images",draw=True):
        self.detector = htm.handDetector(detectionCon=detectionCon)
        if(draw):
            self.images_folder = images_folder
            self.overlayList = self.loadImages()
        self.draw = draw

    def loadImages(self):
        myList = os.listdir(self.images_folder)
        myList.sort()
        overlayList = []
        for imgPath in myList:
            image = cv2.imread(f"{self.images_folder}/{imgPath}")
            overlayList.append(image)
        return overlayList

    def fingerIsopen(self, fingerArray, flag=1):
        count = 0
        if(flag==1):
            if((fingerArray[0][2]>fingerArray[1][2])and 
                fingerArray[1][2]>fingerArray[2][2] and
                fingerArray[2][2]>fingerArray[3][2]):
                    count = 1
        else:
            if(fingerArray[0][1]>fingerArray[1][1]and 
                fingerArray[1][1]>fingerArray[2][1] and
                fingerArray[2][1]>fingerArray[3][1]):
                    count = 1
        return count

    def countFingers(self,img):
        
        img = self.detector.findHands(img)
        imgNumber = 0
        lmlist = self.detector.findPosition(img, draw=False)
        if (len(lmlist)!=0):
            indexFinger=[lmlist[5],lmlist[6],lmlist[7],lmlist[8]]
            middleFinger=[lmlist[9],lmlist[10],lmlist[11],lmlist[12]]
            ringFinger=[lmlist[13],lmlist[14],lmlist[15],lmlist[16]]
            pinkyFinger=[lmlist[17],lmlist[18],lmlist[19],lmlist[20]]
            thumb=[lmlist[1],lmlist[2],lmlist[3],lmlist[4]]
            count = [self.fingerIsopen(indexFinger), self.fingerIsopen(middleFinger),
                     self.fingerIsopen(ringFinger), self.fingerIsopen(pinkyFinger), self.fingerIsopen(thumb, 0)]
            imgNumber = sum(count)
        if(self.draw):
            h,w,c = self.overlayList[imgNumber].shape
            img[0:h, 0:w] = self.overlayList[imgNumber]
        
        return imgNumber,img
        
        
        

def main():
    
    cap = cv2.VideoCapture(0)
    Counter=FingerCounter(draw=False)
    while True:
        success, img = cap.read()
        img=cv2.flip(img,1)
        fingerCount,img=Counter.countFingers(img)
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()