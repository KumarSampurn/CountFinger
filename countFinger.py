import cv2
import time
import os
import math
import handTrackingModule as htm

class FingerCounter:
    def __init__(self, wCam=640, hCam=480, detectionCon=0.75, images_folder="CountFinger/images"):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, wCam)
        self.cap.set(4, hCam)
        self.detector = htm.handDetector(detectionCon=detectionCon)
        self.images_folder = images_folder
        self.overlayList = self.load_images()

    def load_images(self):
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

    def count_fingers(self):
        success, img = self.cap.read()
        img = cv2.flip(img, 1)
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
        h,w,c = self.overlayList[imgNumber].shape
        img[0:h, 0:w] = self.overlayList[imgNumber]
        
        
        cv2.imshow('Image', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.cap.release()
            cv2.destroyAllWindows()
            return imgNumber

def main():
    finger_counter = FingerCounter()
    while True:
        finger_count = finger_counter.count_fingers()
        if finger_count is not None:
            print(f"Finger count: {finger_count}")
            break


if __name__ == "__main__":
    main()