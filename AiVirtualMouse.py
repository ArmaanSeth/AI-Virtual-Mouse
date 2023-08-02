#! AIVirtualMouse\py38venv\Scripts\python.exe

import numpy as np
import cv2 as cv
import time
import autopy
import HandTrackingModule as htm

##########################
wCam,hCam=640,480
frameR=200
smoothening=5
##########################

ctime=0
ptime=0
plocX,plocY=0,0
clocX,clocY=0,0
wScr,hScr=autopy.screen.size()

cap=cv.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)

detector=htm.HandDetector(detectionCon=0.75)

while True:
    success,img=cap.read()
    img=cv.flip(img,1)

    image=detector.findHands(img)
    lmList,bbox=detector.findPosition(img)
    
    if len(lmList)!=0:
        # print(lmList)
        x1,y1=lmList[8][1:]
        x2,y2=lmList[12][1:]


        fingers=detector.fingersUp()
        cv.rectangle(img,(int(frameR/2),int(frameR/2)),(wCam-int(frameR/2),hCam-frameR),(255,0,0),2)
        
        # Moving Mode:if index finger is up only
        if fingers[1]==1 and fingers[2]==0:
            print("Moving Mode")

            #convert video coordinates to screen coordinates
            if x1<int(frameR/2):
                x1=int(frameR/2)
            elif x1>wCam-int(frameR/2):
                x1=wCam-int(frameR/2)
            if y1<int(frameR/2):
                y1=int(frameR/2)
            elif y1>hCam-frameR:
                y1=hCam-frameR
            x3=np.interp(x1,(int(frameR/2),wCam-int(frameR/2)),(0,wScr))
            y3=np.interp(y1,(int(frameR/2),hCam-frameR),(0,hScr))

            # Smoothing values
            clocX=plocX+(x3-plocX)/smoothening
            clocY=plocY+(y3-plocY)/smoothening

            #Move mouse
            autopy.mouse.move(clocX,clocY)
            cv.circle(img,(x1,y1),10,(255,0,150),-1)

            plocX,plocY=clocX,clocY

        # Select Mode:when both fingers are up
        if fingers[1]==1 and fingers[2]==1:
            print("Moving Mode")
            length,img,lineInfo=detector.findDistance(8,12,img)
            # print(length)
            if length<35:
                cv.circle(img,(lineInfo[4],lineInfo[5]),10,(0,255,0),-1)
                autopy.mouse.click()

    ctime=time.time()
    fps=1/(ctime-ptime)
    ptime=ctime
    cv.putText(img,f'FPS:{int(fps)}',(500,50),cv.FONT_HERSHEY_PLAIN,2,(0,0,255),2)

    cv.imshow("AI Virtual Mouse",img)
    if cv.waitKey(1) & 0xff==ord('q'):
        break
    
cap.release()
cv.destroyAllWindows()

