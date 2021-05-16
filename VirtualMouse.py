import cv2
import numpy as np
import HandModule as htm
import time 
import autopy # to move mouse 


##############################################################
wCam, hCam = 640, 480
wScreen, hScreen = autopy.screen.size()
cTime, pTime = 0, 0
detector = htm.handDetector(maxHands= 1)
frameReduct = 100 # Frame to mouse the move reduction
smoothening = 7
prevX, curX, prevY, curY = 0, 0, 0, 0
##############################################################

cap = cv2.VideoCapture(0)
cap.set(3, wCam) # Width
cap.set(4, hCam) # height
while cap.isOpened():

    # 1. Find the hand landmarks 
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bBox = detector.findPosition(img)
    cv2.rectangle(img, (frameReduct, frameReduct), (wCam-frameReduct, hCam-frameReduct), (255, 0, 255), 2)
    if len(lmList) != 0:
    # 2. Get tip of index and middle finger 
    # Index finger : Mouse moves
    # Both finger together : Click 
        x1, y1 = lmList[8][1:]      # coordinates of finger
        x2, y2 = lmList[12][1:]
        # print(x1, y1, x2, y2)



    # 3. CHekc which fingers are up 
        fingers = detector.fingersUp()
        # print(fingers)


    # 4. Check if it's in moving mode
        if fingers[1] == 1 and fingers[2] == 0:
        # else convert coordinate (to make 640*480 resolution work on 1920*1080)
            x3 = np.interp(x1, (frameReduct, wCam-frameReduct), (0, wScreen))
            y3 = np.interp(y1, (frameReduct, hCam-frameReduct), (0, hScreen))

        # 5. Smoothen value 
            curX = prevX + (x3-prevX) / smoothening
            curY = prevY + (y3-prevY) / smoothening

            # 6. Move mouse 
            autopy.mouse.move(wScreen-curX, curY)
            cv2.circle(img, (x1, y1), 15, (255, 0, 0), cv2.FILLED)

            prevX, prevY = curX, curY
    # 7. Check if in clicking mode 
        # If distance is short : CLick
        if fingers[1] == 1 and fingers[2] == 1:
            length, img, data= detector.findDistance(8, 12, img)
            # print(length)
            if length < 40: 
                cv2.circle(img, (data[4], data[5]), 15, (0, 0, 255), cv2.FILLED) 
                autopy.mouse.click()

    # 9. FrameRate
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
    
    # 10. Display
    cv2.imshow("Window", img)    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
