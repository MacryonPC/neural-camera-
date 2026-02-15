import cv2 
import mediapipe as mp
import numpy as np
import time








face_cascade_db = cv2.CascadeClassifier(cv2.data.haarcascades+ "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+ "haarcascade_eye.xml")
upper_body = cv2.CascadeClassifier(cv2.data.haarcascades+ "haarcascade_upperbody.xml")
fron_smile = cv2.CascadeClassifier(cv2.data.haarcascades+ "haarcascade_smile.xml")
#(##########Разрешение экрана###########)#
wCam, hCam = 1200, 640
#(#####################)#
    
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FPS, 25)
camera.set(3, wCam)
camera.set(4, hCam)

#mpHands = mp.solutions.hands.Hands
#hands = mpHands.Hands()
#mpDraw = mp.solutions.drawing_utils





mpDraw = mp.solutions.drawing_utils
hand = mp.solutions.hands
hands = mp.solutions.hands.Hands(static_image_mode=False, 
    max_num_hands = 1,
    min_tracking_confidence=0.5,
    min_detection_confidence=0.5)



object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)
codec = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('Camera.avi', codec, 20.0, (640, 480))

def object_definition_camera():
    pTime = 0
    while camera.isOpened():
        
       
        success, img = camera.read()
    
        out.write(img)
        mask = object_detector.apply(img)
        _, mask = cv2.threshold(mask, 254, 123, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
   
        
      
        #detections = []
        #for cnt in contours:
            #area = cv2.contourArea(cnt)
            #if area > 1000:
            #cv2.drawContours(img, [cnt], -1, (0,255,0), 1)
                #x, y, w, h = cv2.boundingRect(cnt)
                #cv2.rectangle(img, (x,y), (x + w, y + h), (0,255,0), 2)
                #detections.append([x, y, w, h])
          
    
         
         
         
         
        imgRGB = cv2.resize(img, (0,0), None, 0.25, 0.25)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        
        
        upper = upper_body.detectMultiScale(img_gray, 1.1, 10)
        for (tx, ty, tw, th) in upper:
            cv2.rectangle(img, (tx,ty), (tx + tw, ty + th), (230,135,23), 3)
            cv2.putText(img, 'Upper',(tx,ty),  cv2.FONT_HERSHEY_PLAIN, 2 ,(230,255,354),2 )
            
            
            
        # Исправление синтаксических ошибок и цветов   
       faces = face_cascade_db.detectMultiScale(img_gray, 1.2, 15)
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (45, 255, 255), 2)  # исправлен цвет
    cv2.putText(img, 'faces', (x + 60, y - 5), cv2.FONT_HERSHEY_PLAIN, 2, (230, 255, 255), 2)
           cv2.putText(img, 'Id: ',(x + 1 ,y - 5),  cv2.FONT_HERSHEY_PLAIN, 2 ,(230,255,354),2 )
           # Область лица для детекции глаз
           img_gray_face = img_gray[y:y + h, x:x + w]  # исправлено имя переменной
    eyes = eye_cascade.detectMultiScale(img_gray_face, 1.2, 15)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(img, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (235, 125, 23), 3)
        cv2.putText(img, 'eye', (x + ex, y + ey), cv2.FONT_HERSHEY_PLAIN, 2, (230, 255, 255), 2)
               
               
               img_smail_fons = img_gray[y:y+h, x:x+w]
               fons = fron_smile.detectMultiScale(img_smail_fons, 1.2, 15)
               for (lx, ly, lw, lh) in fons:
                    cv2.rectangle(img, (x + lx, y + ly), (x + lx + lw, y + ly + lh), (125,235,23), 3)
                    cv2.putText(img, 'svaile',(x + lx, y + ly),  cv2.FONT_HERSHEY_PLAIN, 2 ,(230,255,354),2 )
                    
                    
                  
        
        
             
       
    
        #####################данные FPS   
        
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}',(40,70),cv2.FONT_HERSHEY_COMPLEX, 1,(0, 0, 0), 1)
        cv2.putText(img, "Id:", (39,150),cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 1)
        ############################
        result = hands.process(imgRGB)
   
    
    
        if result.multi_hand_landmarks:
            for id, lm in enumerate(result.multi_hand_landmarks[0].landmark):
                h, w, _=img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(img, (cx, cy), 2, (225, 245, 123),3)
                if id == 8:
                   cv2.circle(img, (cx, cy), 20, (232, 12, 123), cv2.FILLED) 
                if id == 12:
                   cv2.circle(img, (cx, cy), 15,(38, 24, 177), cv2.FILLED)
                if id == 16:
                   cv2.circle(img, (cx, cy), 15,(0, 158, 142), cv2.FILLED) 
                if id == 20:
                   cv2.circle(img, (cx, cy), 15,(254, 254, 0), cv2.FILLED) 
                if id == 4:
                   cv2.circle(img, (cx, cy), 15,(0, 255, 0), cv2.FILLED)
            mpDraw.draw_landmarks(img, result.multi_hand_landmarks[0], mp.solutions.hands.HAND_CONNECTIONS)   
        
    #results = hands.process(imgRGB)
    #if results.multi_hand_landmarks:
       # for handLms in results.multi_hand_landmarks:
           # mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            #for id, point in enumerate(handLms.landmark):
                #width, height, color = img.shape
               # width, height = int(point.x * height), int(point.y * width)
                #if id == 8:
                 #   cv2.circle(img, (width, height), 20,(12, 0, 455), cv2.FILLED)
               # if id == 12:
                    #cv2.circle(img, (width, height), 20,(38, 24, 177), cv2.FILLED)
               # if id == 16:
                 #   cv2.circle(img, (width, height), 20,(0, 158, 142), cv2.FILLED) 
                #if id == 20:
                   # cv2.circle(img, (width, height), 20,(254, 254, 0), cv2.FILLED) 
                #if id == 4:
                    #cv2.circle(img, (width, height), 20,(0, 255, 0), cv2.FILLED)
        cv2.imshow('Mask', mask)           
        cv2.imshow("Camera",img)
        if cv2.waitKey(1)& 0xFF== ord('q'):
            break    
    
    out.release()   
    camera.release()
    cv2.destroyAllWindows()
        
    
def main():
    object_definition_camera()       
      
if __name__ == "__main__":
    main()
        
   
        
        
    
         
    

  
  
      


