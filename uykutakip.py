import cv2
import numpy as np
import dlib
from imutils import face_utils
import time
import pygame

cap=cv2.VideoCapture(0)
detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") # önceden eğitilmiş yüz datasını eklemeyi unutmayın. aynı isim aratmasıyla indirebilirsin.
sleep=0
active=0   
status=""
color=(0,0,0)
pygame.mixer.init()
pygame.mixer.music.load("ses.mp3")  #kendinize ait ses dosyasını eklemeyi unutmayın

def compute(ptA,ptB):
    return np.linalg.norm(ptA - ptB)

def blinked(a,b,c,d,e,f):
    up=compute(b,d)+compute(c,e)
    down=compute(a,f)
    ratio=up/(2.0*down)

    if ratio>0.25:
        return 2
    elif ratio>0.21 and ratio<=0.25:
        return 1
    else:
        return 0

while True:
    _,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    faces=detector(gray)
    for face in faces:
        x1,y1,x2,y2=face.left(),face.top(),face.right(),face.bottom()

        face_frame=frame.copy()
        cv2.rectangle(face_frame,(x1,y1),(x2,y2),(255,255,255),2)

        landmarks=predictor(gray,face)
        landmarks=face_utils.shape_to_np(landmarks)

        left_blink=blinked(landmarks[36],landmarks[37],landmarks[38],landmarks[41],landmarks[40],landmarks[39])
        right_blink=blinked(landmarks[42],landmarks[43],landmarks[44],landmarks[47],landmarks[46],landmarks[45])

        if left_blink==0 or right_blink==0:
            sleep+=1
            active=0
            if sleep>3:
                status="Uyuyor!"
                color=(255,0,0)
                if sleep == 4:
                    pygame.mixer.music.play(-1)  
        else:
            sleep=0
            active+=1
            if active>6:
                status="iyi uyumus "
                color=(255,255,0)
                pygame.mixer.music.stop()  
        cv2.putText(face_frame,status,(100,100),cv2.FONT_HERSHEY_SIMPLEX,1.2,color,3)

        for n in range(0,68):
            x,y=landmarks[n]
            cv2.circle(face_frame,(x,y),1,(255,255,255),-1)
        cv2.imshow("tespit et",face_frame)
    cv2.imshow("ana kamera",frame)

    if cv2.waitKey(1)&0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()

# en son düzenlediğin hali ibrahim
