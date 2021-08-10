import numpy as np
from utils import Image, FaceDetection, Face
from base import BasicFaceDetector, BasicLandmarkDetector
import dlib
import cv2
from copy import deepcopy

from landmark_detectors import DlibDetector
from hog import HogDetector
import time




face_detector = HogDetector()
#face_detector = ViolaJonesDetector()
#face_detector = YoloDetector()

landmark_detector = DlibDetector(expand_dims = (0, 0))



vid = cv2.VideoCapture(0)

while(True):
    
    ret, frame = vid.read()
    
    img = Image(None, {}, frame=frame)
    face_detections = face_detector.detect(img)
    faces = landmark_detector.detect(face_detections)
    
    image = deepcopy(img.image)
    
    for face in faces:    
        
        image = cv2.rectangle(image, face.face_box[0], face.face_box[1], (255,0,0), 4)  
    
        for x, y in face.landmarks:
            x, y = int(x), int(y)
            image = cv2.circle(image, (x, y), radius=4, color=(0, 0, 255), thickness=-1)
    
    #for x, y in info["landmarks"]:
    #    x, y = int(x), int(y)
    #    annotated_image = cv2.circle(image, (x, y), radius=3, color=(0, 0, 255), thickness=-1)
    
    cv2.imshow('frame', image)
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
vid.release()
cv2.destroyAllWindows()