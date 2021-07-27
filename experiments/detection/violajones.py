import numpy as np
from utils import Image, FaceDetection
from base import BasicFaceDetector
import cv2



class ViolaJoneDetector(BasicFaceDetector):
    
    def __init__(self, filepath):
        self.filepath = filepath
        self.detector = self.load_detector()
        super().__init__()
        
        
    def load_detector(self):
        detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        return detector
    
    
    def detect(self, image):
        
        image_gray = cv2.cvtColor(image.image, cv2.COLOR_BGR2GRAY)

        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        detected_faces = face_cascade.detectMultiScale(image_gray )
        
        detection = FaceDetection(image, detected_faces)
        
        return detection
        
        
        


image = Image('C:\\Users\\talha ijaz\\Documents\\thesis\\fddb\\pics\\2002\\07\\24\\big\\img_586.jpg',
              {'n': 1, 'faces': [{'box': [(105, 65), (247, 274)]}]})


grayscale_image = cv2.cvtColor(image.image, cv2.COLOR_BGR2GRAY)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
detected_faces = face_cascade.detectMultiScale(grayscale_image)


for (column, row, width, height) in detected_faces:
    cv2.rectangle(
        image.annotated_image,
        (column, row),
        (column + width, row + height),
        (0, 255, 0),
        2
    )
    
image.show()