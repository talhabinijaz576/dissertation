import numpy as np
from utils import Image, FaceDetection
from base import BasicFaceDetector
import dlib
import cv2
import os
import time



class HogDetector(BasicFaceDetector):
    
    
    
    def __init__(self):
        
        self.detector = self.load_detector()
        super().__init__()
        
        
    def load_detector(self):
        detector = cv2.HOGDescriptor()
        #detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        detector = dlib.get_frontal_face_detector()
        return detector
    
    
    def get_bounding_box(self, detection): 
        
        
        box = [ (detection.left(), detection.top()), 
                (detection.right(), detection.bottom()) ]
        
        return box
    
    def detect(self, image):
        
        image_rgb = cv2.cvtColor(image.image, cv2.COLOR_BGR2GRAY)
        
        start_time = time.time()
        detections = self.detector(image_rgb, 1)
        time_taken = time.time() - start_time
    
        detections = list(map(self.get_bounding_box, detections))
        
        
        detection = FaceDetection(image, detections, time_taken, alg = "HOG")
        
        return detection
    
    
#image = Image('C:\\Users\\talha ijaz\\Documents\\thesis\\fddb\\pics\\2002\\07\\24\\big\\img_586.jpg',
#              {'n': 1, 'faces': [{'box': [(105, 65), (247, 274)]}]})



#hd = HogDetector()
#detection = hd.detect(image)
#detection.show()
        
        
        



    