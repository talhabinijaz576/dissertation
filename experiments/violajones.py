import numpy as np
from utils import Image, FaceDetection
from base import BasicFaceDetector
import cv2
import os
import time



class ViolaJonesDetector(BasicFaceDetector):
    
    
    detectors = [
        "haarcascade_frontalface_default.xml",
        "haarcascade_frontalface_alt.xml",
        "haarcascade_frontalface_alt_tree.xml",
        "haarcascade_frontalface_alt2.xml",
        ]
    
    
    def __init__(self, detector_type = 0):
        self.detector = self.load_detector(detector_type)
        super().__init__()
        
        
    def load_detector(self, i=0):
        weight_path = os.path.join("haarcascades", self.detectors[i])
        detector = cv2.CascadeClassifier(weight_path)
        return detector
    
    
    def detect(self, image):
        
        
        
        image_gray = cv2.cvtColor(image.image, cv2.COLOR_BGR2GRAY)
        
        start_time = time.time()
        detections = self.detector.detectMultiScale(image_gray)
        time_taken = time.time() - start_time
        detections = list(map(lambda d: [ (d[0], d[1]), 
                                         (d[0]+d[2], d[1]+d[3] ) ], detections))
        
        
        detection = FaceDetection(image, detections, time_taken, alg = "VJ")
        
        return detection
        
        
        



    