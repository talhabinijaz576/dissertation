import os
import inspect
import cv2
import numpy as np
from copy import deepcopy
import time

from ingestor_recog import LFWIngestor, CACDIngestor
from violajones import ViolaJonesDetector
from hog import HogDetector
from landmark_detectors import DlibDetector
import face_recognition
from base import BasicRecognizer
from utils import FaceVerification

import dlib




class ArcfaceRecognizer(BasicRecognizer):
    
    algorithm = "ArcFace"
    
    def __init__(self, face_detector, landmark_detector, dataset, tolerance = 0.6):
        
        self.face_detector = face_detector
        self.landmark_detector = landmark_detector
        self.dataset = dataset
        self.tolerance = tolerance

    
    def convert_boxes_to_array(self, faces):
        
        # (top, right, bottom, left)
    
        func1 = lambda box: [box[0][1], box[1][0], box[1][1], box[0][0]]
        face_boxes = [face.face_box for face in faces]
        face_boxes = list(map(func1,  face_boxes))
        face_boxes = np.array(face_boxes)
        
        return face_boxes
    
    
    def calculate_batch_embeddings(self, faces):
        
        embeddings = [self.calculate_face_embeddings(face) for face in faces]
        embeddings = np.vstack(embeddings)
        
        return embeddings
        
        
    def calculate_face_embeddings(self, face):
        
        image = face.face_image
        y_dim, x_dim, _ = image.shape
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = [[0, x_dim, y_dim, 0]]
        encodings = face_recognition.face_encodings(image_rgb, boxes)
        encodings = np.array(encodings).reshape((-1, 128))
        
        return encodings
        

    def calculate_face_distance(self, face, face_encodings):
    
        if len(face_encodings) == 0:
            return np.empty((0))
    
        distances = np.linalg.norm(face_encodings - face, axis=1)
        
        return distances
    
    
    def verify_faces(self, face1, face2):
    
        distance = self.calculate_face_distance(face1, face2)
        is_verified = (distance < self.tolerance)
         
        return distance, is_verified
    
    
    
    # def verify(self, img1, img2):
        
    #     face_detections1 = self.face_detector.detect(img1)
    #     faces1 = self.landmark_detector.detect(face_detections1)
    #     encodings1 = self.calculate_face_embeddings(faces1[0])
        
        
    #     start_time = time.time()
        
    #     face_detections2 = self.face_detector.detect(img2)
    #     faces2 = self.landmark_detector.detect(face_detections2)
    #     encodings2 = self.calculate_face_embeddings(faces2[0])
        
        
    #     distances, is_verified = self.verify_faces(encodings1, encodings2)
    #     min_distance = np.min(distances)
        
    #     time_taken = time.time() - start_time
        
    #     verification = FaceVerification(img1, faces1, img2, faces2, is_verified,  min_distance, self.tolerance, time_taken)
        
    #     return verification
        
    
    
    





# root_folder = "C:\\Users\\talha ijaz\\Documents\\thesis"
# hog = HogDetector()
# dlib_ = DlibDetector(expand_dims = (0.0, 0.0))
# recognizer = ArcfaceRecognizer(face_detector = hog, landmark_detector = dlib_, dataset = None, tolerance = 0.6)

# ing1 = CACDIngestor(os.path.join(root_folder, "CACD2000"))
# #ing1 = LFWIngestor(os.path.join(root_folder, "lfw"))




# l = []
# for _ in range(100):

    
#     try:
#         img1, img2, is_matching = ing1.get_pair()
#         verification = recognizer.verify(img1, img2)
#         success = is_matching == verification.verified
#         l.append(success)
#         print(success, verification.verified, verification.distance, verification.time_taken)
#         #verification.show()
#     except:
#         continue
        
    
# print()
# print(np.mean(l))
    


















