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
from keras_vggface.vggface import VGGFace
from keras_vggface import utils

import tensorflow as tf





class Senet50Recognizer(BasicRecognizer):
    
    algorithm = "VGG-Face2 (SENET-50)"
     
    
    def __init__(self, face_detector, landmark_detector, dataset, tolerance = 0.6):
        
        self.face_detector = face_detector
        self.landmark_detector = landmark_detector
        self.dataset = dataset
        self.tolerance = tolerance
        self.model = VGGFace(model='senet50', include_top=False)

    
    def convert_boxes_to_array(self, faces):
        
        # (top, right, bottom, left)
    
        func1 = lambda box: [box[0][1], box[1][0], box[1][1], box[0][0]]
        face_boxes = [face.face_box for face in faces]
        face_boxes = list(map(func1,  face_boxes))
        face_boxes = np.array(face_boxes)
        
        return face_boxes
    
    
    def calculate_batch_embeddings(self, faces):
        
        images = list(map(lambda face: face.face_image, faces))
        x = np.array(images).astype(float)
        x = utils.preprocess_input(x, version=1) # or version=2
        encodings = self.model.predict(x)
        encodings = encodings.reshape((-1, 2048))
        
        return encodings
        
    

    def calculate_face_embeddings(self, face):
        
        image = face.face_image
        #image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        x = image.astype(float)
        x = np.expand_dims(x, axis=0)
        x = utils.preprocess_input(x, version=1) # or version=2
        encodings = self.model.predict(x)
        encodings = encodings.reshape((1, -1))

        return encodings
        

    def calculate_face_distance(self, face, face_encodings):
    
        if len(face_encodings) == 0:
            return np.empty((0))
    
        distances = np.linalg.norm(face_encodings - face, axis=1)
        
        return distances
    
    
    def calculate_cosine_distance(self, source_representation, test_representation):
        
        a = np.matmul(np.transpose(source_representation), test_representation)
        b = np.sum(np.multiply(source_representation, source_representation))
        c = np.sum(np.multiply(test_representation, test_representation))  
        distance = 1 - (a / (np.sqrt(b) * np.sqrt(c)))
        
        return distance
    
         
    def l2_normalize(self, x, axis=-1, epsilon=1e-10):
        output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
        return output
         
    
    def calculate_euclidean_distance(self, source_representation, test_representation):
        
        euclidean_distance = source_representation - test_representation
        euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
        euclidean_distance = np.sqrt(euclidean_distance)
        #euclidean_distance = l2_normalize(euclidean_distance )
        
        return euclidean_distance
    
    
    def verify_faces(self, face1, face2):
    
        distance = self.calculate_face_distance(face1, face2)
        is_verified = (distance < self.tolerance)
         
        return distance, is_verified
    
    
    
    # def verify(self, img1, img2):
        
    
        
    #     face_detections1 = self.face_detector.detect(img1)
    #     faces1 = self.landmark_detector.detect(face_detections1)
    #     face1 = faces1[0]
    #     encodings1 = self.calculate_face_embeddings(face1)
        
    #     start_time = time.time()
    #     face_detections2 = self.face_detector.detect(img2)
    #     faces2 = self.landmark_detector.detect(face_detections2)
    #     face2 = faces2[0]
    #     encodings2 = self.calculate_face_embeddings(face2)
        
    #     distances, is_verified = self.verify_faces(encodings1, encodings2)
    #     min_distance = np.min(distances)
    #     score = min_distance
        
    #     time_taken = time.time() - start_time
        
    #     verification = FaceVerification(img1, faces1, img2, faces2, is_verified, min_distance, self.tolerance, time_taken)
        
    #     return verification
        
    
    
    





# root_folder = "C:\\Users\\talha ijaz\\Documents\\thesis"
# ing1 = CACDIngestor(os.path.join(root_folder, "CACD2000"))
# #ing1 = LFWIngestor(os.path.join(root_folder, "lfw"))




# face_detector = HogDetector()
# landmark_detector = DlibDetector(expand_dims = (0.0, 0.0), target_dim = (224, 224))

# recognizer = Senet50Recognizer(face_detector, landmark_detector, None, tolerance=185.0)


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
# print(np.mean(l))

# img1, img2, is_matching = ing1.get_pair()

# #model = VGGFace(model='senet50', include_top=False)

# face_detections1 = face_detector.detect(img1)
# faces1 = landmark_detector.detect(face_detections1)

# face_detections2 = face_detector.detect(img2)
# faces2 = landmark_detector.detect(face_detections2)


# encodings1 = recognizer.calculate_face_embeddings(faces1[0])
# encodings2 = recognizer.calculate_face_embeddings(faces2[0])

# distances, is_verified = recognizer.verify_faces(encodings1, encodings2)
# min_distance = np.min(distances)



# match_list = []
# nonmatch_list = []

# for _ in range(100):

#     img1, img2, is_matching = ing1.get_pair()
        
#     face_detections1 = face_detector.detect(img1)
#     faces1 = landmark_detector.detect(face_detections1)
    
#     face_detections2 = face_detector.detect(img2)
#     faces2 = landmark_detector.detect(face_detections2)
    
#     if(len(faces1) ==0 or len(faces2) == 0):
#         continue
    
#     encodings1 = recognizer.calculate_face_embeddings(faces1[0])
#     encodings2 = recognizer.calculate_face_embeddings(faces2[0])
    
    
#     distances, is_verified = recognizer.verify_faces(encodings1, encodings2)
    
#     if(is_matching):
#         match_list.append(distances[0])
#     else:
#         nonmatch_list.append(distances[0])
        
#     min_distance = np.min(distances)
    
    
#     print(distances, is_verified, is_matching)
    
# match_list.sort()
# nonmatch_list.sort()



    







































