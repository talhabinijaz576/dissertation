import numpy as np
from utils import Image, FaceDetection, Face
from base import BasicFaceDetector, BasicLandmarkDetector
import dlib
import cv2
import os
from copy import deepcopy
from utils import select_relevant_landmarks

from mtcnn import MTCNN
import time



class MCTNNDetector(BasicLandmarkDetector):
  
    
    def __init__(self, expand_dims = (0, 0)):
        
        self.detector = self.load_detector()
        self.expand_dims = expand_dims
        super().__init__()
        
        
    def load_detector(self):
        detector = MTCNN()
        return detector
                    
    
    def detect(self, face_detection):
        
        image = face_detection.image.image
        faces = []
        
        for face_box in face_detection.detections:
        
            start_time = time.time()
            
            cropped_image, shift = self.get_cropped_face(image, face_box) 
            detections = self.detector.detect_faces(cropped_image)   
            
            if(len(detections) > 0):
                
                detection = detections[0]                  
                landmarks = [
                    detection["keypoints"]["left_eye"],
                    detection["keypoints"]["right_eye"],
                    detection["keypoints"]["nose"],
                    detection["keypoints"]["mouth_left"],
                    detection["keypoints"]["mouth_right"],
                ]            
                landmarks = np.array(landmarks)      
                
                time_taken = time.time() - start_time   
                
                expand_x, expand_y = shift
                [ (left, top), (right, bottom) ] = face_box
                face_box = [ (left - expand_x, top - expand_y), (right + expand_x, bottom + expand_y) ]
                
                face = Face(image, cropped_image, face_box, landmarks, time_taken, "mtcnn")  
                faces.append(face)
            
        return faces
    
    
    
    
    
class DlibDetector(BasicLandmarkDetector):
  
    
    def __init__(self, expand_dims = (0, 0), target_dim = (256, 256)):
        
        self.detector = self.load_detector()
        self.expand_dims = expand_dims
        self.target_dim = target_dim
        super().__init__()
        
        
    def load_detector(self):
        detector = dlib.shape_predictor("shape_predictors/shape_predictor_68.dat")
        return detector
      

    def get_largest_face(self, face_boxes):
        
        if(len(face_boxes) < 2):
            return face_boxes
        
        sizes = list(map(lambda box: (box[1][0] - box[0][0])*(box[1][1] - box[0][1]) , face_boxes))
        argmax = np.argmax(sizes)
        output_boxes = [face_boxes[argmax]]
        #print("reducing boxes: ", sizes, argmax)
        
        return output_boxes
    
    def detect(self, face_detection, testing = False):
        
        image = deepcopy(face_detection.image.image)
        faces = []
        
        face_detections = self.get_largest_face(face_detection.detections)
        
        for face_box in face_detections:

            cropped_image, shift = self.get_cropped_face(image, face_box)             
            start_time = time.time()

            [ (left, top), (right, bottom) ] = face_box
            
            gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
            
            
            box = dlib.rectangle(left=0, top=0, right=gray_image.shape[1], bottom=gray_image.shape[0])
            shape = self.detector(gray_image, box)
            cropped_landmarks = np.array([(point.x, point.y) for point in shape.parts()])
            cropped_landmarks = cropped_landmarks.astype(float)
            
            if(True or testing):
                cropped_landmarks = select_relevant_landmarks(cropped_landmarks)
            
            time_taken = time.time() - start_time   
            
            #box = dlib.rectangle(left=left, top=top, right=right, bottom=bottom)
            #shape = self.detector(gray_image, box)
            #landmarks = np.array([(point.x, point.y) for point in shape.parts()])
            #landmarks = select_relevant_landmarks(landmarks)
            
            expand_x, expand_y = shift
            [ (left, top), (right, bottom) ] = face_box
            face_box = [ (left - expand_x, top - expand_y), (right + expand_x, bottom + expand_y) ]
            
            face = Face(image, cropped_image, face_box, cropped_landmarks, time_taken,  "dlib", target_dim = self.target_dim)  
            faces.append(face)
  
            
        return faces
            
    
    
    
#image = Image('C:\\Users\\talha ijaz\\Documents\\thesis\\fddb\\pics\\2002\\07\\24\\big\\img_586.jpg',
#              {'n': 1, 'faces': [{'box': [(105, 65), (247, 274)]}]})



#hd = HogDetector()
#detection = hd.detect(image)
#detection.show()
        
        
        



    