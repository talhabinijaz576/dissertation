import numpy as np
import pandas as pd
import random
import utils
import cv2
from copy import deepcopy
from utils import Image, FaceVerification
import time
        


class BasicIngestor:
    
    def __init__(self):
        self.random_order = self.get_random_permutation()
    
    
    def get_random_permutation(self):
        
        permutations = np.random.permutation(len(self.files))
        self.current_permutation_index = 0
        
        return permutations
    
    
    def get_image(self, i=None, filename=None):
        
        if(i==None):
            face_filepath = filename
        else:
            face_filepath = self.files[i]
        
        info = self.labels[face_filepath]
        self.current_permutation_index +=1
        image = utils.Image(face_filepath, info)
        
        return image
        
        
        
    def get_random_image(self):
        
        if(self.current_permutation_index >= len(self.files)):
            self.random_order = self.get_random_permutation()
        
        i = self.random_order[self.current_permutation_index]
        face_filepath = self.files[i]
        info = self.labels[face_filepath]
        self.current_permutation_index +=1
        image = utils.Image(face_filepath, info)
        
        return image
    
    
    def get_random_images(self, n=1):
        images = [self.get_random_image() for i in range(n)]
        return images
    
    
    
class BasicFaceDetector:
    
    def __init__(self):
        pass
    
    
    
class BasicLandmarkDetector:
    
    def __init__(self):
        pass  
    
    
    def get_cropped_face(self, image, face_box):
        

        
        [(left, top), (right, bottom)] = face_box  
        
        left = max(left, 0)
        right = max(right, 0)
        top = max(top, 0)
        bottom = max(bottom, 0)
        
        expand_x = abs(int( 0.5 * self.expand_dims[0] * (right - left ) ))
        expand_y = abs(int( 0.5 * self.expand_dims[1] * (top - bottom ) ))
        
        shift = (expand_x, expand_y)
        
        #print(left, right, top, bottom)
        #print(expand_x, expand_y)
        
        left = int(left - expand_x )
        top =int( top - expand_y )
        right = int(right + expand_x )
        bottom = int(bottom + expand_y )
        

        cropped_image = deepcopy(image)[top: bottom, left: right]
        
        
        return cropped_image, shift
    

        
        
        
class BasicFaceRecognitonIngestor:
    
    
    def __init__(self):
        pass
        
    
    def get_pair(self, matching = None, probabilty_matching = 0.5):
        
        random_is_matching = np.random.random() > probabilty_matching
        is_matching = matching == True or random_is_matching
        
        if(is_matching):
            img1, img2 = self.get_matching_images()
        
        else:
            img1, img2 = self.get_nonmatching_images()
            
        return img1, img2, is_matching
        
    
    
    def get_matching_images(self):
        
        person = np.random.choice(self.multiple_examples)
        i1, i2 = np.random.permutation(len(self.mapping[person]))[:2]
        file1 = self.mapping[person][i1]
        age, gender, race = self.get_annotation(file1)
        img1 = Image(file1, info = {"person" : person, "i" : i1, "age" : age, "gender" : gender, "race": race })
        
        file2 = self.mapping[person][i2]
        age, gender, race = self.get_annotation(file2)
        img2 = Image(file2, info = {"person" : person, "i" : i2, "age" : age, "gender" : gender, "race": race})
        
        return img1, img2
    
    
    def get_nonmatching_images(self):
        
        person1, person2 = np.random.choice(self.multiple_examples, 2, replace = False)
        
        i1 = np.random.permutation(len(self.mapping[person1]))[0]
        file1 = self.mapping[person1][i1]
        age, gender, race = self.get_annotation(file1)
        img1 = Image(file1, info = {"person" : person1, "i" : i1, "age" : age, "gender" : gender, "race": race})
        
        i2 = np.random.permutation(len(self.mapping[person2]))[0]
        file2 = self.mapping[person2][i2]
        age, gender, race = self.get_annotation(file2)
        img2 = Image(file2, info = {"person" : person2, "i":  i2, "age" : age, "gender" : gender, "race": race})
        
        return img1, img2
        
    
    
    
    
class BasicRecognizer:
    
    def __init__():
        pass
    
    
    def verify(self, img1, img2):
        
    
        
        face_detections1 = self.face_detector.detect(img1)
        faces1 = self.landmark_detector.detect(face_detections1)
        face1 = faces1[0]
        encodings1 = self.calculate_face_embeddings(face1)
        
        start_time = time.time()
        face_detections2 = self.face_detector.detect(img2)
        faces2 = self.landmark_detector.detect(face_detections2)
        face2 = faces2[0]
        encodings2 = self.calculate_face_embeddings(face2)
        
        distances, is_verified = self.verify_faces(encodings1, encodings2)
        distance = np.min(distances)
        
        time_taken = time.time() - start_time
        
        verification = FaceVerification(img1, faces1, img2, faces2, is_verified, distance, self.tolerance, time_taken, self.algorithm)
        
        return verification
    
    
        
    def show_face(self, faces, title=None, wait=True):
        
        if(len(faces) == 0):
            return 
        
        if(title == None):
            title = f"detection {np.random.randint(0, 10000)}"
        
        image = deepcopy(faces[0].image)
        
        for face in faces:            
            image = cv2.rectangle(image, face.face_box[0], face.face_box[1], (255,0,0), 4)  
        
            for x, y in face.landmarks:
                x, y = int(x), int(y)
                image = cv2.circle(image, (x, y), radius=4, color=(0, 0, 255), thickness=-1)
        
        cv2.imshow(title, image)
        
        if(wait):
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        