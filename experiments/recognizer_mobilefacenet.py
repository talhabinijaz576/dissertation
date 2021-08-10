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
from base import BasicRecognizer
from utils import FaceVerification
from skimage import transform
import random

from imageio import imread
import tensorflow as tf

from sklearn.preprocessing import normalize



class MobileFacenetRecognizer(BasicRecognizer):
     
    algorithm = "MobileFacenet"
    
    def __init__(self, face_detector, landmark_detector, dataset, tolerance = 0.6):
        
        self.face_detector = face_detector
        self.landmark_detector = landmark_detector
        self.dataset = dataset
        self.tolerance = tolerance
        
        model_file = "models/keras_mobilenet.h5"
        self.model = tf.keras.models.load_model(model_file, compile=False)
        
        # self.embedder = insightface.iresnet50(pretrained=True)
        # self.embedder.eval()
        
        # mean = [0.5] * 3
        # std = [0.5 * 256 / 255] * 3
        # self.preprocess = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean, std)
        # ])
        
        # data_transforms = {
        #     'train': transforms.Compose([
        #         #transforms.RandomHorizontalFlip(),
        #         transforms.ToTensor(),
        #         #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        #     ]),
        #     'val': transforms.Compose([
        #         transforms.ToTensor(),
        #         #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #     ]),
        # }
        # self.transformer = data_transforms['train']
        
        # scripted_model_file = r"C:\Users\talha ijaz\Downloads\mobilefacenet_scripted.pt"
        # model_file = r"C:\Users\talha ijaz\Downloads\mobilefacenet.pt"
        
        # model = MobileFaceNet()
        # #model = torch.jit.load(scripted_model_file)
        # state_dict = torch.load(model_file, torch.device("cpu"))
        # model.load_state_dict(state_dict)
        # model.eval()
        # self.model = model
    
    def convert_boxes_to_array(self, faces):
        
        # (top, right, bottom, left)
    
        func1 = lambda box: [box[0][1], box[1][0], box[1][1], box[0][0]]
        face_boxes = [face.face_box for face in faces]
        face_boxes = list(map(func1,  face_boxes))
        face_boxes = np.array(face_boxes)
        
        return face_boxes
    
    
    def calculate_batch_embeddings(self, faces):
        
        
        processed_images = []
        for face in faces:

            temp_path = "temp.jpg"
            image = face.face_image
            cv2.imwrite(temp_path, image)
            img = imread(temp_path)
            processed_images.append(img)
            
        img_array = np.array(processed_images)
        nimgs = (img_array - 127.5) * 0.0078125
        encodings = self.model.predict(nimgs)
        encodings = normalize(encodings)
        
        return encodings
        
    

    def calculate_face_embeddings(self, face):
        
        temp_path = "temp.jpg"
        image = face.face_image
        
        cv2.imwrite(temp_path, image)
        img = imread(temp_path)
        
        x = np.expand_dims(img, axis=0)
        nimgs = (x - 127.5) * 0.0078125
        
        encodings = self.model.predict(nimgs)
        encodings = normalize(encodings)
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
        
    #     time_taken = time.time() - start_time
        
    #     verification = FaceVerification(img1, faces1, img2, faces2, is_verified, min_distance, self.tolerance, time_taken)
        
    #     return verification
        
    
    
    





# root_folder = "C:\\Users\\talha ijaz\\Documents\\thesis"
# ing2 = CACDIngestor(os.path.join(root_folder, "CACD2000"))
# ing1 = LFWIngestor(os.path.join(root_folder, "lfw"))


# face_detector = HogDetector()
# landmark_detector = DlibDetector(expand_dims = (0.0, 0.0), target_dim = (112, 112))

# recognizer = MobileFacenetRecognizer(face_detector, landmark_detector, None, tolerance=1.25)


# random.seed(0)
# np.random.seed(0)

# l = []
# for _ in range(100):

    
#     try:
#         img1, img2, is_matching = ing1.get_pair()
#         verification = recognizer.verify(img1, img2)
#         success = is_matching == verification.verified
#         l.append(success)
#         print(success, verification.distance, verification.time_taken)
#         #if(not success): verification.show()
#     except:
#         pass
    
    
# print()
# print(np.mean(l))
    


# l = []
# for _ in range(100):

    
#     try:
#         img1, img2, is_matching = ing1.get_pair()
#         verification = recognizer.verify(img1, img2)
#         success = is_matching == verification.verified
#         l.append(success)
#         print(success, verification.verified, verification.score, verification.time_taken)
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

# for _ in range(0):
    
#     try:
#         img1, img2, is_matching = ing1.get_pair()
            
#         face_detections1 = face_detector.detect(img1)
#         faces1 = landmark_detector.detect(face_detections1)
        
#         face_detections2 = face_detector.detect(img2)
#         faces2 = landmark_detector.detect(face_detections2)
        
#         faces1[0].show_face(False)
#         faces2[0].show_face(True)
        
#         if(len(faces1) ==0 or len(faces2) == 0):
#             continue
        
#         encodings1 = recognizer.calculate_face_embeddings(faces1[0])
#         encodings2 = recognizer.calculate_face_embeddings(faces2[0])
#         #print(encodings1.shape, encodings2.shape)
        
        
#         distances, is_verified = recognizer.verify_faces(encodings1, encodings2)
        
#         if(is_matching):
#             match_list.append(distances[0])
#         else:
#             nonmatch_list.append(distances[0])
            
#         min_distance = np.min(distances)
        
        
#         print(distances, is_verified, is_matching)
        
        
#     except Exception as e:
#         print(e)
    
    
# match_list.sort()
# nonmatch_list.sort()

















































