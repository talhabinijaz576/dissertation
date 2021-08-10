import os
import inspect
import cv2
import numpy as np
from copy import deepcopy
import time


# from ingestor_recog import LFWIngestor, CACDIngestor
# from violajones import ViolaJonesDetector
from hog import HogDetector
from landmark_detectors import DlibDetector
from base import BasicRecognizer
from utils import FaceVerification


from facenet_pytorch import MTCNN, InceptionResnetV1

from utils_facenet import detect_face, extract_face

from PIL import Image
import torch




def extract(img, batch_boxes, image_size=160, margin=0):

    batch_mode = True
    if (not isinstance(img, (list, tuple)) and not (isinstance(img, np.ndarray) and len(img.shape) == 4) and not (isinstance(img, torch.Tensor) and len(img.shape) == 4)):
        img = [img]
        batch_boxes = [batch_boxes]
        batch_mode = False


    faces = []
    for im, box_im in zip(img, batch_boxes):

        faces_im = []
        for i, box in enumerate(box_im):
            face = extract_face(im, box, image_size, margin, None)
            face = fixed_image_standardization(face)
            faces_im.append(face)

        faces_im = torch.stack(faces_im)
        faces.append(faces_im)

    if not batch_mode:
        faces = faces[0]

    return faces


def fixed_image_standardization(image_tensor):
    processed_tensor = (image_tensor - 127.5) / 128.0
    return processed_tensor


class FacenetRecognizer(BasicRecognizer):
    
    algorithm = "Facenet/OpenFace"
    
    def __init__(self, face_detector, landmark_detector, dataset, tolerance = 0.6):
        
        self.face_detector = face_detector
        self.landmark_detector = landmark_detector
        self.dataset = dataset
        self.tolerance = tolerance
        #self.model = build_model("VGG-Face")
        self.model = InceptionResnetV1(pretrained='vggface2')
        state_dict = torch.load("models/vgg_facenet.pt")
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        self.mtcnn = MTCNN()
        
        ##self.model.load_weights("models/facenet_keras_weights.h5")

    
    def calculate_batch_embeddings(self, faces):
        
        processed_tensors = []
        for face in faces:
            image = face.image
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)        
            
            [(left, top), (right, bottom)] = face.face_box
            box = [left, top, right, bottom]
        
            batch_boxes = [box]
            processed_tensor = extract(image_rgb, batch_boxes)    
            processed_tensors.append(processed_tensor)
            
        
        final_tensor = torch.vstack(processed_tensors)
        #print(final_tensor.shape)
        embeddings = self.model(final_tensor)
        embeddings = embeddings.detach().numpy()
        
        return embeddings
    

    def calculate_face_embeddings(self, face):
        
        
        image = face.image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)         
    
        [(left, top), (right, bottom)] = face.face_box
        box = [left, top, right, bottom]
    
        batch_boxes = [box]
        processed_tensor = extract(image_rgb, batch_boxes)
        
        processed_tensor.unsqueeze(0)
        embeddings = self.model(processed_tensor)
        embeddings = embeddings.detach().numpy() 

        return embeddings
        

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
        
    #     #start_time = time.time()
        
        
    #     distances, is_verified = self.verify_faces(encodings1, encodings2)
    #     min_distance = np.min(distances)
        
    #     time_taken = time.time() - start_time
        
    #     verification = FaceVerification(img1, faces1, img2, faces2, is_verified, min_distance, self.tolerance, time_taken)
        
    #     return verification
        
    
    
    





# root_folder = "C:\\Users\\talha ijaz\\Documents\\thesis"
# ing1 = CACDIngestor(os.path.join(root_folder, "CACD2000"))
# ing1 = LFWIngestor(os.path.join(root_folder, "lfw"))

# face_detector = HogDetector()
# landmark_detector = DlibDetector(expand_dims = (0.0, 0.0), target_dim = (160, 160))
# recognizer = FacenetRecognizer(face_detector, landmark_detector, None, 1)


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


#mtcnn = MTCNN()
#resnet = InceptionResnetV1(pretrained='vggface2').eval()
#img = Image.open(img1.filepath)
#img_cropped = mtcnn(img)
# Or, if using for VGGFace2 classification
#resnet.classify = True
#img_probs = resnet(img_cropped.unsqueeze(0))



# face = faces1[0]
# image = face.face_image
# #image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
# input_shape_x, input_shape_y = functions.find_input_shape(model)


# img = functions.preprocess_face(img = face.face_image, 
#                                 target_size=(input_shape_y, input_shape_x), 
#                                 enforce_detection = True, 
#                                 detector_backend = 'opencv',
#                                 align = True)

# img = functions.normalize_input(img = img, normalization = "base")
 
# embedding = recognizer.model.predict(img)[0].tolist()







































l = []
for _ in range(0):

    
    try:
        img1, img2, is_matching = ing1.get_pair()
        verification = recognizer.verify(img1, img2)
        success = is_matching == verification.verified
        l.append(success)
        print(success, verification.score, verification.time_taken)
        #if(not success): verification.show()
    except:
        pass
    
    
#print()
#print(np.mean(l))
    


















