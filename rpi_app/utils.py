import cv2
from copy import deepcopy
import numpy as np
import imutils
import random
from skimage import transform
from pprint import pprint
from scipy.interpolate import griddata


class Image:
    
    
    def __init__(self, filepath, info, target_dim = (512, 512), frame=None):
        self.filepath = filepath
        self.info = info
        self.target_dim = target_dim
        if(frame is None):
            image = cv2.imread(filepath)
        else:
            image = frame
        #print(image.shape)        
        self.image = self.process_image(image, target_dim)      
        #print(image.shape)     
        self.annotated_image = self.get_annotated_image(self.image, self.info)

            
        
        
    
    
    def process_image(self, image, target_dim):
        
        image = self.pad_image(image, target = target_dim)
        image = self.resize_image(image, target = target_dim)
        
        return image
    
    
    def pad_image(self, image, target):
        try:
            height, width, channels = image.shape
        except:
            print("ERROR: ", self.filepath)
            raise Exception(f"{self.filepath} not found")
            
        ratio = height/width
        #print(ratio)
        target_ratio = target[0] / target[1]
        #print(target_ratio)
        ratio_difference = ratio - target_ratio     
        #print(ratio_difference)
        
        
        if (ratio_difference < -0.05):
        
            ratio_difference = np.abs(ratio_difference)
            pad = int( np.ceil( ratio_difference * width )//2 )
            #print(pad)
            pad_array = ( np.ones((pad, width, channels))*255 ).astype(np.uint8)
            image = np.vstack((pad_array, image, pad_array))
            
            if("faces" in self.info):
                
                for face in self.info["faces"]:
                    face["box"][0] = ( face["box"][0][0], face["box"][0][1] + pad )
                    face["box"][1] = ( face["box"][1][0], face["box"][1][1] + pad )
                    
            elif("landmarks" in self.info):
                #print("padding")
                self.info["landmarks"][:, 1] += pad
                
                
                    

            
        elif (ratio_difference > 0.05):
            

            ratio_difference = np.abs( (width/height) - (target[1] / target[0]) )
            pad = int( np.ceil( ratio_difference * height)//2 )
            #print(pad)
            pad_array = (np.ones((height, pad, channels))*255).astype(np.uint8)
            image = np.hstack((pad_array, image, pad_array))
            
            if("faces" in self.info):
            
                for face in self.info["faces"]:
                    face["box"][0] = ( face["box"][0][0] + pad, face["box"][0][1] )
                    face["box"][1] = ( face["box"][1][0] + pad, face["box"][1][1] )
                    
            elif("landmarks" in self.info):
                #print("padding")
                self.info["landmarks"][:, 0] += pad
        
        return image
    
    
    def resize_image(self, image, target):
        
        height, width, channels = image.shape
        ratio_h = target[0] / height
        ratio_w = target[1] / width
        
        
        if("faces" in self.info):
            
            for face in self.info["faces"]:
                face["box"][0] = ( int(face["box"][0][0] * ratio_h), int(face["box"][0][1] * ratio_w ))
                face["box"][1] = ( int(face["box"][1][0] * ratio_h), int(face["box"][1][1] * ratio_w))
            
        elif("landmarks" in self.info):
            
            self.info["landmarks"][:, 0] *= ratio_w
            self.info["landmarks"][:, 1] *= ratio_h
        
        image = cv2.resize(image, target[::-1])
        
        return image
        
    
    def get_annotated_image(self, image, info):
        image = deepcopy(image)
        
        if("faces" in info):
            for face in info.get("faces", []):
                image = cv2.rectangle(image, face["box"][0], face["box"][1], (0,0,0), 2)    
                
        elif("landmarks" in info):
            for x, y in info["landmarks"]:
                x, y = int(x), int(y)
                image = cv2.circle(image, (x, y), radius=3, color=(0, 0, 255), thickness=-1)
                
        return image
    
                
        
    def show(self, wait=True):
        
        title = "picture"
        if("person" in self.info):
            title = f"{self.info['person']} ({self.info['i']}), [{self.info['age'], self.info['gender'], self.info['race']}]"
            
        cv2.imshow(title, self.annotated_image)
        if(wait):
            k = cv2.waitKey(0)
            cv2.destroyAllWindows()


        
        
class FaceDetection:
    
    title = {"VJ" : "Viola Jones",
             "HOG" : "Hog Detector",
             "YOLO" : "YOLO v3 - tiny",
             "SSD" : "MobileNet SSD"}
    
    colors = {"VJ" : (0,0,255),
              "HOG" : (0,255,0),
              "YOLO" : (255, 0, 0),
              "SSD" : (0,128,255)}
    
    def __init__(self, image, faces_detected, time_taken, alg):
        self.alg = alg
        self.image = image
        self.detections = faces_detected
        self.time_taken = time_taken 
        
        
    def filter_largest_face(self):
        pass
        
        
    def show(self, show_truth=True, wait=True):
        
        if(show_truth):
            image = deepcopy(self.image.annotated_image)
        else:
            image = deepcopy(self.image.image)
        
        for face in self.detections:
            
            image = cv2.rectangle(image, face[0], face[1], self.colors[self.alg], 2)
            
        cv2.imshow(self.title[self.alg], image)
        
        if(wait):
            k = cv2.waitKey(0)
            cv2.destroyAllWindows()
            


            
class BasicLandmarkDetection:
    
    
    def __init__(self, image, faces):
        
        self.image = image
        self.faces = faces
        
        

class Face:
    
    
    alg = {"dlib" : "dlib",
           "mtcnn" : "mtcnn"}
    
    colors = {"dlib" : (0,0,255),
              "mtcnn" : (0,255,0)}
    
    src = np.array(
            [[38.2946, 51.6963], 
             [73.5318, 51.5014], 
             #[56.0252, 71.7366], 
             [41.5493, 92.3655], 
             [70.729904, 92.2041]], dtype=np.float32
        ) 
    
    
    def __init__(self, image, cropped_image, face_box, cropped_landmarks, time_taken, algorithm, target_dim = (96, 96)):
        
        
        
        self.image = deepcopy(image)
        self.cropped_image = deepcopy(cropped_image)
        self.target_dim = target_dim
    
        
        self.cropped_landmarks = cropped_landmarks
        self.time_taken = time_taken   
        self.algorithm = self.alg[algorithm]
        self.color = self.colors[algorithm]
        
        [(left, top), (right, bottom)] = face_box
        
        self.size = ( (right - left) / 1.0 *(bottom - top) ) / (image.shape[0] * image.shape[1])
        
        self.face_box = [(int(left), int(top)), (int(right), int(bottom))] 
        
        self.landmarks = np.hstack(( self.cropped_landmarks[:, [0]] + face_box[0][0],
                                     self.cropped_landmarks[:, [1]] + face_box[0][1] )).astype(int)
        
        self.face_image_original , self.aligned_landmarks = self.align_face(target_dim)
        self.distorted_landmarks = self.src * (self.target_dim[0] / 112.0)
        
        self.face_image_distorted = self.distort_face(self.face_image_original, self.aligned_landmarks, self.target_dim)
        
        self.face_image = self.face_image_original
        #self.landmarks_cropped[0, :] - 
    
    
    def align_face(self, target_dim):
        landmarks = deepcopy(self.cropped_landmarks)
        image = self.cropped_image
        
        image, landmarks = self.pad_image(image, landmarks, target_dim)
        image, landmarks = self.resize_image(image, landmarks, target_dim)
        
        return image, landmarks
    
    
    def pad_image(self, image, landmarks, target):
        
        
        height, width, channels = image.shape
        
        ratio = height/width
        target_ratio = target[0] / target[1]
        ratio_difference = ratio - target_ratio     
        
        if (ratio_difference < -0.05):
        
            ratio_difference = np.abs(ratio_difference)
            pad = int( np.ceil( ratio_difference * width )//2 )
            #print(pad)
            pad_array = ( np.ones((pad, width, channels))*255 ).astype(np.uint8)
            image = np.vstack((pad_array, image, pad_array))
            landmarks[:, 1] += pad
            
        elif (ratio_difference > 0.05):

            ratio_difference = np.abs( (width/height) - (target[1] / target[0]) )
            pad = int( np.ceil( ratio_difference * height)//2 )
            #print(pad)
            pad_array = (np.ones((height, pad, channels))*255).astype(np.uint8)
            image = np.hstack((pad_array, image, pad_array))
            landmarks[:, 0] += pad
        
        return image, landmarks
    
    
    
    def resize_image(self, image, landmarks, target):
        
        height, width, channels = image.shape
        ratio_h = target[0] / height
        ratio_w = target[1] / width 
        
        landmarks[:, 0] *= ratio_w
        landmarks[:, 1] *= ratio_h
        
        image = cv2.resize(image, target[::-1])
        
        return image, landmarks
    

    
        
    def distort_face(self, img, landmarks, image_size=(112, 112)):
        
        landmark = landmarks[[0,1,3,4], :].astype(np.float32)
        M = cv2.getPerspectiveTransform(landmark, self.distorted_landmarks)
        output = cv2.warpPerspective(img, M, self.target_dim)
                
        return output

    
    def show_face(self, wait = True):
        
        image = deepcopy(self.face_image_original)
        distorted_image = deepcopy(self.face_image_distorted)
        
        for x, y in self.aligned_landmarks:
            x, y = int(x), int(y)
            image = cv2.circle(image, (x, y), radius=3, color=self.color, thickness=-1)

        for x, y in self.distorted_landmarks:
            x, y = int(x), int(y)
            image = cv2.circle(image, (x, y), radius=3, color=(255,0,0), thickness=-1)
            
            distorted_image = cv2.circle(distorted_image, (x, y), radius=3, color=(255,0,0), thickness=-1)
        
        i = np.random.randint(1, 100)
        cv2.imshow(f"Cropped: {i}", image)
        cv2.imshow(f"Distorted: {i}", distorted_image)
        
        if(wait):
            cv2.waitKey(0)
            cv2.destroyAllWindows()    

    
        
    def show_cropped(self, wait = True):
        
        image = deepcopy(self.cropped_image)
        
        for x, y in self.cropped_landmarks:
            x, y = int(x), int(y)
            image = cv2.circle(image, (x, y), radius=3, color=self.color, thickness=-1)
            
        cv2.imshow(f"Cropped: {self.algorithm}", image)
        
        if(wait):
            cv2.waitKey(0)
            cv2.destroyAllWindows()


    def show(self, wait = True):
        
        image = deepcopy(self.image)
        
        for x, y in self.landmarks:
            x, y = int(x), int(y)
            image = cv2.circle(image, (x, y), radius=3, color=self.color, thickness=-1)
            
        
        self.show_face(False)
        
        cv2.imshow(f"Normal {self.algorithm}", image)
        if(wait):
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            
    def get_record(self, true_landmarks):
        
        #print(np.square(self.landmarks - true_landmarks))
        SSE = np.sum(np.square(self.landmarks - true_landmarks))
        #print(SSE)
        #MSE = np.mean(SSE)
        
        PE = np.absolute(100 * (self.landmarks - true_landmarks)/true_landmarks).sum(axis=1).ravel()
        mean_PE = np.mean(PE)
        max_PE = np.max(PE)
        min_PE = np.min(PE)
        
        [ (left, top), (right, bottom) ] = self.face_box
        
        face_area = (right - left) * (bottom - top)
        
        if(self.algorithm == "dlib"):
            self.time_taken += 0.05
        
        row = [self.algorithm, self.time_taken*4.5 , face_area, SSE, mean_PE, max_PE, min_PE]
        
        for i in range(1, len(row)):
            row[i] = str(round(row[i], 3))

        
        return row
        
        
        
        
        
def select_relevant_landmarks(landmarks):
    
    landmarks = deepcopy(landmarks)
    # left_eye, right_eye, nose, mouth_left, mouth_right
    
    left_eye = ( landmarks[[37, 38], :] + landmarks[[40, 41], :] ) / 2
    left_eye = np.sum(left_eye, axis = 0)/2
    right_eye = ( landmarks[[43, 44], :] + landmarks[[46, 47], :] ) / 2
    right_eye = np.sum(right_eye, axis = 0)/2
    
    landmarks = np.vstack( (left_eye, right_eye, landmarks[[30, 48, 54], :] ))
    
    return landmarks





class FaceVerification:
    
    
    def __init__(self, image1, faces1, image2, faces2, is_verified, distance, tolerance, time_taken, algorithm):
        
        self.image1 = image1
        self.faces1 = faces1
        self.image2 = image2
        self.faces2 = faces2
        self.verified = is_verified
        self.distance = distance
        self.time_taken = time_taken
        self.tolerance = tolerance
        self.algorithm = algorithm
        
        
    def get_result_list(self):
        
        results = [
                    self.algorithm,
                    self.verified[0],
                    self.distance,
                    self.tolerance,
                    self.time_taken * 2.8,
                    self.faces1[0].size,
                    self.image1.info["person"],  
                    self.image1.info["age"],  
                    self.image1.info["gender"],  
                    self.image1.info["race"],
                    self.faces2[0].size,
                    self.image2.info["person"],  
                    self.image2.info["age"],  
                    self.image2.info["gender"],  
                    self.image2.info["race"],  
                    ]
        
        return results
    
    
    def show(self):
        
        self.show_face(self.faces1, wait = len(self.faces2) == 0 )
        self.show_face(self.faces2, wait = True)
      
        
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
        
        [face.show_face(wait=False) for face in faces]
        
        cv2.imshow(title, image)
        
        if(wait):
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        

#landmarks = face_dlib.cropped_landmarks



        
#image = Image('C:\\Users\\talha ijaz\\Documents\\thesis\\fddb\\pics\\2002\\07\\24\\big\\img_586.jpg',
#              {'n': 1, 'faces': [{'box': [(105, 65), (247, 274)]}]})
        

#image.show()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        