import numpy as np
import pandas
import os
import cv2
import face_alignment
import dlib

from utils import Image, FaceDetection
from ingestor_fddb import FddbIngestor
from ingestor_wider import WiderFacesIngestor 
from ingestor_300w import Landmark300Ingestor

from violajones import ViolaJonesDetector
from hog import HogDetector
from landmark_detectors import MCTNNDetector, DlibDetector
import mobilenet

from mtcnn import MTCNN
import time



def remove_irrelevant_face(face_detects, true_landmarks):
    
    
    delete_list = []
    
    for i in range(len(face_detects.detections)):
        
        detection = face_detects.detections[i]
        
        [(left, top), (right, bottom)] = detection
        x_checks = np.bitwise_and( true_landmarks[:, 0] > left ,  true_landmarks[:, 0] < right )
        y_checks = np.bitwise_and( true_landmarks[:, 1] > top ,  true_landmarks[:, 1] < bottom )
        score = (np.mean(x_checks) + np.mean(y_checks) ) /  2  
        
        if(score < 0.8):
            delete_list.append(i)
            
    for i in delete_list[::-1]:
        del face_detects.detections[i]
        
    




root_folder = "C:/Users/talha ijaz/Documents/thesis/300w"

output_file = "results_landmark.csv"

ing = Landmark300Ingestor(root_folder, True)
face_detector = HogDetector()
#face_detector = ViolaJonesDetector()
mtcnn_detector = MCTNNDetector(expand_dims = (1, 1))
dlib_detector = DlibDetector(expand_dims = (0.05, 0.3))


success, failure, no_face = 0, 0, 0
images, mtcnn_shapes, dlib_shapes, landmarks = [], [], [], []

f = open(output_file, "w")

columns = ["Algorithm", "time_taken", "face_area", "SSE", "mean_PE", "max_PE", "min_PE"]

f.write(";".join(columns) + "\n")

for i in range(len(ing.files)):
#for i in range(5):
    
    if(i%15 == 0):
        print(f"Done: {i}")
        
    im = ing.get_image(i)     
    face_detections = face_detector.detect(im)
    
    face_detections = face_detector.detect(im)
    remove_irrelevant_face(face_detections, im.info["landmarks"])
    
    if(len(face_detections.detections) < 1):
        #print(f"Incorrect number: {len(face_detections.detections) }")
        no_face += 1
        continue
    
    try:
        faces_mtcnn = mtcnn_detector.detect(face_detections)
        faces_dlib = dlib_detector.detect(face_detections)
    except:
        failure += 1
        continue
        
    if(len(faces_mtcnn) == 0 or len(faces_dlib) == 0):
        failure +=1
        continue
    
    assert len(face_detections.detections) == 1 and len(faces_mtcnn) == 1 and len(faces_dlib) == 1
     
    #images.append(im)
    #mtcnn_shapes.append(faces_mtcnn)
    #dlib_shapes.append(faces_dlib)
    #landmarks.append(true_landmarks)
    success += 1

    face_mtcnn = faces_mtcnn[0]
    face_dlib = faces_dlib[0]
    
    mtcnn_record = face_mtcnn.get_record(true_landmarks = im.info["landmarks"])
    dlib_record = face_dlib.get_record(true_landmarks = im.info["landmarks"])
    
    f.write(";".join(mtcnn_record) + "\n")
    f.write(";".join(dlib_record) + "\n")
    
    #face_mtcnn.show(False)
    #face_dlib.show(False)
    #im.show(False)   
    #face_detections.show(False, False)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    #break


f.close()


#shape = face_utils.shape_to_np(shape)


    
    
    
    
    
    
    
    



