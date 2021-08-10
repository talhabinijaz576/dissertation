import pandas as pd
import numpy as np
from base import BasicIngestor
import os
import glob
import cv2
import imutils
from utils import select_relevant_landmarks


class Landmark300Ingestor(BasicIngestor):
    
    def __init__(self, root_folder, small=False):
        
        self.small = small
        file_mappings = self.get_file_mapping(root_folder)    
        self.labels = self.load_labels(file_mappings)
        self.files = list(self.labels.keys())

        
        super().__init__()
        
        
    def get_file_mapping(self, root_folder):
        
        all_files = glob.glob(f"{root_folder}/*/*")
        image_files = list(filter(lambda file: file.endswith(".png"), all_files))
        points_files = list(filter(lambda file: file.endswith(".pts"), all_files))
        
        
        file_mappings = {}
        for image_file in image_files:
            
            points_file = image_file[:-4] + ".pts"
            if(points_file in points_files):
                file_mappings[image_file] = points_file
                
        return file_mappings
    
    
    def load_info(self, points_file):
    
        with open(points_file, "r") as f:
            lines = f.readlines()
        
        lines = lines[3:-1]    
        points = list(map(lambda line : [float(float(x)) for x in line.strip().split(" ")], lines))
        points = np.array(points)
        
        points = select_relevant_landmarks(points)
        
        info = {"landmarks" : points}
        
        return info
    
    
    def load_labels(self, file_mappings):
        
        labels = {}
        
        for image_file in list(file_mappings.keys()):

            points_file = file_mappings[image_file]
            info = self.load_info(points_file)
            labels[image_file] = info
            
        return labels
        

        
        labels_file = os.path.join(self.labels_folder, "wider_face_train_bbx_gt.txt")
        with open(labels_file) as f:
            lines = f.readlines()

        face_attributes = ["x1", "y1", "w", "h", "blur", "expression", 
                           "illumination", "invalid", "occlusion", "pose"] 
        
        def extract_face(face_line):
            
            values = list(map(float, face_line.split(" ")))
            face = dict(zip(face_attributes, values))
            
            x1 = int(face["x1"])
            x2 = int(face["x1"]+face["w"])
            y1 = int(face["y1"])
            y2 = int(face["y1"]+face["h"])
            
            face["box"] = [(x1, y1), (x2, y2)]
            
            return face
        
        extract_face_vectorized = np.vectorize(extract_face)
        
        def extract_information(i):
            
            try:
                subpath = lines[i]
                path = os.path.join(self.images_folder, *subpath.split("/"))
                n_faces = int(lines[i+1])
                faces = list(extract_face_vectorized(lines[i+2 : i+2+n_faces]))
                info = {"n" : n_faces, "faces" : faces, "file" : lines[i]}
                return path, info
            except:
                return None, None
            
        extract_information_vectorized = np.vectorize(extract_information)
        
        lines = np.array(list(map(lambda line: line.strip(), lines)))
        indices_map = np.vectorize(lambda line: "--" in line)(lines)
        break_indices = np.where(indices_map)[0]
        
        labels_array = extract_information_vectorized(break_indices)
        labels = dict(zip(*labels_array))
        
        return labels
                
        
        
            


        
        
        
        
        
        
        
        
        
        
    


