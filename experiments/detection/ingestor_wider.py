import pandas as pd
import numpy as np
from base import BasicIngestor
import os


class WiderFacesIngestor(BasicIngestor):
    
    def __init__(self, root_folder):

        self.labels_folder = os.path.join(root_folder, "labels")
        self.images_folder = os.path.join(root_folder, "images")
        self.labels = self.load_labels()
        self.files = list(self.labels.keys())
        
        super().__init__()
        
        
    def load_labels(self):
        
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
                info = {"n" : n_faces, "faces" : faces}
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
                
        
        
        
#root_folder = "C:/Users/talha ijaz/Documents/thesis/wider"
#ing = WiderFacesIngestor(root_folder)

