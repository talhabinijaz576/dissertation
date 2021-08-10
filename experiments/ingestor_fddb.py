import pandas as pd
import numpy as np
from base import BasicIngestor
import os


class FddbIngestor(BasicIngestor):
    
    def __init__(self, root_folder):

        self.labels_folder = os.path.join(root_folder, "labels")
        self.images_folder = os.path.join(root_folder, "pics")
        self.labels = self.load_labels()
        self.files = list(self.labels.keys())
        
        super().__init__()

        
    def load_labels(self):
        
        lines = []
        for i in range(1, 11):
            if(i<10):
                i = f"0{i}"
            filepath = os.path.join(self.labels_folder, f"FDDB-fold-{i}-ellipseList.txt")
            
            with open(filepath, "r") as f:
                file_lines = f.readlines()
                file_lines = list(map(lambda line: line.strip(), file_lines))
                lines = lines + file_lines
            
            
        def extract_face(face_line):
            
            face_line = face_line.replace("  ", " ")
            attributes = list(map(float, face_line.split(" ")[:-1]))
            major_axis_radius, minor_axis_radius, angle, centre_x, centre_y = attributes
            a, b = major_axis_radius, minor_axis_radius,
            delta_x = np.sqrt( ( (a**2) * np.cos(angle)**2 ) + ( (b**2) * np.sin(angle)**2 ) )
            delta_y = np.sqrt( ( (a**2) * np.sin(angle)**2 ) + ( (b**2) * np.cos(angle)**2 ) )
            
            x1 = int( centre_x - delta_x )
            x2 = int( centre_x + delta_x )
            y1 = int( centre_y - delta_y )
            y2 = int( centre_y + delta_y )
            box = [(x1, y1), (x2, y2)]
            face = {"box" : box}
            
            return face
        
        extract_face_vectorized = np.vectorize(extract_face)
        
        def extract_information(i):
            
            subpath = lines[i]
            path = os.path.join(self.images_folder, *subpath.split("/")) + ".jpg"
            n_faces = int(lines[i+1])
            faces = list(extract_face_vectorized(lines[i+2 : i+2+n_faces]))
            info = {"n" : n_faces, "faces" : faces, "file" : lines[i]}
            
            return path, info
            
        extract_information_vectorized = np.vectorize(extract_information)
        
        lines = np.array(list(map(lambda line: line.strip(), lines)))
        indices_map = np.vectorize(lambda line: "img_" in line)(lines)
        break_indices = np.where(indices_map)[0]
        
        labels_array = extract_information_vectorized(break_indices)
        labels = dict(zip(*labels_array))
        
        return labels



