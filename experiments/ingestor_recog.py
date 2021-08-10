import os
import numpy as np
import pandas as pd
import glob
from pathlib import Path
import random

from base import BasicFaceRecognitonIngestor
from utils import Image
import scipy.io
from path import Path




class LFWIngestor(BasicFaceRecognitonIngestor):
    
    
    def __init__(self, root_folder):
        
        self.dataset = "LFW"
        self.root_folder = root_folder
        self.all_files = glob.glob(os.path.join(root_folder, "*", "*"))
        self.people = os.listdir(root_folder)
        self.mapping = self.generate_mapping()
        self.annotations = self.load_annotations()
        
        self.single_examples = list(filter(lambda key: len(self.mapping[key]) == 1 , self.mapping))
        self.multiple_examples = list(filter(lambda key: len(self.mapping[key]) >= 2 , self.mapping))
        
        super().__init__()
        
        
    def get_relative_filename(self, file):
        
        pieces = file.split("_")[1:-1]
        name = "_".join( pieces)
        name = os.path.join(self.root_folder, name)
        filename = os.path.relpath(name, self.root_folder)
        
        return filename
        
    
        

    def generate_mapping(self):
        
        mapping = dict(zip(self.people, [[] for _ in range(len(self.people))]))

        for file in self.all_files:
            relative_path = os.path.relpath(file, self.root_folder)
            person_name = str(Path(relative_path).parent)
            mapping[person_name].append(file)
            
        return mapping
    
    
    def load_annotations(self):
        
        file = os.path.join(self.root_folder, "annotations.csv")
        df = pd.read_csv(file)
        #['File', 'Age', 'Gender', 'Race']
        
        files = list(df["File"])
        values = list(map(list, df.values[:, 1:]))
        annotations = dict(zip(files, values))
        
        return annotations
    
    
    def get_annotation(self, file):

        filename = str(Path(file).name)
        annotation = self.annotations[filename]
        self.annotations = self.load_annotations()
        
        return annotation
        
    
    
    # def load_mat_file(self, file):
        
    #     data = mat73.loadmat(file)

    #     files = []
    #     labels = []
    #     columns = ['File', 'Age', 'Gender', 'Race']  #1 White, 0 Other
    #     for i in range(5):
    #         files.append(np.array(data["image_list_5fold"][i][0]))
    #         labels.append(data["label"][i][0])
        
    #     files = np.vstack(files)
    #     labels = np.vstack(labels)
    #     df = np.hstack((files, labels))
    #     df = pd.DataFrame(df, columns = columns)
        
    #     df[["Age", "Gender", "Race"]] = df[["Age", "Gender", "Race"]].astype(float).astype(int)
        
    #     return df
        
    
    
    
    
class CACDIngestor(BasicFaceRecognitonIngestor):
    
    
    def __init__(self, root_folder):
        
        self.dataset = "UTK"
        self.root_folder = root_folder
        self.all_files = os.listdir(root_folder)
        self.people = self.get_people_list()
        self.mapping = self.generate_mapping()
        self.annotations = self.load_annotations()
        
        self.single_examples = list(filter(lambda key: len(self.mapping[key]) == 1 , self.mapping))
        self.multiple_examples = list(filter(lambda key: len(self.mapping[key]) >= 2 , self.mapping))
        
        super().__init__()
        
        
    def get_relative_filename(self, file):
        filename = os.path.relpath(file, self.root_folder)
        return filename
        
        
        
    def extract_name(self, file):
        
        pieces = file.split("_")[1:-1]
        name = "_".join( pieces)
        
        return name
            
    
    def get_people_list(self):
        
        people = set(map(self.extract_name, self.all_files))
        people = list(people)
        
        return people
        
        

    def generate_mapping(self):
        
        mapping = dict(zip(self.people, [[] for _ in range(len(self.people))]))

        for file in self.all_files:
            person_name = self.extract_name(file)
            file = os.path.join(self.root_folder, file)
            mapping[person_name].append(file)
            
        return mapping
    
    
    def load_annotations(self):
        
        func1 = lambda file: [int(file.split("_")[0]), None, None]
        attributes = list(map(func1, self.all_files))
        annotations = dict(zip(self.all_files, attributes))
        
        return annotations
    
    
    def get_annotation(self, file):

        filename = str(Path(file).name)
        annotation = self.annotations[filename]
        self.annotations = self.load_annotations()
        
        return annotation
    
    
        




#root_folder = "C:\\Users\\talha ijaz\\Documents\\thesis"


#ing1 = LFWIngestor(os.path.join(root_folder, "lfw"))

#ing2 = CACDIngestor(os.path.join(root_folder, "CACD2000"))





#for ing in [ing1]:#, ing2]:
#    
#    for _ in range(100):
#        img1, img2, is_matching = ing.get_pair()
        #print(is_matching)
        #img1.show(wait = False)
        #img2.show(wait = True)



























