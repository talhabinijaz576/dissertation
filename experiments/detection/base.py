import numpy as np
import pandas as pd
import random
import utils
        


class BasicIngestor:
    
    def __init__(self):
        self.random_order = self.get_random_permutation()
    
    def get_random_permutation(self):
        permutations = np.random.permutation(len(self.files))
        self.current_permutation_index = 0
        return permutations
        
        
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
    
    
    
def BasicFaceDetector:
    
    def __init__(self):
        pass
        