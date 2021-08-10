from copy import deepcopy
import os
import numpy as np
import pandas as pd
from path import Path
import time

from utils import Image

from ingestor_recog import LFWIngestor, CACDIngestor
from violajones import ViolaJonesDetector
from hog import HogDetector
from landmark_detectors import DlibDetector
from base import BasicRecognizer
from utils import FaceVerification

from recognizer_cosface import CosfaceRecognizer
from recognizer_vggface import Senet50Recognizer
from recognizer_facenet import FacenetRecognizer
from recognizer_arcface import ArcfaceRecognizer
from recognizer_mobilefacenet import MobileFacenetRecognizer



face_detector = HogDetector()


cosface_recognizer = CosfaceRecognizer(deepcopy(face_detector), 
                                        DlibDetector(expand_dims = (0.0, 0.0), target_dim = (112, 112)),
                                        None, tolerance=1.24)



vgg_recognizer = Senet50Recognizer(deepcopy(face_detector),  
                                    DlibDetector(expand_dims = (0.0, 0.0), target_dim = (224, 224)), 
                                    None, tolerance=185.0)


facenet_recognizer = FacenetRecognizer(deepcopy(face_detector), 
                                        DlibDetector(expand_dims = (0.0, 0.0), target_dim = (160, 160)), 
                                        None, 1)


arcface_recognizer = ArcfaceRecognizer(deepcopy(face_detector), 
                                        landmark_detector = DlibDetector(expand_dims = (0.0, 0.0)), 
                                        dataset = None, tolerance = 0.6)



mobilefacenet_recognizer = MobileFacenetRecognizer(deepcopy(face_detector), 
                                                    DlibDetector(expand_dims = (0.0, 0.0), target_dim = (112, 112)), 
                                                    None, tolerance=1.25)





def get_file_annotations(file):
    
    name = str(Path(file).name)
    annotations = ingestor.annotations[name]
    
    return annotations
        
def extract_list(item):
    
    n_images = len(item[1])
    if(n_images == 0):
        return np.array([[None]*6])
    
    col1 = np.array([item[0]] * n_images).reshape(-1, 1)
    col2 = np.array( [n_images] * n_images ).reshape(-1, 1)
    col3 = np.array(item[1]).reshape(-1, 1)
    col4 = np.array(list(map(get_file_annotations, item[1])))
    
    info = np.hstack([col1, col2, col3, col4])
    
    return info


def extract_faces(image_path):
    #print(image_path)
    try:
        image = Image(image_path, {})
        face_detections = face_detector.detect(image)
        
        if(len(face_detections.detections) == 0):
            return np.nan
    except Exception as e:
        print(e)
        return np.nan
    
    return face_detections


def get_landmark_detection(face_detection):
    if(face_detection is np.nan):
        return np.nan
    
    landmark_detection = recognizer.landmark_detector.detect(face_detection)
    
    return landmark_detection



root_folder = "C:\\Users\\talha ijaz\\Documents\\thesis"



#ing1 = LFWIngestor(os.path.join(root_folder, "lfw"))
ing2 = CACDIngestor(os.path.join(root_folder, "CACD2000"))

ingestor = ing2
folderpath = "embeddings/cacd"





face_detector_class = HogDetector
landmark_detector_class = DlibDetector #(expand_dims = (0.0, 0.0), target_dim = (112, 112))
max_size = 10

items = list(ingestor.mapping.items())

dataset_all = list(map(extract_list, items))
dataset_all = list(filter(lambda row: row[0][0] is not None, dataset_all))
dataset_all = np.vstack(dataset_all)
#df = pd.DataFrame(dataset, columns = ["name", "n_images", "file", "age", "gender", "race"])


face_detector = face_detector_class()
people_all = np.unique(dataset_all[:, 0])
    

np.random.seed(0)
permutations_all = np.random.permutation(len(people_all))

save_i = 1
while( os.path.exists( os.path.join(folderpath, f"embeddings_{save_i}.npy") ) ):
    save_i = save_i + 1
    print(save_i)
    
print(f"Starting from :{save_i}")

#int(people_all.shape[0]/max_size)
for iter_i in range(save_i+2,  save_i+100):
    
    print(f"\n{iter_i}  {max_size}")
    print((iter_i*max_size), ((iter_i*max_size)+max_size))
    
    permutations = permutations_all[(iter_i*max_size):min(((iter_i*max_size)+max_size), people_all.shape[0])]
    
    people = people_all[permutations]    
    is_row_selected = np.vectorize(lambda name: name in people)
    
    selected_inds = is_row_selected(dataset_all[:, 0])
    dataset = dataset_all[selected_inds]
    
    extract_faces_vec = np.vectorize(extract_faces)
    
    t = time.time()
    try:
        face_detections = extract_faces_vec(dataset[:, 2])
    except Exception as e:
        print(e)
        continue
    
    valid_inds = np.array(list(map(lambda x: x is not np.nan, face_detections)))
    
    dataset = dataset[valid_inds]
    face_detections = face_detections[valid_inds]
    
    models = [
                (cosface_recognizer, "cosface"),
                (vgg_recognizer , "vggface"),
                (facenet_recognizer, "facenet"),
                (arcface_recognizer, "arcface"),
                (mobilefacenet_recognizer, "mobilefacenet"),
            ]
    
    embeddings_all = []
    
    print("Dataset: ", dataset.shape)
    print("People: ", people.shape[0])
    
    for recognizer, name in models:
        
        savefile = os.path.join(folderpath, name)
        
        landmark_detections = list(map(get_landmark_detection, face_detections))
        
        faces = list(map(lambda faces: faces[0] if (faces is not np.nan) else np.nan, landmark_detections))
        embeddings = recognizer.calculate_batch_embeddings(faces)
        embeddings_all.append(np.array([name, embeddings]))
        #embeddings_all[name] = embeddings
        print(f"Saving emebeddings ({name})   [time: {time.time() - t}]")
        np.save(os.path.join(folderpath, f"embeddings_{iter_i}.npy"), np.array(embeddings_all))
        print(name, embeddings.shape)
        
        
    print("Saving dataset and people")  
        
    np.save(os.path.join(folderpath, f"dataset_{iter_i}.npy"), dataset)
    np.save(os.path.join(folderpath, f"people_{iter_i}.npy"), people)
        
    print(time.time() - t)
    
    save_i = save_i + 1

#item = items[17]





#face_detector = HogDetector()



















































