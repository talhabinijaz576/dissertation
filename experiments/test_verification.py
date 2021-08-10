from copy import deepcopy
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


import os
import numpy as np


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


root_folder = "C:\\Users\\talha ijaz\\Documents\\thesis"
output_file = "results/results_verification.csv"

lfw_ingestor = LFWIngestor(os.path.join(root_folder, "lfw"))
cacd_ingestor = CACDIngestor(os.path.join(root_folder, "CACD2000"))

ingestors = [("lfw", lfw_ingestor), 
             ("cacd", cacd_ingestor)]

recognizers = [facenet_recognizer, mobilefacenet_recognizer, vgg_recognizer, arcface_recognizer, cosface_recognizer]


columns = [ "dataset", 
            "filepath1", 
            "filepath2", 
            "correct", 
            "same",
            "algorithm",
            "prediction",
            "distance",
            "tolerance",
            "time_taken",
            "size1",
            "person1",  
            "age1",  
            "gender1",  
            "race1",
            "size2",
            "person2",  
            "age2",  
            "gender2",  
            "race2", 
            ]



if(not os.path.exists(output_file)):
    with open(output_file, "w") as f:
        column_row = ";".join(columns) + "\n" 
        f.writelines([column_row])
        #print(column_row)

l = [[] for _ in recognizers]

for ind in range(100):
    
    t = [[] for _ in recognizers]

    for ingestor_name, ingestor in ingestors:
        
        with open(output_file, "a") as f:
            
            for _ in range(100):
            
                try:
                    img1, img2, is_matching = ingestor.get_pair()
                    
                    for i in range(len(recognizers)):
                        

                    
                        try:
                            
                            recognizer = recognizers[i]
                            
                            verification = recognizer.verify(img1, img2)
                            success = is_matching == verification.verified
                            
                            ##"dataset", "filepath1", "filepath2", "correct", "same"
                            record = [ingestor_name, 
                                      os.path.relpath(img1.filepath, root_folder),
                                      os.path.relpath(img2.filepath, root_folder),
                                      success[0],
                                      is_matching] + verification.get_result_list()
                            
                            record = list(map(str, record))
                            
                            record_row = ";".join(record) + "\n" 
                            f.writelines([record_row])
                            
                            
                            l[i].append(success)
                            t[i].append(success)
            
                        except Exception as e:
                            pass
                            #print("Error: ", e)
                    
                except:
                    pass
        
            
    print(f"{ind}: {[np.mean(x) for x in t]},  {[np.mean(x) for x in l], }")
    
















