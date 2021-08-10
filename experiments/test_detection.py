import numpy as np
import pandas
import os
import cv2

from utils import Image, FaceDetection
from ingestor_fddb import FddbIngestor
from ingestor_wider import WiderFacesIngestor 

from violajones import ViolaJonesDetector
from hog import HogDetector
from yolo import YoloDetector
from mobilenet2 import MobileSsdDetector
import mobilenet


def get_detections_results(detection, dataset=""):

    algorithm = detection.alg
    time_taken = detection.time_taken
    n_faces = detection.image.info["n"]
    filename = detection.image.info["file"]
    
    truth_boxes = [face['box'] for face in detection.image.info["faces"]]
    detections = detection.detections
    
    
    def calculate_iou(box1, box2):
        [(x1_1, y1_1), (x2_1, y2_1)] = box1 
        [(x1_2, y1_2), (x2_2, y2_2)] = box2 
        
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
    
        if x_right < x_left or y_bottom < y_top:
            return 0.0
    
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    
        iou = intersection_area / float(box1_area + box2_area - intersection_area)
        assert iou >= 0.0
        assert iou <= 1.0
        
        return iou
    
    
    assignments = np.zeros((len(truth_boxes), )) - 1
    #print("############################################\n")
    #print(truth_boxes)
    #print(detections)
    for i in range(len(truth_boxes)):
        
        ious = np.array([calculate_iou(detection, truth_boxes[i]) for detection in detections])
        
        for j in range(len(ious)):
            if(j >= len(ious) and assignments[j] != -1):
                pass
                #ious[j] = 0

        if(len(ious) > 0):
            best_detection_index = np.argmax(ious)
            if ious[best_detection_index] > 0.4:
                assignments[i] = best_detection_index
                
            #print(f"Results = {i}  : {best_detection_index} ({ious[best_detection_index]})   {assignments}")
            #print(ious)
            #print()
            
    
    ious = []
    for i in range(len(assignments)):
        if(assignments[i] == -1):
            pass
            #ious.append(-1)
        else:
            #print(truth_boxes[i], detections[int(assignments[i])])
            ious.append(calculate_iou(truth_boxes[i], detections[int(assignments[i])]))
            

    n_detections = len(detections)
    assert len(np.where(assignments != -1)[0]) == len(ious)
    true_positives = len(np.where(assignments != -1)[0])
    false_positives = np.max(n_detections - true_positives, 0)
    false_negatives = len(np.where(assignments == -1)[0])
    average_iou = np.round(np.mean(ious) if (len(ious) > 0) else 0, 3)
    
    #print("Detections:", )
    #print("Assignments: ", assignments)
    #print("Ious: ", ious)
    #print("Metrics",  [n_detections, true_positives, false_positives, false_negatives, average_iou])
    
    image_area = detection.image.image.shape[0] * detection.image.image.shape[1]
    
    i = 1    
    rows = []
    for face in detection.image.info["faces"]:
        
        [(x1, y1), (x2, y2)] = face['box']
        
        
        blur = face.get('blur')
        expression = face.get('expression')
        illuumination = face.get('illumination')
        invalid = face.get('invalid')
        occlusion = face.get('occlusion')
        pose = face.get('pose')
        face_area = np.round( ( (x2 - x1) * (y2 - y1) ) / image_area, 3)
        
        
        row = [dataset, filename, algorithm, n_faces, n_detections, true_positives, false_positives, 
               false_negatives, average_iou, time_taken*4.5, face_area , i, blur, expression, illuumination, invalid, 
               occlusion, pose]
    
        row = [str(item) for item in row]
        
        rows.append(row)
        i = i+1
        
    assert len(rows) == n_faces
    
    return rows
                
              

columns =  [
    "dataset",
    "filename", 
    "algorithm",
    "n_faces", 
    "n_detections",
    "true_positives", 
    "false_positives", 
    "false_negatives", 
    "iou",
    "time_taken", 
    "face_area",
    "face_index",
    "blur", 
    "expression", 
    "illumination", 
    "invalid", 
    "occlusion", 
    "pose"
    ]


root_folder = "C:\\Users\\talha ijaz\\Documents\\thesis"

output_file = "results_detection.csv"


fddb_root_folder = os.path.join(root_folder, "fddb")
ingestor1 = FddbIngestor(fddb_root_folder)

wider_root_folder = os.path.join(root_folder, "wider")
ingestor2 = WiderFacesIngestor(wider_root_folder)


vj = ViolaJonesDetector()
hog = HogDetector()
yolo = YoloDetector()
ssd = MobileSsdDetector()
ssd2 = mobilenet.MobileSsdDetector()


f = open(output_file, "w")
f.write(";".join(columns)+"\n")


for ingestor in [ingestor2, ingestor1]:
    
    n_files = len(ingestor.files)
    
    for i in range(n_files):
        
        try:
            image = ingestor.get_image(i)
        except:
            continue
        
        for alg in [vj, hog, yolo, ssd]:
            
            detection = alg.detect(image)
            dataset = str(type(ingestor)).split("'")[1].split("_")[1].split(".")[0].upper()
            rows = get_detections_results(detection, dataset = dataset)
            
            for row in rows:
                row[1] +=  "_" + str(i) 
                f.write(";".join(row)+"\n")
            #print(detection.time_taken)
            #detection.show()
            
        if(i>0 and i%100 == 0):
            print(f"Dataset ({type(ingestor)}): {i} done")
            
    print("\n")
            
f.close()












