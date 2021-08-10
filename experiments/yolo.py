import warnings
warnings.filterwarnings("ignore")
import numpy as np
from utils import Image, FaceDetection
from base import BasicFaceDetector
import cv2
import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import PIL
from yolo3.yolo import YOLO




class DetectorAdpater:
    
    def __init__(self, model_path):
        paddle.enable_static()
        self.model = fluid.Executor(fluid.CPUPlace())
        components = fluid.io.load_inference_model(dirname = model_path, executor = self.model)
        self.inference_model = components[0] 
        self.target_names = components[1] 
        self.fetch_targets = components[2] 
    
    def predict(self, images, confidence_threshold = 0.5, nms_threshold=0.3):
        if len(images)==0: 
            return [], []
        
        image_array = list(map(scale_img, images))
        
        out1, out2 = self.model.run(program = self.inference_model,
                                    feed = { self.target_names[0] : np.array(image_array, dtype='float32') },
                                    fetch_list = self.fetch_targets)

        yolo_bouding_boxes = _get_bboxes([out1, out2])
        selected_boxes = _NMS(yolo_bouding_boxes,confidence_threshold, nms_threshold)
        selected_boxes = list(map(toBoundingBox, selected_boxes))
        
        return image_array, selected_boxes


class YoloDetector(BasicFaceDetector):
    
    
    def __init__(self, detector_type = 0):
        self.detector = self.load_detector()
        super().__init__()
        
        
    def load_detector(self, i=0):
        detector = YOLO()
        return detector
    
    
    def detect(self, image, min_conf = 0.6):
        
        
        
        image_pil = PIL.Image.fromarray(cv2.cvtColor(image.image, cv2.COLOR_BGR2RGB))
        
        start_time = time.time()
        detections = self.detector.detect_image(image_pil)
        time_taken = time.time() - start_time
        
        detections = list(filter(lambda x: x[1] > min_conf, detections))
        
        detections = list(map(lambda row: [ (row[0][1], row[0][0]), 
                                            (row[0][3], row[0][2]), ], detections))
        
        detection = FaceDetection(image, detections, time_taken, alg = "YOLO")
        
        return detection
   


#img = 'C:\\Users\\talha ijaz\\Documents\\thesis\\fddb\\pics\\2002\\07\\24\\big\\img_586.jpg'



#image = Image('C:\\Users\\talha ijaz\\Documents\\thesis\\fddb\\pics\\2002\\07\\24\\big\\img_586.jpg',
#              {'n': 1, 'faces': [{'box': [(105, 65), (247, 274)]}]})

#yolo = YoloDetector()
#detection = yolo.detect(image)
#detection.show()

        
        
#img = cv2.imread('C:\\Users\\talha ijaz\\Documents\\thesis\\fddb\\pics\\2002\\07\\24\\big\\img_586.jpg')
        
#image = Image('C:\\Users\\talha ijaz\\Documents\\thesis\\fddb\\pics\\2002\\07\\24\\big\\img_586.jpg',
#              {'n': 1, 'faces': [{'box': [(105, 65), (247, 274)]}]})






