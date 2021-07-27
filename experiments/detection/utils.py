import cv2
from copy import deepcopy


class Image:
    
    def __init__(self, filepath, info):
        self.filepath = filepath
        self.info = info
        self.image = cv2.imread(filepath)
        self.annotated_image = self.get_annotated_image()
        
        
    def get_annotated_image(self):
        image = deepcopy(self.image)
        for face in self.info["faces"]:
            image = cv2.rectangle(image, face["box"][0], face["box"][1], (255,0,0), 2)    
        return image
                
        
    def show(self):
        cv2.imshow("picture", self.annotated_image)
        k = cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        
        
class FaceDetection:
    
    def __init__(self, image, detected_faces):
        self.image = image
        self.detections = detected_faces