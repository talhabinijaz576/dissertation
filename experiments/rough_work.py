import numpy
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




root_folder = "C:\\Users\\talha ijaz\\Documents\\thesis"

fddb_root_folder = os.path.join(root_folder, "fddb")
ingestor1 = FddbIngestor(fddb_root_folder)

wider_root_folder = os.path.join(root_folder, "wider")
ingestor2 = WiderFacesIngestor(wider_root_folder)

#face = fddb_ingestor.get_random_image()
#face = wider_ingestor.get_random_image()
#face.show()

#image = fddb_ingestor.get_random_image()
#d = ssd.detect(fddb_ingestor.get_random_image()).time_taken
#print(d)
#d = ssd.detect(fddb_ingestor.get_random_image()).time_taken
#print(d)


vj = ViolaJonesDetector()
#detection = vj.detect(image)
#detection.show()

hog = HogDetector()
#detection = hog.detect(image)
#detection.show()

yolo = YoloDetector()
#detection = yolo.detect(image)
#detection.show()

ssd = MobileSsdDetector()
#ssd2 = mobilenet.MobileSsdDetector()
#detection = ssd.detect(image)
#detection.show()

for image in ingestor1.get_random_images(10):
    vj.detect(image).show(False, False)
    hog.detect(image).show(False, False)
    yolo.detect(image).show(False, False)
    ssd.detect(image).show(False, False)
    image.show()
    
    
for image in ingestor2.get_random_images(10):
    vj.detect(image).show(False, False)
    hog.detect(image).show(False, False)
    yolo.detect(image).show(False, False)
    ssd.detect(image).show(False, False)
    image.show()





















