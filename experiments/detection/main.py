import numpy
import pandas
import os
import cv2

from ingestor_fddb import FddbIngestor
from ingestor_wider import WiderFacesIngestor 




root_folder = "C:\\Users\\talha ijaz\\Documents\\thesis"

fddb_root_folder = os.path.join(root_folder, "fddb")
fddb_ingestor = FddbIngestor(fddb_root_folder)

wider_root_folder = os.path.join(root_folder, "wider")
wider_ingestor = WiderFacesIngestor(wider_root_folder)

#face = fddb_ingestor.get_random_image()
#face = wider_ingestor.get_random_image()
#face.show()



