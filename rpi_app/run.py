from utils import Image, FaceDetection, Face
from landmark_detectors import DlibDetector
from recognizer_facenet import FacenetRecognizer
from hog import HogDetector


import websockets
import json
import ast
import requests
import websocket
import _thread
import time
import numpy as np
import cv2
from pprint import pprint
from copy import deepcopy





DEBUG = False

class FaceRecognitionPipeline:
    
    
    def __init__(self, url, recognizer):
        
        self.url = url
        self.base_url = url.split("/app")[0]
        self.recognizer = recognizer
        self.sync_images()
        
        self.running = False
        self.frame = None
        self.face_detections = None
        self.face = None
        self.display_image = None
        self.frame_message = ""
        self.is_recognizing = False
        
        
    def stop(self):
        self.running = False

    
    def sync_images(self):
        
        url = self.url + "?get_images_list=True"
        print(f"Syncing images at {url} ")
        employees_dict = self.get_employees(url)
        ids = []
        names = []
        urls = []
        
        for employee_id in employees_dict:
            
            images = employees_dict[employee_id]["images"]
            name = employees_dict[employee_id]["name"]
            
            urls_temp = list(map(lambda filename: self.base_url + filename, images))
            names_temp = [name] * len(urls_temp)
            ids_temp = [employee_id] * len(urls_temp)
            
            ids += ids_temp
            names += names_temp
            urls += urls_temp
        
        self.ids = np.array(ids).ravel()
        self.names = np.array(names).ravel()
        self.images = list(map(self.get_image, urls))
        try:
            self.embeddings = np.vstack(list(map(self.get_embeddings, self.images)))
        except:
            self.embeddings = np.array([])
        print(f"Embeddings: {self.embeddings.shape} ({self.names.shape}, {self.ids.shape})")
        #for i in range(len(names)):
        #    images[i].show()
        
        
    def get_employees(self, url):
        
        try:
            r = requests.get(url, json = {"get_images_list": True})
            content = r.content.decode()
            #pprint(content)
            info = ast.literal_eval(content)
            return info
        except:
            pass
        
        return {}
    
    
    def get_image(self, url, filename = "temp.png"):
        
        try:
            #print(f"Obtaining Image: {url}")
            r = requests.get(url)
            file = open(filename, "wb")
            file.write(r.content)
            file.close()
            image = Image(filename, {})
            image = self.process_image(image) 
            
            return image
            
        except Exception as e:
            print(e)
        
        return None
    
    
    def process_image(self, image):
        
        return image
    
    
    
    def get_embeddings(self, image):
        
        face_detections = self.recognizer.face_detector.detect(image)
        faces = self.recognizer.landmark_detector.detect(face_detections)
        face = faces[0]
        if(DEBUG):
            face.show()
        embeddings = self.recognizer.calculate_face_embeddings(face)
        
        return embeddings
    
    
    def face_detection_loop(self):
        
        while(self.running):

            if self.frame is not None:
                self.face_detections = self.recognizer.face_detector.detect(self.frame)
            else:
                self.face_detections = None
            
            #time.sleep(0.1)
                
                
    def face_alignment_loop(self):
        
        while(self.running):

            if self.face_detections is not None:
                #print("Face detections: ", self.face_detections)
                faces = self.recognizer.landmark_detector.detect(self.face_detections)
                if(len(faces) > 0):
                    self.face = faces[0]
                    #print("Face Detected: ", self.face)
                else:
                    self.face = None
            else:
                self.face = None
                
            #time.sleep(0.1)
    
    
    def reset_frame_message(self, delay = 5):
        
        def reset_function():
            time.sleep(delay)
            self.frame_message = ""
            
        _thread.start_new_thread(reset_function, ())
        
        return None
    
    
    def request_access(self, employee_id):
        
        print(f"Requesting acccess for employee: {employee_id}")
        
        try:
            url = self.url + f"?request_access={employee_id}"
            r = requests.get(url)
            content = r.content.decode().replace("true", "True").replace("false", "False")
            info = ast.literal_eval(content)
            pprint(info)
            has_access = bool(info["access_granted"])
            
            return has_access
            
        except Exception as e:  
            #print(e)
            return False
        
    
    
    def recognize_face(self, timeout=10, is_infinite=True):
        
        self.is_recognizing = True
        request_access = False
        start_time = time.time()
        
        if(not is_infinite):
            self.frame_message = "Detecting..."
        
        while( (time.time() - start_time) <= timeout ):
            
            face = deepcopy(self.face)
            if(face is None):
                continue
            
            if(self.embeddings.shape[0] == 0):
                if(is_infinite):
                    self.frame_message = "Unknown Person"
                else:
                    self.frame_message = "Unknown Person [Access Denied]"
                    self.request_access(-1)
                    self.reset_frame_message()
                    
                return False, "", 0
                
                
            algorithm_start_time = time.time()
            embeddings = self.recognizer.calculate_face_embeddings(face).reshape(1, -1)
            distances = self.recognizer.calculate_face_distance(embeddings, self.embeddings).ravel()
            
            if(len(distances) == 0):
                if(is_infinite):
                    self.frame_message = "Unknown Person"
                else:
                    self.frame_message = "Unknown Person [Access Denied]"
                    self.request_access(-1)
                    self.reset_frame_message()
                    
                return False, "", 0
            
            i_argmin = np.argmin(distances)
            min_distance = distances[i_argmin]
            #print("Recognition array :", distances, min_distance)
            
            if(min_distance < self.recognizer.tolerance):
                
                time_taken = round( time.time() - algorithm_start_time, 3)
                #print(i_argmin, self.names, type(i_argmin), type(self.names))
                name = self.names[i_argmin]
                employee_id = self.ids[i_argmin]
                
                if(not is_infinite):
                    
                    has_access = self.request_access(employee_id)
                    access_message = "Access Granted" if has_access else "Access Denied"
                    self.frame_message = f"{name}  [{access_message}]  ({time_taken}s)"
                    self.reset_frame_message()
                    
                else:
                    
                    self.frame_message = f"{name}"
                    
                self.is_recognizing = False
                #(True, name, time_taken)
                
                return True, name, time_taken
            
            else:
                if(is_infinite):
                    self.frame_message = "Unknown Person"
                else:
                    self.frame_message = "Unknown Person [Access Denied]"
                    #request_access = True
                    self.reset_frame_message()
                    self.request_access(-1)
                    break
                    
                
            
        
        if(request_access):
            self.request_access(-1)
        self.is_recognizing = False
        
        if(not is_infinite):
             time_taken = round( time.time() - start_time, 3) 
             self.frame_message = "Detection unsuccessful"
             self.reset_frame_message()
             return False, "", time_taken
             
        
        
        return False, "", None
    
    
    def recognize_face_async(self, timeout=10):
        
        if(True or not self.is_recognizing):
            print("Starting recognition")
            _thread.start_new_thread(self.recognize_face, (timeout, False))
            self.is_recognizing = False
            
        else:
            print("Recognition already in progress")
            
            
    def face_recognition_loop(self):
        
        while(self.running):
            self.recognize_face(10, True)
            #time.sleep(0.1)
        
        
                  
                
    def draw_frame_loop(self):
        
        while(self.running):
            
            if(self.frame is None):
                time.sleep(0.1)
                self.display_image = None
                continue
            
            self.display_image = deepcopy(self.frame.image)
            
            cv2.putText(self.display_image, 
                        self.frame_message, 
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                        (0, 0, 255), 2, cv2.LINE_4)
            
            face = deepcopy(self.face)
            if(face is not None):
                
                self.display_image = cv2.rectangle(self.display_image, 
                                                   face.face_box[0], 
                                                   face.face_box[1], (255,0,0), 4)  
            
                for x, y in face.landmarks:
                    x, y = int(x), int(y)
                    self.display_image = cv2.circle(self.display_image, (x, y), radius=4, color=(0, 0, 255), thickness=-1)
            
            #time.sleep(0.1)
            
            #for x, y in info["landmarks"]:
            #    x, y = int(x), int(y)
            #    annotated_image = cv2.circle(image, (x, y), radius=3, color=(0, 0, 255), thickness=-1)
            
            
    def show_stream(self):
        
        key = False
        while(self.running):
            
            if(self.frame is None):
                time.sleep(0.1)
                self.display_image = None
                time.sleep(0.1)
                continue
            
            display_image = deepcopy(self.frame.image)
            cv2.putText(display_image, 
                        self.frame_message, 
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                        (0, 0, 255), 2, cv2.LINE_4)
            
            face = deepcopy(self.face)
            if(face is not None):
                
                display_image = cv2.rectangle(display_image, face.face_box[0], face.face_box[1], (255,0,0), 4)  
                for x, y in face.landmarks:
                    x, y = int(x), int(y)
                    display_image = cv2.circle(display_image, (x, y), radius=4, color=(0, 0, 255), thickness=-1)
                   
                    
            cv2.imshow('frame', display_image)
            key = cv2.waitKey(1)
                
            if(key != -1):
                #print("Key pressed: ", key, 0xff)
                
                if(key == ord('d')):
                    self.recognize_face_async(timeout=10)
                
                if(key == ord('q')):
                    self.stop()
                    
                if(key == ord('s')):
                    self.sync_images()
                    
                
    
    
    def run_loop(self, infinite_recognition = True):
        
        self.running = True
        
        vid = cv2.VideoCapture(0)
        ret, frame = vid.read()
        self.frame = Image(None, {}, frame = frame)
        
        _thread.start_new_thread(self.face_detection_loop, ())
        _thread.start_new_thread(self.face_alignment_loop, ())
        #_thread.start_new_thread(self.draw_frame_loop, ())
        _thread.start_new_thread(self.show_stream, ())
        
        if(infinite_recognition):
            print("Infinite recognititon")
            _thread.start_new_thread(self.face_recognition_loop, ())
        
        while(self.running):
            
            ret, frame = vid.read()
            self.frame = Image(None, {}, frame = frame)
            time.sleep(0.1)
            
            

        
        self.frame = None
        vid.release()
        cv2.destroyAllWindows()







class Application:
    
    
    def __init__(self, config_file = "config.json"):
        
        self.config = self.load_config(config_file)
        self.token = self.get_device_token()
        self.url = self.config["url"].format(self.token)
        self.websocket_url = self.config["websocket_url"].format(self.token)
        self.recognizer = FacenetRecognizer(HogDetector(), 
                                            DlibDetector(expand_dims = (0.0, 0.0), target_dim = (160, 160)), 
                                            None, 1)
        self.pipeline = FaceRecognitionPipeline(self.url, self.recognizer)
        


    def load_config(self, filepath):
        
        with open(filepath) as f:
            lines = f.readlines()
            lines = list(map(str.strip, lines))
            config_str = "".join(lines)
            config = ast.literal_eval(config_str)
            
        return config
    
    
    def get_device_token(self, n=10):
        
        try:
            device_token_file = self.config["device_token_file"]
            with open(device_token_file, "r") as f:
                token = f.readlines()[0].strip()
                
        except:
            token = "NONE"
        
        return token
    
    
    
    def initialize_websocket(self):
        
        
        
        def on_message(ws, message):
            message = ast.literal_eval(message)
            event = message["event"]
            if(event == "sync"):
                self.pipeline.sync_images()
                
        
        self.ws = websocket.WebSocketApp(self.websocket_url, on_message = on_message,)
        _thread.start_new_thread(self.ws.run_forever, ())
        
    
    def stop(self):
        
        self.ws.close()
        self.pipeline.stop()
        print("Application stopped")
            


    def run(self, infinite_recognition = True):
        
        self.initialize_websocket()
        print("Application Started")
        self.pipeline.run_loop(infinite_recognition)
        self.stop()
        
    
    
    
if __name__ == "__main__":
    
    app = Application()
    app.run(infinite_recognition = False)

























