import websockets
import json
import ast
import requests
import random




class Installer:
    
    def __init__(self, config_file = "config.json"):
        
        self.config = self.load_config(config_file)
        self.device_number = self.generate_device_token()
        self.url = self.config["url"].format(self.device_number)
        self.websocket_url = self.config["websocket_url"].format(self.device_number)


    def load_config(self, filepath):
        
        with open(filepath) as f:
            lines = f.readlines()
            lines = list(map(str.strip, lines))
            config_str = "".join(lines)
            config = ast.literal_eval(config_str)
            
        return config
    
    
    def generate_device_token(self, n=10):
        
        random_list = [random.randint(68, 90) for _ in range(n)]
        chr_list = list(map(lambda n: chr(n), random_list))
        token = "".join(chr_list)
        
        device_token_file = self.config["device_token_file"]
        
        with open(device_token_file, "w") as f:
            f.write(token)
        
        return token
    
    
    def register_device(self, activation_code):
        
        url = self.url 
        r = requests.post(url, data = {"register_device": True, 
                                       "activation_code": activation_code})
        #print(r.status_code)
        if(r.status_code == 200):
           # content = r.content.decode().replace("true", "True").replace("false", "False")
            #response = ast.literal_eval(content)
            return True
        
        else:
            return False
            
            


    def run(self):
        
        print("Running device installer")
        
        while(True):
            activation_code = input("Enter room activation code: ")
            success = self.register_device(activation_code)
            if(success):
                break
            
        print("Device has been successfully registered")
        
        


Installer().run()



