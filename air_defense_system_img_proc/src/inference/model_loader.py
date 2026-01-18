#model yüklemek ve test etmek için yazılmıştır
from ultralytics import YOLO
import os

#yapıcı metot
class DetectorModel:
    def __init__ (self, model_path, device="cpu"): #bilgisayarda çalışırken default cpu 
        self.model_path=model_path
        self.device=device
        self.model=None
 
    #modelin olup olmadığını kontrol eder eğer varsa nesne yapıp modele atar yoksa uygun hata verir
    def load_model(self):
        if not os.path.exists(self.model_path):
            print(f"[ERROR] Model not found:{self.model_path}")
            return False
        

        self.model=YOLO(self.model_path)
        print("[INFO] Model loaded succesfully")
        return True
    
    #modelin yüklü olup olamdığını ayrıyetten kontro etmek için  boş değilse true  
    def is_loaded(self):
        return self.model is not None


#eklenecekler; !!!! device