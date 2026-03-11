import torch
import os

# Cấu hình chung
class Config:
    def __init__ (self, model_name=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model parameters
        self.Batch_size = 16
        self.Num_workers = 2
        self.pin_memory = True if torch.cuda.is_available() else False
        self.Learning_rate = 1e-4
        self.Epochs = 20
        self.IMAGE_SIZE = (224, 224)
        self.MODEL_NAME = model_name or None

        # Paths
        self.TRAIN_DIR = r"src\data\face\train"
        self.VAL_DIR = r"src\data\face\val"
        self.TEST_DIR = r"src\data\face\test"




 