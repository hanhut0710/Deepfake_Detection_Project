import torch
import os
from pathlib import Path

Project_ROOT = Path(__file__).parent

Data_DIR = Project_ROOT / "data" / "dataset_faces"

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
        self.TRAIN_DIR = Data_DIR / "train"
        self.VAL_DIR = Data_DIR / "val"
        self.TEST_DIR = Data_DIR / "test"




 