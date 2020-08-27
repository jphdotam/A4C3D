import os
from glob import glob
import torch

MODEL_DIR = "/home/james/data/a4c3d/models/004"

model_paths = glob(os.path.join(MODEL_DIR, "*.pt"))

for model_path in model_paths:
    model = torch.load(model_path)['model']
    torch.save(model, model_path+'.model')
