import torch
import torch.nn as nn
import pdb

from model.model import create_model
from config import Config

cfg = Config()

latnet = create_model()
img = torch.rand(cfg.BATCH_SIZE, 3, 128, 256)
logit, pred = latnet(img)
print(logit)

print(pred)



