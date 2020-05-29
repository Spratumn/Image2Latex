import torch
import os
import torch.nn as nn
import pdb
import cv2
import matplotlib.pyplot as plt
from model.model import create_model, load_model, save_model
from config import Config

cfg = Config()

from utils.image_process import ImageProcess
from utils.label_process import LabelProcess


class LatexRec:
	def __init__(self):
		self.model = create_model()
		# self.model = load_model(self.model, path)
		self.image_processor = ImageProcess()
		self.label_processor = LabelProcess()

	def reco_formula(self, image_path):
		image = cv2.imread(image_path)
		image = self.image_processor.crop_image(image)
		image = image.unsqueeze(0)
		tokens = self.model(image)[0]
		formula = self.label_processor.tokens2formula(tokens)
		return formula


if __name__ == '__main__':
	rec = LatexRec()
	formula = rec.reco_formula('1a00b6791d.png')
	print(formula)






