import numpy as np
import cv2
import imgaug.augmenters as iaa
import imgaug as ia
import torchvision.transforms as transforms
from config import Config

cfg = Config()


class ImageProcess:
	def __init__(self):
		self.transformer = transforms.Compose([transforms.ToTensor()])

	def crop_image(self, image):
		# crop [320,1600]
		image = image[285:797, 50:1650, :]
		image = self.transformer(image)
		return image


