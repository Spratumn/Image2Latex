import numpy as np
import cv2
import imgaug.augmenters as iaa
import imgaug as ia

from config import Config

cfg = Config()


class ImageProcess:
	def __init__(self):
		pass

	def crop_image(self, image):
		# crop [320,1600]
		crop_img = image[285:605, 50:1650, :]
		return crop_img


