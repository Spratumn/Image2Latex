#coding:utf-8
import pandas as pd
import numpy as np
import cv2
import torchvision.transforms as transforms
from torch.utils import data
import imgaug.augmenters as iaa
import imgaug as ia

from utils.image_process import ImageProcess
from utils.label_process import LabelProcess
from config import Config

cfg = Config()


class LatexDataset(data.Dataset):
    def __init__(self, csv_file='test.csv'):
        super(LatexDataset, self).__init__()
        # read csv as data index
        self.data_paths = pd.read_csv((cfg.CSV_DIR + '/' + csv_file),
                                      header=None,
                                      names=["image", "label"])
        self.image_paths = self.data_paths["image"].values[1:]
        self.label_paths = self.data_paths["label"].values[1:]
        with open(cfg.FORMULAS_DIR, 'r', encoding="latin_1") as f:
            self.formulas_lines = f.readlines()
        self.transformer = transforms.Compose([transforms.ToTensor()])
        self.image_processor = ImageProcess()
        self.label_processor = LabelProcess()

    def __len__(self):
        return len(self.label_paths)

    def __getitem__(self, index):
        image = cv2.imread(self.image_paths[index])
        formula_index = self.label_paths[index]
        formula = self.formulas_lines[int(formula_index)]
        image = self.image_processor.crop_image(image)
        formula_tensor = self.label_processor.formulas2tensor(formula)

        return image, formula_tensor





