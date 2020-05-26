import numpy as np
import cv2
import pandas as pd
import pdb
import matplotlib.pyplot as plt


CSV_PATH = "../data_list/train.csv"

"""
read all image then add all section pixel together
"""


def get_formula_section():
    data_paths = pd.read_csv(CSV_PATH,
                             header=None,
                             names=["image", "label"])
    image_paths = data_paths["image"].values[1:]
    h, w, c = cv2.imread(image_paths[0]).shape
    formula_section = np.zeros((h, w))
    i = 0
    for image_path in image_paths:
        image = cv2.imread(image_path, 0)
        formula_section[image > 0] += 1
        i += 1
        if i == 200:
            break
    plt.imshow(formula_section)
    plt.show()
    cv2.imwrite('formula_section.png', formula_section)


if __name__ == '__main__':
    get_formula_section()
    # section = cv2.imread('formula_section.png')
    # plt.imshow(section)
    # plt.show()

