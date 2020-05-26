import os
import pandas as pd  
from sklearn.utils import shuffle  # random
import pdb


CSV_DIR = "../data_list"
IMAGE_DIR = r"C:/Users/QIU/Desktop/dataset/image2latex100k/formula_images/formula_images/"
LABEL_DIR = r"C:/Users/QIU/Desktop/dataset/image2latex100k/"

TRAIN_LABEL_DIR = LABEL_DIR + "im2latex_train.lst"
TEST_LABEL_DIR = LABEL_DIR + "im2latex_test.lst"
VAL_LABEL_DIR = LABEL_DIR + "im2latex_validate.lst"
FORMULAS_PATH = LABEL_DIR + "im2latex_formulas.lst"


def get_data_dir(label_dir):

    image_list = []
    label_list = []
    with open(label_dir, 'r') as f:
        lines = f.readlines()
    for line in lines:
        idx, image_name = line.split(' ')[:2]
        image_path = IMAGE_DIR + image_name + '.png'
        if os.path.exists(image_path):
            image_list.append(image_path)
            label_list.append(idx)
    assert len(image_list) == len(label_list)
    return image_list, label_list


def make_datalist():
    train_image_list, train_label_list = get_data_dir(TRAIN_LABEL_DIR)
    test_image_list, test_label_list = get_data_dir(TEST_LABEL_DIR)
    val_image_list, val_label_list = get_data_dir(VAL_LABEL_DIR)

    train_data = pd.DataFrame({'image': train_image_list, 'label': train_label_list})
    train_data = shuffle(train_data)
    train_data.to_csv(CSV_DIR + "/train.csv", index=False)

    test_data = pd.DataFrame({'image': test_image_list, 'label': test_label_list})
    test_data = shuffle(test_data)
    test_data.to_csv(CSV_DIR + "/test.csv", index=False)

    val_data = pd.DataFrame({'image': val_image_list, 'label': val_label_list})
    val_data = shuffle(val_data)
    val_data.to_csv(CSV_DIR + "/val.csv", index=False)


if __name__ == '__main__':
    make_datalist()






