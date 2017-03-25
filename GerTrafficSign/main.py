from parse import parse, final_file
from utils import split_data
from model import train_model
import os

data_path = "GTSRB/Final_Training/Images/"
dataset_file = "dataset.npz"


if __name__ == "__main__":
    parse(data_path)

    split_data(os.path.join(data_path, final_file), os.path.join(data_path, dataset_file))

    print train_model(os.path.join(data_path, dataset_file))
