import os
import pandas as pd
import cv2
import numpy as np


resize = (32, 32)
subdir_file = "data.npz"
final_file = "data.npz"


def get_all_dirs(path):
    return [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]


def get_all_files(path):
    return [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]


def get_csv_file(files):
    csv_files = filter(lambda x: x.find(".csv") != -1, files)
    if len(csv_files) == 1:
        return csv_files[0]
    else:
        raise Exception(".csv file not found! Please make sure each directory contains exactly one .csv file.")


def load_csv_file(file):
    return pd.read_csv(file, sep=';')


def handle_dirs(path):
    subdirs = get_all_dirs(path)    # Process all subdirectories

    X = []
    y = []

    for dir in subdirs:
        # Process files
        print dir
        handle_files(dir)

        # Append global dataset
        data = np.load(os.path.join(dir, subdir_file))
        X.append(data["X"])
        y.append(data["y"])

    # Save data
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    print X.shape
    print y.shape
    np.savez(os.path.join(path, final_file), X=X, y=y)


def handle_files(path):
    csv_file = get_csv_file(get_all_files(path))
    data = load_csv_file(csv_file)

    X = []
    y = []

    for _, row in data.iterrows():
        file_name, width, height, x1, y1, x2, y2, label = row

        # Load image
        file_path = os.path.join(path, file_name)
        img = cv2.imread(file_path)

        # "Preprocessing"
        img = img[y1:y2, x1:x2]                     # Extract region of "interest"
        img = cv2.resize(img, dsize=resize)         # resize to uniform size
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Greyscale

        #plot_img(img)

        X.append(img)
        y.append(label)

    # Save data
    X = np.array(X)
    y = np.array(y)
    np.savez(os.path.join(path, subdir_file), X=X, y=y)


def parse(path):
    handle_dirs(path)
