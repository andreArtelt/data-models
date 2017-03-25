import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, hamming_loss


valid_size = 0.1
test_size = 0.2


def split_data(data_path_in, data_path_out):
    data = np.load(data_path_in)
    X = data["X"]
    y = data["y"]

    X, y = shuffle(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size+valid_size)
    X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=1-valid_size)

    np.savez(data_path_out, X_train=X_train, X_valid=X_valid, X_test=X_test, y_train=y_train, y_valid=y_valid,
             y_test=y_test)


def load_data(path):
    data = np.load(path)
    return data["X_train"], data["X_valid"], data["X_test"], data["y_train"], data["y_valid"], data["y_test"]


def score(y_pred, y):
    return {'accuracy': accuracy_score(y, y_pred),
            'f1-score': f1_score(y, y_pred, average='weighted'),
            'precision': precision_score(y, y_pred, average='weighted'),
            'recall': recall_score(y, y_pred, average='weighted'),
            'hamming_loss': hamming_loss(y, y_pred)}
