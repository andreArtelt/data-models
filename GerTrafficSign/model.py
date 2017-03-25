import numpy as np
from utils import load_data, score
from keras.utils import visualize_util
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adadelta, Adam
import matplotlib.pyplot as plt


def preprocess(x, y):
    # Normalize and convert to to one-hot encoding
    return x / 255.0, to_categorical(y)


def dummy_model(x_train, y_train, x_valid, y_valid, x_test):
    return np.zeros(x_test.shape[0])


def mlp_model(x_train, y_train, x_valid, y_valid, x_test):
    save_model_path = "mlp_model.h5"
    nClass = y_train.shape[1]

    # Flatten input
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1]*x_train.shape[2]))
    x_valid = np.reshape(x_valid, (x_valid.shape[0], x_valid.shape[1]*x_valid.shape[2]))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1]*x_test.shape[2]))

    # Build model
    model = Sequential()
    model.add(Dense(600, init='he_normal', input_dim=x_train.shape[1]))   # Input: 32x32 = 1024
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())

    model.add(Dense(200, init='he_normal'))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))

    model.add(Dense(nClass, init='he_normal'))
    model.add(Activation('softmax'))

    opt = Adadelta(lr=1.2, rho=0.95, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    # Callbacks
    early_stopping = EarlyStopping(patience=6)  # Stop after 6 epochs with no improvement
    model_checkpoint = ModelCheckpoint(filepath="mlp_model_checkpoint.h5", monitor='val_acc', save_best_only=True,
                                       mode='auto')
    reduce_learningrate = ReduceLROnPlateau(monitor='val_loss', factor=0.5, verbose=1, epsilon=0.0001, patience=2,
                                            min_lr=0.001)

    # Fit/Train model
    model.fit(x_train, y_train, nb_epoch=100, batch_size=128, shuffle=True, validation_data=(x_valid, y_valid),
              callbacks=[early_stopping, model_checkpoint, reduce_learningrate])

    # Save model params
    model.save_weights(save_model_path, overwrite=True)

    # Predict labels for test set
    #model.predict(x_test, batch_size=128, verbose=1)
    return model.predict_classes(x_test, batch_size=128, verbose=1)


def cnn_model(x_train, y_train, x_valid, y_valid, x_test):
    save_model_path = "cnn_model.h5"
    nClass = y_train.shape[1]

    # Reshape input
    x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1], x_train.shape[2]))
    x_valid = np.reshape(x_valid, (x_valid.shape[0], 1, x_valid.shape[1], x_valid.shape[2]))
    x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1], x_test.shape[2]))

    # Build model
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same',
                            input_shape=(1, x_train.shape[2], x_train.shape[3]), init="he_normal"))
    model.add(MaxPooling2D((2, 2), border_mode='same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Convolution2D(16, 3, 3, activation='relu', border_mode='same', init="he_normal"))
    model.add(MaxPooling2D((2, 2), border_mode='same'))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dropout(0.2))

    model.add(Dense(128, init="he_normal"))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    #model.add(Dropout(0.1))

    model.add(Dense(nClass, init='he_normal'))
    model.add(Activation('softmax'))

    opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    # Callbacks
    early_stopping = EarlyStopping(patience=6)  # Stop after 6 epochs with no improvement
    model_checkpoint = ModelCheckpoint(filepath="cnn_model_checkpoint.h5", monitor='val_acc', save_best_only=True,
                                       mode='auto')
    reduce_learningrate = ReduceLROnPlateau(monitor='val_loss', factor=0.5, verbose=1, epsilon=0.0001, patience=2,
                                            min_lr=0.001)

    # Fit/Train model
    model.fit(x_train, y_train, nb_epoch=35, batch_size=128, shuffle=True, validation_data=(x_valid, y_valid),
              callbacks=[early_stopping, model_checkpoint, reduce_learningrate])

    # Save model params
    model.save_weights(save_model_path, overwrite=True)

    # Predict labels for test set
    # model.predict(x_test, batch_size=128, verbose=1)
    return model.predict_classes(x_test, batch_size=128, verbose=1)


def plot_2img(img1, img2):
    ax = plt.subplot(121)
    plt.imshow(img1.reshape(32, 32))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(122)
    plt.imshow(img2.reshape(32, 32))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    plt.show()


def train_model(data_path):
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_data(data_path)

    X_train, y_train = preprocess(X_train, y_train)
    X_valid, y_valid = preprocess(X_valid, y_valid)
    X_test, _ = preprocess(X_test, y_test)

    #y_pred = mlp_model(X_train, y_train, X_valid, y_valid, X_test)
    y_pred = cnn_model(X_train, y_train, X_valid, y_valid, X_test)

    #y_pred = dummy_model(X_train, y_train, X_valid, y_valid, X_test)

    return score(y_pred, y_test)

