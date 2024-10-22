import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
from sklearn.metrics import recall_score,f1_score,precision_score
import itertools
import pandas as pd
import keras
import tensorflow as tf
from keras.callbacks import TensorBoard
from time import time
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
from keras.layers.advanced_activations import ReLU
from collections import Counter
from keras.models import load_model


def createxy(data):
    X, y = [], []
    for features, label in data:
        X.append(features / 255)
        y.append(label)
    return X, y


def createxyshuffled(data):
    X, y = [], []
    for features, label in data:
        X.append(features)
        y.append(label)
    return X, y


def calculate_metrics(y_true, y_predictions):
    '''

    :param y_true: Specify the truth of labels for the data
    :param y_predictions: Specify the predicted labels with the model
    :return: Return the value of F1, recall, precision and AUC
    '''
    y_trues = np.array([np.argmax(y_true[i]) for i in range(len(y_true))])
    y_pred = np.array([y_predictions[i][np.argmax(y_predictions[i])] for i in range(len(y_predictions))])
    y_t = [np.argmax(y_true[i]) for i in range(len(y_true))]
    y_p = [np.argmax(y_predictions[i]) for i in range(len(y_predictions))]
    f1 = f1_score(y_t, y_p, average='binary')
    recall = recall_score(y_t, y_p, average='binary')
    prec = precision_score(y_t, y_p, average='binary')
    fpr, tpr, thresholds = metrics.roc_curve(y_trues, y_pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return f1, recall, prec, auc


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    :param cm: confusion matrix values
    :param classes:
    :param normalize:
    :param title:
    :param cmap:
    :return:
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)  # , rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def create_training_data(datadir, img_size):
    '''
    
    :param datadir: Specify the path to load.
    :param img_size: Specify the size of the image to reshape.
    :return: Returns a list of lists with images and labels
    '''
    lst = []
    categories = ['NORMAL', 'PNEUMONIA']
    for category in categories:
        path = os.path.join(datadir, category)  # path to normal or pneumonia
        class_num = categories.index(category)  # Set category by index in categories: 0 -> Normal, 1 -> Pneumonia
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # read the original image
                new_array = cv2.resize(img_array, (img_size, img_size))  # resizes the original image to a IMG_SIZE
                lst.append([new_array, class_num])  # Appends to the list a tuple with array resized and each label
            except Exception as e:
                pass
    return lst