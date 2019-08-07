import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
from sklearn.metrics import recall_score,f1_score,precision_score
import itertools
import pickle
from sklearn.metrics import confusion_matrix
import pandas as pd
import keras
from keras.models import Sequential,load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
from keras.layers.advanced_activations import ReLU
from keras.utils import to_categorical
from keras.callbacks import TensorBoard
from time import time
from collections import Counter



def createxy(data):
    '''
    This function generates
    :param data:
    :return:
    '''
    X, y = [], []
    for features, label in data:
        X.append(features / 255)
        y.append(label)
    return X, y


def createxyshuffled(data):
    '''

    :param data:
    :return:
    '''
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
    :return: Returns a list of lists with images (X) and labels (y)
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
    random.shuffle(lst)
    X, y = createxy(lst)
    X = np.array(X).reshape(-1, img_size, img_size, 1)
    return X,y

def plot_examples(X,y):
    dic = {0: 'NORMAL', 1: 'PNEUMONIA'}
    plt.figure(figsize=(20, 12))
    for index, img in enumerate(X[:5]):
        plt.subplot(1, 6, index + 1)
        plt.imshow(img.reshape(200, 200), cmap='gray')
        plt.axis('off')
        plt.title(dic.get(y[index]))
    plt.show()

def importingdata(path,type):
    pickle_in1=open('{}X_{}.pickle'.format(path,type),'rb')
    pickle_in2=open('{}y_{}.pickle'.format(path,type),'rb')
    X,y=pickle.load(pickle_in1),pickle.load(pickle_in2)
    return X,y


def plotting_acc_loss_evolution(model):
    '''
    This function plots the evolution of the training and validation accuracy and loss in the fitting time
    :param model: the fitted model you want to plot
    :return: two graphs
    '''
    acc = model.history['acc']
    val_acc = model.history['val_acc']
    loss = model.history['loss']
    val_loss = model.history['val_loss']
    epochs = range(len(acc))
    plt.plot(epochs, acc, 'r', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()

def processing_model(model,X_test,y_test):
    y_test_binary = to_categorical(y_test)
    predictions = model.predict(X_test)
    matrix = confusion_matrix(y_test_binary.argmax(axis=1), predictions.argmax(axis=1))
    preds = np.argmax(model.predict(X_test), axis=1)
    cm = confusion_matrix([np.argmax(i) for i in y_test_binary], preds)
    keys = ['NORMAL', 'PNEUMONIA']
    plt.figure(figsize=(8, 8))
    plot_confusion_matrix(cm, keys, normalize=True)
    return predictions, matrix

def loading_model(path):
    return load_model(path)

def load_new_image(list_data_dir,IMG_SIZE=200):
  categories = ['NORMAL', 'PNEUMONIA']
  img_list=[]
  for datadir in list_data_dir:
    img_array = cv2.imread(datadir, cv2.IMREAD_GRAYSCALE)   # resizes the original image to a IMG_SIZE
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))    # resizes the original image to a IMG_SIZE
    datadir=datadir.split('/')
    class_num=categories.index(datadir[-2])     # Set category by index in categories: 0 -> Normal, 1 -> Pneumonia
    img_list.append([new_array,class_num])              # Appends to the list a tuple with array resized and each label
  X,y = createxy(img_list)
  X = np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE,1)
  y = to_categorical(y)
  return X,y

def predict_new_images(path,X,y):
    dictionary={0:'NORMAL',1:'PNEUMONIA'}
    model=loading_model(path)
    new_predictions = model.predict(X)
    #matrix_loaded = confusion_matrix(y.argmax(axis=1), new_predictions.argmax(axis=1))
    predictions=[np.argmax(new_predictions[i]) for i in range(len(new_predictions))]
    for i in range(len(predictions)):
        if i==0:
            try:
                print('The {}st image is {} and the model predicts {} with a {:.2f}% of confidence'.format(i+1,dictionary.get(y[i][1]),dictionary.get(predictions[i]),(new_predictions[i][predictions[i]])*100))
            except:
                print('The {}st image is {} and the model predicts {} with a {:.2f}% of confidence'.format(i + 1,dictionary.get(y[i][0]), dictionary.get(predictions[i]),(new_predictions[i][predictions[i]])*100))
        else:
            print('The {}nd image is {} and the model predicts {} with a {:.2f}% of confidence'.format(i+1,dictionary.get(y[i][1]),dictionary.get(predictions[i]),(new_predictions[i][predictions[i]])*100))
    return new_predictions

def load_internet_image(list_data_dir,labels,IMG_SIZE=200):
  categories = ['NORMAL', 'PNEUMONIA','N','P']
  img_list=[]
  for datadir in list_data_dir:
    img_array = cv2.imread(datadir, cv2.IMREAD_GRAYSCALE)   # resizes the original image to a IMG_SIZE
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))    # resizes the original image to a IMG_SIZE
    #datadir=datadir.split('/')
    #print('Write the type of the X Ray image(normal or pneumonia):')
    #response=input()    # Set category by index in categories: 0 -> Normal, 1 -> Pneumonia
    #response=response.upper()
    #class_num = categories.index(str(response))
    image_label=labels[list_data_dir.index(datadir)]
    image_label=image_label.upper()
    class_num = (categories.index(str(image_label)))%2
    img_list.append([new_array,class_num])              # Appends to the list a tuple with array resized and each label
  X,y = createxy(img_list)
  X = np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE,1)
  if len(y)==1:
      if y==[1]:
          y=[[0,1]]
      else: y= [[1,0]]
  else: y = to_categorical(y)
  return X,y

def plotting_predictions(predictions,y_theoric):
    dictionary = {0: 'NORMAL', 1: 'PNEUMONIA'}
    #plt.figure(figsize=(12, 5))
    for i in range(len(predictions)):
        plt.subplot(2,len(predictions) , i + 3)
        plt.bar(['Normal', 'Pneumonia'], predictions[i], color=['g', 'r'])
        for j in range(len(predictions[i])):
            plt.text(x=j - 0.1, y=predictions[i][j] / 2, s='{:.2f} %'.format((predictions[i][j]) * 100), size=12)
        plt.title('This case is suposed to be {},\n and the model predicts:'.format(dictionary.get(y_theoric[i][1])))
    plt.show()

def plot_images(X,y,number_of_examples):
    dic = {0: 'NORMAL', 1: 'PNEUMONIA'}
    plt.figure(figsize=(12, 5))
    for index, img in enumerate(X[:number_of_examples]):
        plt.subplot(2, number_of_examples, index + 1)
        plt.imshow(img.reshape(200, 200), cmap='gray')
        plt.axis('off')
        plt.title('The {} image is a {} case'.format(index+1,dic.get(y[index][1])))
