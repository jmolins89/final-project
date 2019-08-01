from os import listdir
from os.path import isfile, join
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#img = cv2.imread('NORMAL2-IM-0960-0001.jpeg', 0)
def addimagesdf(df,test_train_val,type,code_type,counter):
    '''
    :param df: df al que añadir imágenes
    :param test_train_val: Carpeta de la que importar: test,train, val
    :param type: tipo de neumonía:NORMAL,BACTERIA,VIRUS
    :param code_type: 0-NORMAL,1-BACTERIA,2-VIRUS
    :param counter: Indicar el primer valor desde el que empieza el contador
    :return: Devuelve el df con las columnas añadidas, devuelve el valor desde el que empezará el siguiente contador
    '''
    mypath = '/Users/molins/Desktop/FINAL PROJECT/chest-xray-resized/{}/{}/'.format(test_train_val,type)
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    counter=counter
    for imagen in onlyfiles:
        try:
            try:
                img = cv2.imread('{}{}'.format(mypath, imagen), 0)
                lst = []
                for i in img:
                    for j in i:
                        lst.append(j)
                lst.append(0)
                df[counter] = lst
                counter += 1
            except AttributeError:
                print('{} is None Type object'.format(imagen))
        except Exception as e:
            print(str(e))
    return df,counter

def generate_csv(df,train_test_val):
    df = pd.DataFrame()
    df,counter = addimagesdf(df,train_test_val,'NORMAL',0,0)
    df,counter1 = addimagesdf(df,train_test_val,'PNEUMONIA/BACTERIA',counter,1)
    df,counter2 = addimagesdf(df,train_test_val,'PNEUMONIA/VIRUS',counter,2)
    df = df.T
    df.to_csv('/Users/molins/Desktop/FINAL PROJECT/images_{}.csv'.format(train_test_val),index=False)
    return df

generate_csv()



#arr = np.reshape(img, (1,img.shape[0]*img.shape[1]))


