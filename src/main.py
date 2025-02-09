from Functions import *
import argparse
import warnings
import tensorflow as tf
import os
import time
import matplotlib.pyplot as plt
from keras.utils import to_categorical

def parse():
    parser = argparse.ArgumentParser()  # analizador de argumentos

    grupo = parser.add_mutually_exclusive_group()  # grupo mutuamente excluyente (solo una operacion)

    grupo.add_argument('-e', '--eval', help='Muestra un resumen del modelo generado.',action='store_true')  # action guarda el argumento
    grupo.add_argument('-p', '--pred', help='Predice el resultado de una  nueva imagen de rayos X.', action='store_true')


    parser.add_argument('-i','--images', help='Lista de imágenes a evaluar.',type=str)
    parser.add_argument('-l','--labels', help='Lista de categorías de las imágenes.',default='P,N',type=str)

    return parser.parse_args()


def evaluate():
    print('Loading data...')
    X_test, y_test = importingdata('/Users/molins/Desktop/final-project/src/', 'test')
    print('Loading model...')
    model = loading_model('/Users/molins/Desktop/final-project/output/cnn-chest-x-ray.h5')
    print('The model structure:')
    model.summary()
    print('Processing model...')
    predictions, matrix= processing_model(model, X_test, y_test)
    f1, recall, prec, auc = calculate_metrics(to_categorical(y_test),predictions)
    print('\nF1: {}\n\nRecall: {}\n\nPrecision: {}\n\nAUC: {}\n'.format(f1, recall, prec, auc))
    time.sleep(3)
    plt.show()



def predict(list,labels):
    print('Scanning images...')
    try:
        X,y = load_new_image(list,200)
        plot_images(X,y,len(y))
        print('Studying if is something wrong...')
        predictions=predict_new_images('/Users/molins/Desktop/final-project/output/cnn-chest-x-ray.h5', X, y)
        plotting_predictions(predictions,y,len(y))
    except:
        X,y = load_internet_image(list,labels,200)
        plot_images(X, y, len(y))
        #plt.show()
        print('Studying if is something wrong...')
        predictions=predict_new_images('/Users/molins/Desktop/final-project/output/cnn-chest-x-ray.h5', X, y)
        plotting_predictions(predictions,y,len(y))



    #print('Work in progress...')

def main():  # funcion principal
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    warnings.filterwarnings("ignore")
    tf.logging.set_verbosity(tf.logging.ERROR)
    args = parse()
    if args.eval: evaluate()
    elif args.pred:  predict(list((args.images).split(',')),list((args.labels).split(',')))
    else: print(args,'Especifica argumento de entrada -e para evaluar el modelo y -p para evaluar una nueva imagen')


if __name__=="__main__":
    main()