from Functions import *
import argparse
import warnings
import tensorflow as tf
import os

def parse():
    parser = argparse.ArgumentParser()  # analizador de argumentos

    grupo = parser.add_mutually_exclusive_group()  # grupo mutuamente excluyente (solo una operacion)

    grupo.add_argument('-e', '--eval', help='Muestra un resumen del modelo generado.',action='store_true')  # action guarda el argumento
    grupo.add_argument('-p', '--pred', help='Predice el resultado de una  nueva imagen de rayos X.', action='store_true')


    parser.add_argument('images', help='Lista de imágenes a evaluar.',nargs='+')

    return parser.parse_args()


def evaluate():
    print('Loading data...')
    X_test, y_test = importingdata('/Users/molins/Desktop/final-project/src/', 'test')
    print('Loading model...')
    model = loading_model('/Users/molins/Desktop/final-project/output/cnn-chest-x-ray.h5')
    #model.summary()
    print('Processing model...')
    matrix = processing_model(model, X_test, y_test)

def predict(list):
    print('Scanning images...')
    try:
        X,y = load_new_image(list,200)
        plot_images(X,y,len(y))
        print('Studying if is something wrong...')
        predictions=predict_new_images('/Users/molins/Desktop/final-project/output/cnn-chest-x-ray.h5', X, y)
        plotting_predictions(predictions,y)
    except:
        X,y = load_internet_image(list,200)
        plot_images(X, y, len(y))
        print('Studying if is something wrong...')
        predictions=predict_new_images('/Users/molins/Desktop/final-project/output/cnn-chest-x-ray.h5', X, y)
        plotting_predictions(predictions,y)


    #print('Work in progress...')

def main():  # funcion principal
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    warnings.filterwarnings("ignore")
    tf.logging.set_verbosity(tf.logging.ERROR)
    args = parse()
    if args.eval: evaluate()
    elif args.pred:  predict(list(args.images))
    else: print(args,'Especifica argumento de entrada -e para evaluar el modelo y -p para evaluar una nueva imagen')


if __name__=="__main__":
    main()