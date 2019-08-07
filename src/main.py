from Functions import *
import argparse
import warnings


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
    X,y=load_new_image(list,200)
    print('Studying if is something wrong...')
    predict_new_images('/Users/molins/Desktop/final-project/output/cnn-chest-x-ray.h5',X,y)
    #print('Work in progress...')

def main():  # funcion principal
    warnings.filterwarnings("ignore")
    args = parse()
    print(args.images)
    if args.eval: evaluate()
    elif args.pred:  predict(list(args.images))
    else: print(args,'Especifica argumento de entrada -e para evaluar el modelo y -p para evaluar una nueva imagen')


if __name__=="__main__":
    main()