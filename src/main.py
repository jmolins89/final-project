from Functions import *
import argparse
import warnings


def parse():
    parser = argparse.ArgumentParser()  # analizador de argumentos

    grupo = parser.add_mutually_exclusive_group()  # grupo mutuamente excluyente (solo una operacion)

    grupo.add_argument('-e', '--eval', help='Muestra un resumen del modelo generado.',action='store_true')  # action guarda el argumento
    grupo.add_argument('-p', '--pred', help='Predice el resultado de una  nueva imagen de rayos X.', action='store_true')
    #grupo.add_argument('-m', '--mult', help='Realiza la multiplicacion de dos numeros.', action='store_true')
    #grupo.add_argument('-d', '--div', help='Realiza la division de dos numeros.', action='store_true')

    #parser.add_argument('images', help='Lista de imágenes a evaluar.', type=list)
    #parser.add_argument('n2', help='Segundo numero de la operacion.', type=float)

    return parser.parse_args()


def evaluate():
    print('Loading data...')
    X_test, y_test = importingdata('/Users/molins/Desktop/final-project/src/', 'test')
    print('Loading model...')
    model = loading_model('/Users/molins/Desktop/final-project/output/cnn-chest-x-ray.h5')
    #model.summary()
    print('Processing model...')
    matrix = processing_model(model, X_test, y_test)

def predict():
    X,y=load_new_image(['/Users/molins/Desktop/final-project/input/chest_xray/val/NORMAL/NORMAL2-IM-1431-0001.jpeg','/Users/molins/Desktop/final-project/input/chest_xray/val/PNEUMONIA/person1954_bacteria_4886.jpeg'],200)
    predict_new_images('/Users/molins/Desktop/final-project/output/cnn-chest-x-ray.h5',X,y)
    print('Work in progress...')

def main():  # funcion principal
    warnings.filterwarnings("ignore")
    args = parse()

    if args.eval: evaluate()
    elif args.pred:  predict()
    else: print(args,'Especifica argumento de entrada -e para evaluar el modelo y -p para evaluar una nueva imagen')


if __name__=="__main__":
    main()