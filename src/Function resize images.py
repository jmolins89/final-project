from os import listdir
from os.path import isfile, join
import cv2

def resize(path,w,destiny):
    '''
    :param path: Carpeta de imágenes
    :param w: Ancho que le quieres dar a la nueva imagen
    :param destiny: carpeta destino dónde se guardarán las fotos
    :return: devuelve error si hay fotos que no se pueden formatear
    '''
    mypath = path
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    for imagen in onlyfiles:
        try:
            try:
                img = cv2.imread('{}{}'.format(path,imagen), 0)
                width=w
                height=375 #(width * img.shape[0])/img.shape[1]
                #height, width = img.shape
                #imgScale = scale
                #newX,newY = img.shape[1]*imgScale, img.shape[0]*imgScale
                newimg = cv2.resize(img,(int(width), int(height)))
                cv2.waitKey(0)
                cv2.imwrite("{}{}".format(destiny,imagen), newimg)
            except AttributeError:
                print('{} is None Type object'.format(imagen))
        except Exception as e:
            print(str(e))
    return 'Image resized'


resize('/Users/molins/Desktop/FINAL PROJECT/input/chest_xray/train/NORMAL/',300,'/Users/molins/Desktop/FINAL PROJECT/input/chest-xray-resized/train/NORMAL/')
resize('/Users/molins/Desktop/FINAL PROJECT/input/chest_xray/train/PNEUMONIA/VIRUS/',300,'/Users/molins/Desktop/FINAL PROJECT/input/chest-xray-resized/train/PNEUMONIA/VIRUS/')
resize('/Users/molins/Desktop/FINAL PROJECT/input/chest_xray/train/PNEUMONIA/BACTERIA/',300,'/Users/molins/Desktop/FINAL PROJECT/input/chest-xray-resized/train/PNEUMONIA/BACTERIA/')
resize('/Users/molins/Desktop/FINAL PROJECT/input/chest_xray/test/NORMAL/',300,'/Users/molins/Desktop/FINAL PROJECT/input/chest-xray-resized/test/NORMAL/')
resize('/Users/molins/Desktop/FINAL PROJECT/input/chest_xray/test/PNEUMONIA/VIRUS/',300,'/Users/molins/Desktop/FINAL PROJECT/input/chest-xray-resized/test/PNEUMONIA/VIRUS/')
resize('/Users/molins/Desktop/FINAL PROJECT/input/chest_xray/test/PNEUMONIA/BACTERIA/',300,'/Users/molins/Desktop/FINAL PROJECT/input/chest-xray-resized/test/PNEUMONIA/BACTERIA/')
resize('/Users/molins/Desktop/FINAL PROJECT/input/chest_xray/val/NORMAL/',300,'/Users/molins/Desktop/FINAL PROJECT/input/chest-xray-resized/val/NORMAL/')
resize('/Users/molins/Desktop/FINAL PROJECT/input/chest_xray/val/PNEUMONIA/VIRUS/',300,'/Users/molins/Desktop/FINAL PROJECT/input/chest-xray-resized/val/PNEUMONIA/VIRUS/')
resize('/Users/molins/Desktop/FINAL PROJECT/input/chest_xray/val/PNEUMONIA/BACTERIA/',300,'/Users/molins/Desktop/FINAL PROJECT/input/chest-xray-resized/val/PNEUMONIA/BACTERIA/')
