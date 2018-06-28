import numpy as np
import matplotlib.pyplot as plt
import imageio 
import libreria as lib

from skimage.color import rgb2lab
from skimage.morphology import binary_closing,binary_erosion
from os import scandir, getcwd
from os.path import abspath

def ls(ruta = getcwd()):
    return [arch.name for arch in scandir(ruta) if arch.is_file()]


cad = 'C:/Users/nineil/Desktop/Jeffri/USP/Procesamiento de Imagenes/Proyecto_PI/DataSegmentada/Opcional/plagaT/'
A = ls(cad)

for i in A:
    print(i)
    C = cad + i
    I = imageio.imread(C)
    imglab = rgb2lab(I) # Conversão de RGB a lab
    imglab = (imglab + [0, 128, 128]) / [100, 255, 255] # Normalizaçã0
    imglab = imglab[:,:,1] # Canal a*
   
    mask = binary_erosion(binary_closing(imglab > 0.49, np.ones((3,3)))) # operaçiões morfologicas
    R = lib.segmenta(I,mask)
    D = 'C:/Users/nineil/Desktop/Jeffri/USP/Procesamiento de Imagenes/Proyecto_PI/DataSegmentada/Experimento/T/'+i
    imageio.imwrite(D,R)