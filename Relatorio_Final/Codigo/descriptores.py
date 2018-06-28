import numpy as np
import imageio 
import libreria as lib


from os import scandir, getcwd
from os.path import abspath

def ls(ruta = getcwd()):
    return [abspath(arch.path) for arch in scandir(ruta) if arch.is_file()]

def main():
    
    A = ls('C:/Users/nineil/Desktop/Jeffri/USP/Procesamiento de Imagenes/Proyecto_PI/DataSegmentada/Experimento2/T2')# ruta absoluta
    
    X = np.ones((1,10),dtype = float);
    
    for i in A:      
        cad = i # leitura do nome da imagen
        I = imageio.imread(cad) # leitura da imagen
        des = np.concatenate((lib.texturedescriptor(I),lib.colordescriptor(I),[3]),axis = 0)
        des = np.matrix(des)
        X = np.concatenate((X,des),axis = 0)
        
    np.savetxt('C:/Users/nineil/Desktop/Jeffri/USP/Procesamiento de Imagenes/Proyecto_PI/DataDescriptor/batataT.dat', X); # ruta absoluta

main()
