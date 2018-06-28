
from sklearn import preprocessing
import numpy as np

import libreria as lib
def main():
    
    D1 = np.loadtxt('C:/Users/nineil/Desktop/Jeffri/USP/Procesamiento de Imagenes/Proyecto_PI/DataDescriptor/BatataH.dat') # Ruta dos descriptores da batata saudavel 
    D2 = np.loadtxt('C:/Users/nineil/Desktop/Jeffri/USP/Procesamiento de Imagenes/Proyecto_PI/DataDescriptor/BatataE.dat') # Ruta dos descriptores da batata com plaga leve 
    D3 = np.loadtxt('C:/Users/nineil/Desktop/Jeffri/USP/Procesamiento de Imagenes/Proyecto_PI/DataDescriptor/BatataT.dat') # Ruta dos descriptores da batata com plaga tardia 
    
    D1 = lib.resize(D1) 
    D2 = lib.resize(D2) 
    D3 = lib.resize(D3) 
    
    Data = np.concatenate((D1,D2,D3),axis = 0 ) # Concatenando as matrices
    
    DataH = lib.HoldOut(Data) # HoldOut 
    Train = DataH[0] 
    Test = DataH[1]
    
    X = Train[:,0:9] # X treinamento
    X = preprocessing.scale(X) # Normalização
    Y = Train[:,9].astype(int) # Y treinamento
    
    features = Test[:,0:9] # X Test
    features = preprocessing.scale(features) # Normalização
    output = Test[:,9].astype(int) # Y test

    clf = lib.Treinamento(X,Y,'MLP') # escolhiendo classificador 
    Test = lib.Test(clf,features,output,X,Y) # Test


main()

