import numpy as np
import matplotlib.pyplot as plt
import imageio 
import libreria as lib

def main():
    
    cad = "C:/Users/nineil/Desktop/prueba.jpg" # leitura do nome da imagen
    
    I = imageio.imread(cad) # leitura da imagen
    
    
    MaskFundo = lib.mask_fundo(I.copy()) # Masara que segmenta do fundo con a folha
    MaskArea = lib.mask_area(I.copy()) # Mascara que separa as áreas verdes da folha
    IFundo = lib.segmenta(I.copy(),MaskFundo) # Segmentação con o fundo
    IArea = lib.segmenta(I.copy(),MaskArea) # Segmentação con a area
    IResult = lib.segmenta(I.copy(),MaskFundo*MaskArea) # Segmentação Final
    
    fig, ax = plt.subplots(ncols=3, sharex=True, sharey=True,figsize=(10, 5))
        
    ax[0].imshow(IFundo, cmap=plt.cm.gray)
    ax[0].set_title('Segmentação da folha com o fundo')
        
    ax[1].imshow(IArea, cmap=plt.cm.gray)
    ax[1].set_title('Segmentação das áreas afetadas da folha')
        
    ax[2].imshow(IResult, cmap = plt.cm.gray)
    ax[2].set_title('Segmentacion final')
        
        
    for a in ax:
       a.axis('off')
        
    plt.tight_layout()
    plt.show()
    

main()