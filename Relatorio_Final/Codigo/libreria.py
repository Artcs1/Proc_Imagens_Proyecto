
import numpy as np
import matplotlib.pyplot as plt

def graficar(datos):
	plt.figure(1)
	x = range(len(datos))
	plt.xticks([0, 50, 100, 150, 200, 255], [0, 50, 100, 150, 200, 255])
	plt.bar(x, datos, align='center')
	plt.title("Histograma")
	plt.xlabel("Valores de intensidad")
	plt.ylabel('Numero de pixeles')

	return None

def rgb2xyz(I):
    M =np.array([[0.412453, 0.357580, 0.180423],
                 	[0.212671, 0.715160, 0.072169],
                 [0.019334, 0.119193, 0.950227]]) # matrix M
    arr = I/255 # Normalização
    
    
    mask = arr > 0.04045  
    arr[mask] = np.power((arr[mask] + 0.055) / 1.055, 2.4)
    arr[~mask] /= 12.92 

    return arr @ M.T.copy() # Multiplicação

def xyz2lab(I):
    
    M = (0.95047, 1., 1.08883) # Vetor de ilumação
    I = I/M
    
    mask = I > 0.008856
    I[mask] = np.power(I[mask], 1. / 3.)
    I[~mask] = 7.787 * I[~mask] + 16. / 116.

    x, y, z = I[:,:, 0], I[:,:, 1], I[..., 2]

    #Scalação
    L = (116. * y) - 16.
    a = 500.0 * (x - y)
    b = 200.0 * (y - z)
    
    return np.concatenate([x[..., np.newaxis] for x in [L, a, b]], axis=-1)

def rgb2lab(I):
    return xyz2lab(rgb2xyz(I)) 

def rgb2hsv(I):
    """
    Values of HSV -> 0 a  1  
    R' = R/255
    G' = G/255
    B' = B/255
    Cmax = max(R',G',B')
    Cmin = min(R',G',B')
    delta = Cmax - Cmin
    H
    ((G'-B')/delta) , Cmax = R'
    ((B'-R')/delta + 2) , Cmax = G'
    ((R'-G')/delta + 4) , Cmax = B'
    S = delta/Cmax
    V = Cmax
    """
    I = np.asanyarray(I) # Conversão para array
    I = I/255 # Conversão dos valores para escala de 0 a 1
    
    out = np.empty_like(I) # Criação do array output que vai a ser a salida 
        
    Cmax = I.max(-1)
    Cmin = I.min(-1)
    
    # Espaçio de cor V
    V = Cmax
    
    # Espacio de cor S
    old_settings = np.seterr(invalid='ignore')
    delta = Cmax-Cmin
    S = delta / V
    S[delta == 0.0] = 0.0

    # Espacio de cor H
    # R'= Cmax
    idx = (I[:, :, 0] == V)
    out[idx, 0] = (I[idx, 1] - I[idx, 2]) / delta[idx]

    # G' = Cmax
    idx = (I[:, :, 1] == V)
    out[idx, 0] = 2.0 + (I[idx, 2] - I[idx, 0]) / delta[idx]

    # B' = Cmax
    idx = (I[:, :, 2] == V)
    out[idx, 0] = 4.0 + (I[idx, 0] - I[idx, 1]) / delta[idx]
    
    H = (out[:, :, 0] / 6.0) % 1.0 #normalizaçao
    H[delta == 0.0] = 0.0

    np.seterr(**old_settings)
    # Salida
    out[:, :, 0] = H
    out[:, :, 1] = S
    out[:, :, 2] = V

    # Removendo Nan
    out[np.isnan(out)] = 0
    return out

def Otsu(I,nbins = 256):
    
    I = I.reshape(-1) # Conversão a vetor
    H,bin_c = np.histogram(I,nbins) # Calculando o histogram
    bin_c = (bin_c[:-1] + bin_c[1:]) / 2. # Operacão para poder computar
    H.astype(float)
    
    # Weights
    weight1 = np.cumsum(H)
    weight2 = np.cumsum(H[::-1])[::-1]
    
    #Means
    mean1 = np.cumsum(bin_c*H)/weight1
    mean2 = (np.cumsum((bin_c*H)[::-1])/weight2[::-1])[::-1]
    
    #Variance
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2 # maximizando a beetween class
    idx = np.argmax(variance12)
    threshold = bin_c[:-1][idx]
    return threshold


def fill_holes(I,structure):
    """
        A ideia preenche os extremos com uns, 
        e expande das extremos para a dentro hasta que tope con a máscara,el complemento de isso
        vai a ser o resultado
    """
    structure.astype(bool) # Conversão a bool
    mask = np.logical_not(I) 
    tmp = np.zeros(mask.shape,bool)

    
    output = bin_dilationT(tmp,structure)
    output = output * mask 
    while ( not  (( tmp == output ).all() )): # Enquanto sean diferentes
        tmp = output
        output = bin_dilationT(tmp,structure)
        output = output * mask 
    
    np.logical_not(output, output) # complemento
    return output
    

def bin_dilationT(I,structure):
    
    structure.astype(bool) # Conversão a bool
    out = np.empty_like(I)
    out.astype(bool)
    
    # Calculo das dimensiones do padding
    f1 = structure.shape[1]//2
    f2 = ((structure.shape[1]-1)//2)
    f3 = structure.shape[0]//2
    f4 = ((structure.shape[0]-1)//2)
    
    # padding 
    M = np.concatenate((np.ones((I.shape[0],f2), dtype=bool),I,np.ones((I.shape[0],f1), dtype=bool)),axis = 1 )
    M = np.concatenate((np.ones((f4,M.shape[1]), dtype=bool),M,np.ones((f3,M.shape[1]), dtype=bool)),axis = 0 )
    
    # Recorre matriz 
    for i in np.arange(I.shape[0]):
        for j in np.arange(I.shape[1]):
            G = M[i:i+structure.shape[0],j:j+structure.shape[1]]
            F = np.sum(G * structure)
            if F == 0 : # Veja se uma estrutura de pixel corresponde à parte recortada da matriz
                out[i,j] = False
            else:
                out[i,j] = True
    
    return out
    
def bin_erosion(I,structure):
    
    structure.astype(bool) # Conversão a bool
    out = np.empty_like(I)
    out.astype(bool)
    E = np.sum(structure)

    # Calculo das dimensiones do padding
    f1 = structure.shape[1]//2
    f2 = ((structure.shape[1]-1)//2)
    f3 = structure.shape[0]//2
    f4 = ((structure.shape[0]-1)//2)
    
    # padding 
    M = np.concatenate((np.zeros((I.shape[0],f1), dtype=bool),I,np.zeros((I.shape[0],f2), dtype=bool)),axis = 1 )
    M = np.concatenate((np.zeros((f3,M.shape[1]), dtype=bool),M,np.zeros((f4,M.shape[1]), dtype=bool)),axis = 0 )
    
    # Recorre matriz 
    for i in np.arange(I.shape[0]):
        for j in np.arange(I.shape[1]):
            G = M[i:i+structure.shape[0],j:j+structure.shape[1]]
            F = np.sum(G * structure) 
            if F == E : # Veja se a estrutura corresponde à parte recortada da matriz
                out[i,j] = True
            else:
                out[i,j] = False
    
    return out

def bin_dilation(I,structure):
    
    structure.astype(bool)  # Conversão a bool
    out = np.empty_like(I)
    out.astype(bool)
    
    # Calculo das dimensiones do padding
    f1 = structure.shape[1]//2
    f2 = ((structure.shape[1]-1)//2)
    f3 = structure.shape[0]//2
    f4 = ((structure.shape[0]-1)//2)
    
    # padding 
    M = np.concatenate((np.zeros((I.shape[0],f2), dtype=bool),I,np.zeros((I.shape[0],f1), dtype=bool)),axis = 1 )
    M = np.concatenate((np.zeros((f4,M.shape[1]), dtype=bool),M,np.zeros((f3,M.shape[1]), dtype=bool)),axis = 0 )
    
    # Recorre matriz
    for i in np.arange(I.shape[0]):
        for j in np.arange(I.shape[1]):
            G = M[i:i+structure.shape[0],j:j+structure.shape[1]]
            F = np.sum(G * structure) #Veja se uma estrutura de pixel corresponde à parte recortada da matriz
            if F == 0 :
                out[i,j] = False
            else:
                out[i,j] = True
    
    return out

def bin_closing(I,structure): 
    
    # Primeiro dilation
    # Segundo erosion
    I = bin_dilation(I,structure)
    return bin_erosion(I,structure)
        
def bin_opening(I,structure):
    
    
    # Primeiro erosion
    # Segundo dilation
    I = bin_erosion(I,structure)
    return bin_dilation(I,structure)

def generate():    
    """
        Structure
        0 1 0
        1 1 1
        0 1 0
    """
    Z = np.ones((3,3),dtype = bool )
    Z[0,0] = False
    Z[0,2] = False
    Z[2,0] = False
    Z[2,2] = False
    return Z

def mask_area(I):

    imglab = rgb2lab(I) # Conversão de RGB a lab
    imglab = (imglab + [0, 128, 128]) / [100, 255, 255] # Normalizaçã0
    imglab = imglab[:,:,1] # Canal a*
   
    mask = bin_erosion(bin_closing(imglab > 0.49, np.ones((3,3))),generate()) # operaçiões morfologicas

    return mask

def mask_fundo(I):
    
    thresh = Otsu(rgb2hsv(I)[:,:,1]) # Otsu
    binary = rgb2hsv(I)[:,:,1] > thresh # umbralização
    binary = bin_closing(binary,np.ones((18,18)))   # operação morfologica
    binary = fill_holes(binary,generate()) # preencher buracos dentro da máscara
    return binary

def segmenta(I,mask1):
    
    # Aplição da máscara a uma imagen
    I[:,:,0] = I[:,:,0]*mask1
    I[:,:,1] = I[:,:,1]*mask1
    I[:,:,2] = I[:,:,2]*mask1
    
    return I