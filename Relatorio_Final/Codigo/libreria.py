import itertools
import numpy as np
import matplotlib.pyplot as plt
import random

from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn import neighbors
from sklearn.metrics import confusion_matrix
# Segmentacion

def graficar(dados):
	plt.figure(1)
	x = range(len(dados))
	plt.xticks([0, 50, 100, 150, 200, 255], [0, 50, 100, 150, 200, 255])
	plt.bar(x, dados, align='center')
	plt.title("Histograma")
	plt.xlabel("Valores de intensidad")
	plt.ylabel('Numero de pixeles')

	return None

def rgb2xyz(I):
    """
        I = matriz (Imagen)
    """
    M =np.array([[0.412453, 0.357580, 0.180423],
                 	[0.212671, 0.715160, 0.072169],
                 [0.019334, 0.119193, 0.950227]]) # matrix M
    arr = I/255 # Normalização
    
    
    mask = arr > 0.04045  
    arr[mask] = np.power((arr[mask] + 0.055) / 1.055, 2.4)
    arr[~mask] /= 12.92 

    return arr @ M.T.copy() # Multiplicação

def xyz2lab(I):
    """
        I = matriz (Imagen)
    """
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
    """
        I = matriz (Imagen)
    """
    return xyz2lab(rgb2xyz(I)) 

def rgb2hsv(I):
    """
        I = matriz(Imagen)
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
    """
        I = matriz (Imagen)
        nbins = int 
    """
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
        I = matriz ( imagen)
        structure = matriz (structure morfologica)
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
    """
        I = matriz (Imagen)
        structure = matriz (structure morfologica)
    """
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
    """
        I = matriz (Imagen)
        structure = matriz (structure morfologica)
    """
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
    """
        I = matriz (Imagen)
        structure = matriz (structure morfologica)
    """
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
    """
        I = matriz (Imagen)
        structure = matriz (structure morfologica)
    """
    # Primeiro dilation
    # Segundo erosion
    I = bin_dilation(I,structure)
    return bin_erosion(I,structure)
        
def bin_opening(I,structure):
    
    """
        I = matriz (Imagen)
        structure = matriz (structure morfologica)
    """
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
    """
        I = matriz (Imagen)
    """
    imglab = rgb2lab(I) # Conversão de RGB a lab
    imglab = (imglab + [0, 128, 128]) / [100, 255, 255] # Normalizaçã0
    imglab = imglab[:,:,1] # Canal a*
   
    mask = bin_erosion(bin_closing(imglab > 0.49, np.ones((3,3))),generate()) # operaçiões morfologicas

    return mask

def mask_fundo(I):
    """
        I = matriz (Imagen)
    """
    thresh = Otsu(rgb2hsv(I)[:,:,1]) # Otsu
    binary = rgb2hsv(I)[:,:,1] > thresh # umbralização
    binary = bin_closing(binary,np.ones((18,18)))   # operação morfologica
    binary = fill_holes(binary,generate()) # preencher buracos dentro da máscara
    return binary

def segmenta(I,mask1):
    """
        I = matriz (Imagen)
        mask1 = matriz booleana
    """ 
    # Aplição da máscara a uma imagen
    I[:,:,0] = I[:,:,0]*mask1
    I[:,:,1] = I[:,:,1]*mask1
    I[:,:,2] = I[:,:,2]*mask1
    
    return I

# Descriptores 


def co_ocurrence_matrix(image_data, b = 8, shift=[1,1]):
	output = np.zeros((int(2**b),int(2**b)))
    #Matriz de co-conocurrencia
    
	for x in range(image_data.shape[0]):
		for y in range(image_data.shape[1]):
			try:
				i = image_data[x , y]
				j = image_data[x + shift[0] , y + shift[1]]
				output[i , j] += 1
			except:
				pass

	prob = output/np.sum(output)

	return [output,prob]


def texturedescriptor(image_data, b = 8, shift=[1,1]):  
    
   
	[coocurrence, normal_prob] = co_ocurrence_matrix(image_data, b, shift)
  
    # Media
	u_i = 0
	for i in range(int(2**b)):
		u_i += i * np.sum(normal_prob[i , :])

	u_j = 0
	for j in range(int(2**b)):
		u_j += j * np.sum(normal_prob[: , j])

    #Sigma
	sigma_i = 0
	for i in range(int(2**b)):
		sigma_i += ((i - u_i)**2) * np.sum(normal_prob[i , :])

	sigma_j = 0
	for j in range(int(2**b)):
		sigma_j += ((j - u_j)**2) * np.sum(normal_prob[: , j])

	#Energy

	energy = np.sum(np.power(normal_prob,2))

    #Entropy
	entropy = - np.sum(normal_prob * np.log(normal_prob + 1e-32))

    #Contrast
	contrast = 0
	for i in range(int(2**b)):
		for j in range(int(2**b)):
			contrast += ((i - j) ** 2) * normal_prob[i , j]
	contrast *= (1 / ((b - 1) ** 2))

    #Correlation
	correlation = 0
	for i in range(int(2**b)):
		for j in range(int(2**b)):
			correlation += (i * j) * normal_prob[i , j]
	if (sigma_i * sigma_j) > 0:
		correlation = (correlation - (u_i * u_j)) / (sigma_i * sigma_j)
	else:
		correlation = 0

    #Homogenity
	homogenity = 0
	for i in range(int(2**b)):
		for j in range(int(2**b)):
			homogenity += (normal_prob[i , j] / (1 + (np.abs(i - j))))

	return [energy, entropy,contrast, correlation, homogenity]

def gray_scale(dim_3_matrix):
    # to gray scale
	r, g, b = dim_3_matrix[ : , : , 0 ] , dim_3_matrix[ : , : , 1 ] , dim_3_matrix[ : , : , 2 ]

	return np.floor(0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)

def histogramcolor(gray_img, b = 8):
    """
        gray_img = matriz (Imagen)
        b = int (bits)
    """    

    H = np.histogram(gray_img,range(int(2**b+1)))
    #Histograma de color
    H = H[0]
    #probabilidades do histograma de color
    prob = H / np.sum(H)
    return [H,prob]

def colordescriptor(image_data,b = 8):
    """
        image_data = matriz (Imagen)
        b = int (bits)
    """  

    [hog , normal_prob] = histogramcolor(image_data,b = 8)
    
    # Media
    U = np.sum( hog * normal_prob )
    
    #Descio Padrão
    SD = np.sqrt ( np.sum(np.power((hog - U),2)*normal_prob) )
    
    #Skewness
    S = np.sum( np.power((hog- U),3)*normal_prob )/np.power(SD,3)
     
    #Kurtosis
    K = (np.sum(np.power((hog- U),4)*normal_prob)/np.power(SD,4)) - 3  
    
    return [U,SD,S,K]


# Testing


def resize(data,siz = 152): # Resize
    """
        data = matriz(Imagen)
        siz = int
    """  

    data = data[1:]
    setA = np.arange(152)
    idx = random.sample(list(setA),siz) # Amostragem da Data
    dataset = data[idx,:]
    return dataset

def HoldOut(dataset, train_size = 0.7): # Hold Out
    """
        dataset = matriz (Imagen)
        train_set = float
    """  

    row = dataset.shape[0]
    setA = np.arange(row)
    idx = random.sample(list(setA), int(np.floor(row * train_size ))) # amostragem de dataset 
    setB = idx
    idx2 = np.setdiff1d(setA,setB) # ids do complemento
    traindata = dataset[idx,:]
    testdata = dataset[idx2,:]
    return [traindata,testdata]

def Treinamento(X,Y,C = 'SVM'): # Treinamento
    """
        X(X_train) = matriz
        Y(Y_train) = matriz
        C = 'SVM' | 'KNN ' | 'MLP'
    """   

    print(C)
    if C == 'SVM': 
        C_range = np.logspace(-2, 10, 13) 
        gamma_range = np.logspace(-9, 3, 13)
        param_grid = dict(gamma=gamma_range, C=C_range)
        cv = StratifiedShuffleSplit(n_splits=10, test_size=0.33, random_state=42)
        clf = GridSearchCV(SVC(), param_grid=param_grid,cv = cv)
    if C == 'KNN':
        clf = neighbors.KNeighborsClassifier(n_neighbors = 5, weights = 'distance')
    if C == 'MLP':
        clf = MLPClassifier(hidden_layer_sizes = 5,random_state = 97)
    
    clf.fit(X,Y)
    return clf

def Test(clf,features,output,X,Y): # Test -> accuracy , precisão , recall , f1 - medida , HoldOut
    """
        clf = clasificador
        features(X _test) = matriz
        output(Y_test) = matriz
        X(X_train) = matriz
        Y(Y_train) = matriz
    """
    pred= clf.predict(features) # predição
    ac=accuracy_score(output, pred ) # accuracy
    print('Tasa de acerto no treinamento', accuracy_score(Y, clf.predict(X) ))
    print('Tasa de acerto no test', ac)

    print('\n')
    MP = 0
    MR = 0
    Mf1 = 0
    
    for i in np.arange(3): # Calculando precisão , recall ,f1 - medida por cada clase
        c= i+1;
        print('Clase',c)
        PP = np.sum(np.logical_and(output == c, pred == c)) # Verdadero Positivo
        NP = np.sum(np.logical_and(output == c, pred != c)) # Falso Negativo
        PN = np.sum(np.logical_and(output != c, pred == c)) # Verdadero Negativo
        precision = PP/(PP+NP)
        recall = PP/(PP+PN)
        f1 = 2*(precision*recall)/(precision+recall)
        MP = MP + precision
        MR = MR + recall
        Mf1 = Mf1 + f1
        print('Precision',precision)
        print('Recall',recall)
        print('f1',f1)
        print('\n')
    
    # Promedio de M , MR , Mf1
    print('\n')
    print('Promedio de Precision',MP/3)
    print('Promedio de Recall',MR/3)
    print('Promedio de f1',Mf1/3)

    cnf_matrix = confusion_matrix(output, pred) # matriz de confusão
    plt.figure()
    class_names = ["Batata Saudavel","Batata plaga leve","Batata con plaga Tardia"]
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Matriz de confusão normalizada')

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues): # matriz de confusão

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Rótulo verdadeiro')
    plt.xlabel('Rótulo previsto')