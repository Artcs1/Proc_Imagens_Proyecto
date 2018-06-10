**Número Usp:** 10655837

**Área do projeto:** Aprendizado de Características

**Contextualização do Problema :** 

cada ano o mercado de produção de batata sofre perda devido a infestação de pragas, uma estimativa da Organização das Nações Unidas para Alimentação e Agricultura (FAO), do 20% até 40% de perda na produção. Se essas pragas forem detectadas a tempo, podem ser tomadas medidas preventivas para mitigar a as perdas  produção e econômicas. Tradicionalmente, a detecção dessas pragas é feita por um especialista humano, mas nem todos os produtores podem cobrir as despesas. Assim, o objetivo deste projeto é a detecção e classificação de doenças da batata causadas por pragas.


 O tipo de imagens a ser utilizado são separados em 3 grupos: Imagens de folhas de batata em boas condições, folha de batata Imagens afectada por Phytophthora infestans (pragas avançado), as imagens da folha da batata afectada por Alternaria solani(praga leve) , as imagenes foram obtidos de Plant_village_Dataset [link aqui](https://github.com/spMohanty/PlantVillage-Dataset/tree/master/raw/color) são imagens no formato RGB com dimensões de 256x256.

![alt text](https://raw.githubusercontent.com/Artcs1/Proc_Imagens_Proyecto/master/tipo_de_hojas.png)

**Objetivo do projeto:** Detecção e classificação de doenças da batata

**Etapas:**

*  Segmentação:
    * Segmentação da folha com o fundo: 
        
        * Use uma máscara para separar a folha do fundo, usando un algoritmo automático que calcula o threshold como [Otsu](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4310076)

    * Segmentação das áreas afetadas da folha: 

        * Elimine as regiões verdes da imagem tomando como princípio que elas representam áreas em bom estado. Métodos possíveis: Trabalhar com o canal verde da imagem,Separe as regiões verdes por meio de umbralización .Também é possível trabalhar na escala L* a* b* .

* Extração de recursos:

    * Na imagem resultante serão calculados os descritores tanto na cor quanto na textura(Ej: Euclidean Distance ,Logarithmic Distance,  Mean , Kurtosis ,standard deviation , Entropy , etc)

    *Reduzir a dimensionalidade das características (Opcional): Avaliar diferentes conjuntos de recursos ou aplicar principal component analysis(PCA) , para encontrar a melhor combinatório dos descritores

* Separação de dados (métodos possíveis):
   
    * Hold - Out
    * K-fold Cross Validation

* Classificação: 

    * É utilizado um algoritmo de classificação que receberá como entrada Se utiliza un algoritmo de clasificación que recibirá como entrada os features extraídos e dados rotulados e irá produzir um classificador. Algoritmos possíveis a serem testados:

        
       * KNN
       * MLP

* Teste:
     *  A eficiência será medida com uma medida de erro quadrático para cada classificador com o conjunto de teste
