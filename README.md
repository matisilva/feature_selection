# Reducción de dimensionalidad

# Objetivo:
Aplicar diversas técnicas de selección de features para reducir la 
dimensionalidad de los vectores generados.

En una primera instancia se aplicaran metodos no supervisados para la reducción
de dimensionalidad, es decir basandose solo en la varianza de los features o 
influencia en el resultado.

Por otro lado también se aplicarán técnicas de selección de features basandose
en filtrar aquellos K features que sean determinantes para la clasificación
objetivo. Esta clasificación será provista como entrenamiento para que el 
selector pueda determinar la relevancia de cada feature. Se utilizarán dos
clasificaciones como vectores objetivo, la etiqueta **POS** y en una segunda
iteración será el **sentido basado en Wordnet**.

# WikiCorpus
Para los experimentos se utilizará una muestra aleatoria del WikiCorpus en 
español y se le aplicarán ambas técnicas.
Debido a que este corpus ya posee etiquetado, solo se realizara un parseo
y normalizacion de texto sobre el mismo.
La **muestra aleatoria** constará en seleccionar de cada archivo taggeado una
porción de hasta 10000 oraciones. [¿Cómo?](random_sentences.py)

# Preproceso de wikicorpus
Se retiraron las lineas no relevantes como los indicadores de principio y fin
de documento. Tambien se evitaron analizar signos de puntuacion o cualquier 
palabra cuyo tag comience con F(indicador de signos). Las palabras de longitud
menor a 4 o con menos de 150 menciones tambien fueron evitadas. Por ultimo
se removieron palabras sin sentido para analisis tales como *endofarticle*. 

# Análisis general de reducción de dimensionalidad.
Como generalidad de la reducción de dimensionalidad ya sea supervisada o no,
vimos que impacta positivamente en el rendimiento de clustering. Se reduce
muchisimo el computo y permite un resultado mas temprano.
También vemos una mejora significativa en cuanto al overfitting. En nuestra
práctica se notó que los clusters eran muy sesgados por la etiqueta POS pero
con una reducción de dimensionalidad esto presentó mejoras.

# Feature selection para reducción de dimensionalidad
Se detallan a continuación los métodos aplicados al corpus vectorizado para 
lograr el objetivo. Vemos que algunos requieren de una clase como target para
determinar la relevancia de los features y otros simplemente funcionan de forma
no supervisada. Mas allá de esta diferencia vemos que este selector es muy
genérico y puede ser aplicado a cualquier matriz en cuestion.
En la sección siguiente explicaremos una breve descripción de como funcionan y
luego se mostrarán los resultados aplicados en la práctica.

```python
def _feature_selection(matrix, method="PCA", target=None):
    print("--Selecting features with {} ".format(method))
    target_components = 50
    if method == "PCA":
        from sklearn.decomposition import PCA
        pca = PCA(n_components=target_components)
        reduced_matrix = pca.fit_transform(matrix)
    if method == "SVD":
        from sklearn.decomposition import TruncatedSVD
        lsa = TruncatedSVD(n_components=target_components)
        reduced_matrix = lsa.fit_transform(matrix)
    if method == "SelectKBest":
        if target is None:
            raise Exception("No target found on supervised _feature_selection")
        from sklearn.feature_selection import SelectKBest, chi2
        X, y = matrix, target
        reduced_matrix = SelectKBest(chi2, k=target_components).fit_transform(X, y)
    if method == "LinearSVC":
        if target is None:
            raise Exception("No target found on supervised _feature_selection")
        from sklearn.svm import LinearSVC
        from sklearn.feature_selection import SelectFromModel
        X, y = matrix, target
        lsvc = LinearSVC(C=1, penalty="l1", dual=False).fit(X, y)
        model = SelectFromModel(lsvc, prefit=True)
        reduced_matrix = model.transform(X)
    if method == "SelectPercentile":
        from sklearn.feature_selection import SelectPercentile, f_classif
        X, y = matrix, target
        selector = SelectPercentile(f_classif, percentile=5)
        reduced_matrix = selector.fit_transform(X, y)
    return reduced_matrix
```

# Métodos no supervisados (reducción de dimensionalidad)

## PCA
```python
    if method == "PCA":
        from sklearn.decomposition import PCA
        pca = PCA(n_components=target_components)
        reduced_matrix = pca.fit_transform(matrix)
```
Principal Component Analysis es una técnica que busca encontrar los ejes sobre
los cuales los datos quedan mejores representados en terminos de sus minimos
cuadrados.
El algoritmo que utilizamos para aplicar esta técnica es 
**sklearn.decomposition.PCA**. Su funiconamiento es bueno pero el tiempo de
demora en el analisis de recursos para matrices esparsas es intolerable.

[Ver resultado](littleSample_PCA.log)

## SVD
```python
    if method == "SVD":
        from sklearn.decomposition import TruncatedSVD
        lsa = TruncatedSVD(n_components=target_components)
        reduced_matrix = lsa.fit_transform(matrix)
```
Truncated Singular Value Descomposition es un método para la reducción de 
similaridad no supervisada. Utiliza la descomposicion de valores singulares
para reducir la dimensionalidad de la matriz de origen. 

[Ver resultado](littleSample_SVD.log)

# Métodos supervisados (selección de features)
Los métodos supervisados tienen como cláusula tener una clase de asignación
para el clasificador de features. En nuestro caso serán la POS tag o Wordnet 
tag las que aplicaremos como target a diferenciar.

## SelectKBest
Selecciona los mejores K features sobre el total garantizando que sean
los mas representativos.
```python
    if method == "SelectKBest":
        if target is None:
            raise Exception("No target found on supervised _feature_selection")
        from sklearn.feature_selection import SelectKBest, chi2
        X, y = matrix, target
        reduced_matrix = SelectKBest(chi2, k=target_components).fit_transform(X, y)
```

[Ver resultado](littleSample_SelectKBest.log)

[Ver resultado con POS](littleSample_SelectKBest_POS.log)

## LinearSVC
Linear Support Vector Classification mediante una funcion de penalidad y
aplicando tecnicas de Support Vector Machines.
```python
    if method == "LinearSVC":
        if target is None:
            raise Exception("No target found on supervised _feature_selection")
        from sklearn.svm import LinearSVC
        from sklearn.feature_selection import SelectFromModel
        X, y = matrix, target
        lsvc = LinearSVC(C=1, penalty="l1", dual=False).fit(X, y)
        model = SelectFromModel(lsvc, prefit=True)
        reduced_matrix = model.transform(X)
```
[Ver resultado](littleSample_LinearSVC.log)

## SelectPercentile
Selecciona el percentile indicado de los features actuales. La selección una
vez mas la hará basandose en la clase de target enviada.
```python
    if method == "SelectPercentile":
        from sklearn.feature_selection import SelectPercentile, f_classif
        X, y = matrix, target
        selector = SelectPercentile(f_classif, percentile=5)
        reduced_matrix = selector.fit_transform(X, y)
```
[Ver resultado](littleSample_SelectPercentile.log)

# Conclusiones
Decidimos no volver a rehacer los clustering sobre el corpus anterior ya que
seria repetir de nuevo el experimento realizado en la anterior etapa.

## Clustering con reduccion de dimensionalidad(no supervisada):
En este caso vimos una mejora en cuanto a los cardinales de los clusters,
siendo cada vez mas homogeneos y eliminando singletones. También fue notoria la 
diferncia de tiempo en cuanto a la ejecución. A pesar del overhead del calculo
de obtener cuales son los features relevantes, el gran impacto esta en el
tiempo ahorrado para el clustering. PCA definitivamente no es un buen método
para reduccion de dimensionalidad de matrices esparsas, no solo por los malos
resultados sino tambien por que fue el mas lento de los procesos aplicados.
TruncatedSVD parecería ser la mejor opción para este tipo de matrices.

## Clustering con feature selection(supervisada)
En este caso distinguimos un comportamiento bastante malo en el uso de 
LinearSVC, pero tanto SelectKBest como SelectPercentile fueron satisfactorios
en su tarea tanto con la etiqueta de POS como objetivo como asi también la
etiqueta de sentido de Wordnet.

Con respecto a esto ultimo vemos que al aplicar como objetivo los sentidos, los
clusters fueron mas orientados a esta diferenciación, mezclando sin diferenciar
adjetivos de verbos o demás. En cuanto a la evaluación con POS tag como 
objetivo del selector de features, vemos que los clusters tuvieron que ver más
con la clase de palabra, dejando agrupados verbos adjetivos o demás.
