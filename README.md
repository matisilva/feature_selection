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
clasificaciones como vectores objetivo, la etiqueta POS en nuestro caso
generada por la libreria spacy y en una segunda iteración será el sentido
basado en Wordnet.

# WikiCorpus
- metodos de vectorizacion, featurizacion normalizacion spacy de clustering with embeddings
- normalizacion de texto
- wiki corpus tagged o raw.
- Parseo


# Métodos supervisados (reducción de dimensionalidad)
500K oraciones(ejemplo)
[Ver resultado](randomSample_spanishEtiquetado)

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
>>>>>>> Stashed changes
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
# Resultados en clustering
## Métodos no supervisados sobre corpus de "la voz del interior".

## Métodos no supervisados aplicados a WikiCorpus

# Resultados en clustering