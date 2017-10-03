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
## PCA

## SVD

# Métodos no supervisados (selección de features)
## SelectKBest

## LinearSVC


# Resultados en clustering