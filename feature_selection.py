from nltk.tokenize import RegexpTokenizer, word_tokenize, sent_tokenize
from sklearn.feature_extraction import DictVectorizer
from sklearn.cluster import KMeans
from collections import defaultdict
from string import punctuation
import matplotlib.pyplot as plt
from datetime import datetime
from nltk.corpus import stopwords
from scipy.spatial.distance import cdist
from nltk.corpus import wordnet as wn
from functools import reduce
import numpy as np
import pprint

ommited_words = ['endofarticle']

pp = pprint.PrettyPrinter(indent=4)

def _wordnet_lemmas(word):
    return wn.synsets(word, lang='spa')[0].lemma_names('spa')

def _wordnet_definition(word):
    return wn.synsets(word, lang='spa')[0].definition()

def _words_filter(word):
    if word[2][0] == "F":
        return True
    word = word[0].lower()
    if len(word) < 2:
        return True
    return reduce(lambda x,y: x or y, list(map(lambda x: x.lower==word.lower, ommited_words)), False)

def parser_wikicorpus(file):
    print("--Parsing {}".format(file))
    with open(file, encoding="ISO-8859-1") as f:
        content=f.readlines()
    _filter_preprocessing = lambda line: "<doc" not in line and "</doc" not in line
    content = list(filter(_filter_preprocessing, content))
    #content = list(map(lambda line: line.split(" "), content))
    content.append('\n')
    sentences = []
    sentence = []
    for line in content:
        if (line != "\n" ):
            sentence.append(line)
        else: 
            sentences.append(sentence)
            sentence = []
    tagged_sentences = []
    extra_data_sentences = []
    for id, sentence in enumerate(sentences):
        tagged_words = []
        extra_data = []
        for word in sentence:
            word = word.split(" ")
            if _words_filter(word):
                continue
            tagged_words.append((word[0].lower(),word[2],word[0]))
            extra_data.append((word[1], word[3]))
        tagged_sentences.append(tagged_words)
        extra_data_sentences.append(extra_data)
    return tagged_sentences, extra_data_sentences

def _tagger(file, tagger_name='stanford'):
    if tagger_name  == "spacy":
        import spacy
        tagger = spacy.load('es_core_web_md')
    raw_text = open(file).read()
    raw_sentences = sent_tokenize(raw_text)
    tagged_sentences = []
    extra_data_sentences = []
    for id, sentence in enumerate(raw_sentences):
        sentence = word_tokenize(sentence)
        tagged_words = []
        extra_data = []
        doc = tagger(" ".join(_clean_sentence(sentence, tuples=False)))
        for token in doc:
            tagged_words.append((str(token).lower(), token.pos_, str(token)))
            extra_data.append((token.dep_, token.head.orth_))
        tagged_sentences.append(tagged_words)
        extra_data_sentences.append(extra_data)
    return tagged_sentences, extra_data_sentences

def _only_filtered_words(file):
    raw_text = open(file).read()
    tokenizer = RegexpTokenizer(r'\w+')
    stop = stopwords.words('spanish') + list(punctuation)
    tokens = tokenizer.tokenize(raw_text.lower())
    tokens = [i for i in tokens if i not in stop]
    return tokens

def _clean_sentence(sentence, tuples=True):
    def need_change(word):
        if word.isdigit():
            return "NUMBER"
        return word
    stop = stopwords.words('spanish') + list(punctuation + "y")
    translator = str.maketrans('', '', punctuation + "¡¿")
    sentence = [word.lower() for word in sentence]
    sentence = [word.translate(translator) for word in sentence]
    sentence = [need_change(word) for word in sentence if len(word) > 1]
    return sentence

def featurize(tagged_sentences, extra_data=None):
    print("--Featurizing")
    featurized = {}
    stopw = stopwords.words('spanish') + list(punctuation)
    for idy, tagged_sentence in enumerate(tagged_sentences):
        for idx, (word, POS, real_word) in enumerate(tagged_sentence):
            if word in featurized.keys():
                features = featurized[word]
            else:
                features = defaultdict(int)
                features['istittle:'] = word.istitle()
                features['isupper'] = word.isupper()
                features['mayusinit'] = word[0].isupper()
                features[extra_data[idy][idx][0]] += 1
                features[extra_data[idy][idx][1]] += 1
                features['target'] = extra_data[idy][idx][1]
            features[POS] += 1
            features['mentions'] += 1
            #preword
            if idx == 0:
                features['START'] += 1
            else:
                features[tagged_sentence[idx - 1][0] + "-"] += 1
                features[tagged_sentence[idx - 1][1] + "-"] += 1
            #prepreword
            if idx <= 1:
                features['START-'] += 1
            else:
                features[tagged_sentence[idx - 2][0] + "--"] += 1
                features[tagged_sentence[idx - 2][1] + "--"] += 1 
            #posword
            if idx == len(tagged_sentence) - 1:
                features['END'] += 1
            else:
                features[tagged_sentence[idx + 1][0] + "+"] += 1
                features[tagged_sentence[idx + 1][1] + "+"] += 1
            #posposword
            if idx >= len(tagged_sentence) - 2:
                features['END+'] += 1
            else:
                features[tagged_sentence[idx + 2][0] + "++"] += 1
                features[tagged_sentence[idx + 2][1] + "++"] += 1
            featurized[word] = features
    return featurized

def _normalize(matrix):
    print("--Normalizing..")
    row_sums = matrix.sum(axis=1)
    return matrix / row_sums[:, np.newaxis]

def _feature_selection(matrix, method="PCA", target=None):
    print("--Selecting features with {} ".format(method))
    target_components = 300
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
        lsvc = LinearSVC(C=0.5, penalty="l1", dual=False).fit(X, y)
        model = SelectFromModel(lsvc, prefit=True)
        reduced_matrix = model.transform(X)
    return reduced_matrix

def vectorize(featurized_words, normalize=True, feature_selection=True):
    print("--Vectorizing...")
    words_index = []
    features_index = []
    mention_index = []
    target_index = []
    for word in featurized_words.keys():
        if featurized_words[word]['mentions'] < 15 or len(word) < 4 or word=="number":
            continue
        mention_index.append(featurized_words[word].pop('mentions'))
        words_index.append(word)
        target_index.append(featurized_words[word].pop('target'))
        features_index.append(featurized_words[word])
    vectorizer = DictVectorizer(sparse=False)
    vectors = vectorizer.fit_transform(features_index)
    if normalize:
        vectors = _normalize(vectors)
    if feature_selection:
        print(vectors.shape)
        vectors = _feature_selection(vectors, method="SelectKBest", target=target_index)
        print(vectors.shape)
    return words_index, vectors, mention_index

def _k_distortion(vectorized_words):
    distortions = []
    K = range(1,40)
    for k in K:
        kmeanModel = KMeans(n_clusters=k).fit(vectorized_words)
        kmeanModel.fit(vectorized_words)
        distortions.append(sum(np.min(cdist(vectorized_words, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / vectorized_words.shape[0])
    # Plot the elbow
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()

def cluster(vectorized_words, word_index):
    print("--Clustering...")
    kmeans = KMeans(n_clusters=60).fit(vectorized_words)
    return kmeans

def preety_print_cluster(kmeans, refs, mentions):
    print("--Making graph...")
    labels = kmeans.labels_
    labeled = defaultdict(list)
    for id, label in enumerate(labels):
        labeled[label].append(refs[id])
    for label in labeled.keys():
        print(label)
        print(labeled[label])
    centroids = kmeans.cluster_centers_
    size = [len(word) for word in refs]
    data = np.array(list(zip(labels, refs, size, mentions)))
    plt.scatter(data[:,0], data[:,2],
             marker='o',
             c=data[:,0],
             s=list(map((lambda x: int(x)*10), data[:,3])),
             facecolors="white",
             edgecolors="blue")
    for idx, point in enumerate(refs):
        if mentions[idx] > 250:
            plt.annotate(point, xy=(data[:,0][idx], data[:,2][idx]))
    plt.ylabel('longitud')
    plt.xlabel('cluster')
    print("Finalizado (%s)" %str(datetime.now()))
    plt.show()

if __name__ == "__main__":
    with_spacy = True
    #tagger = 'spacy'
    distortion = False
    # file = "lavoz1000notas.txt"
    # tagged_sentences, extra_data = _tagger(file, tagger)
    file = "spanishEtiquetado_sample"
    tagged_sentences, extra_data = parser_wikicorpus(file)
    print("Iniciando con {} ({})".format(file , str(datetime.now())))
    words, vectors, mentions = vectorize(featurize(tagged_sentences,
                                                   extra_data=extra_data),
                                                   normalize=True,
                                                   feature_selection=False)
    if distortion:
        _k_distortion(vectors)
    kmeans = cluster(vectors, words)
    preety_print_cluster(kmeans, words, mentions)
