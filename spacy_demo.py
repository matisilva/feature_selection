import spacy
nlp = spacy.load('es_core_web_md')
doc = nlp(u'Batman colaboró con todos en Nicaragua y perdió el sueldo en el casino Royal')
for token in doc:
    print(token, token.pos_, token.dep_, token.head.orth_)
