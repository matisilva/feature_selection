import random
from os import listdir
from os.path import isfile, join


SENTENCES_BY_FILE = 1000

def get_random_sentences(file):
    print("--Getting from {}".format(file))
    with open(file, encoding="latin-1") as f:
        content=f.readlines()
    start = random.randint(0, len(content))
    lines = []
    num_sentences = 0
    ready = False
    while content[start] != "\n":
        start += 1
    print("--Starting at {}".format(start+1))
    for line in content[start+1:]:
        if num_sentences == SENTENCES_BY_FILE - 1:
            ready = True
        if "\n" == line:
            num_sentences += 1
            if ready:
                lines.append(line)
                break
        lines.append(line)
    return num_sentences, lines

mypath = "./tagged.es/"
files = [join(mypath,f) for f in listdir(mypath) if isfile(join(mypath, f))]
lines = []
total_sentences = 0
for file in files:
    num_sentences, sentences = get_random_sentences(file)
    print("--Added {} sentences".format(str(num_sentences)))
    total_sentences += num_sentences
    lines.extend(sentences)
thefile = open('randomSample2_SpanishEtiquetado', 'w')
print("--Writing {}".format(total_sentences))
for item in lines:
    thefile.write("%s" % item)