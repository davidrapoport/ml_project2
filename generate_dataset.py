__author__ = "David"

"""
On loading this file data/ml_dataset_train.csv is loaded into memory.
After that it exposes several generate_* methods which all return a triple of
(X,y,Vectorizer) where the vectorizer is an object which has a transform() method 
which takes a list of N sentences and returns an N*M matrix where M is the number of features.
X is the training matrix and y is the vector of labels
"""

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse.csr import csr_matrix
import re, pdb, os, json
import numpy as np
from collections import Counter
import nltk
np.random.seed(1234)

lemmatizer = WordNetLemmatizer()
pattern_train = r"(\d*),\"(.*)\",(\d*)"
pattern_test = r"(\d*),\"(.*)\""
stopwords = list(stopwords.words("english"))
stopwords.append("__EOS__")

sw = ["right", "love", "people", "feel", "yeah", "one", "see", "something", "want", "year", "yes", "still", "kind", "really", "even", "little", "said", "thank", "would", "make", "two", "actually", "also", "much", "take", "way", "time", "new", "tell", "sort", "life", "good", "around", "get", "never", "story", "back", "lot", "every", "know", "got", "put", "world", "wanted", "come", "day", "look", "going", "great", "made", "like", "called", "many", "could", "work", "well", "say", "different", "thought", "thing", "didn", "part", "came", "always", "let", "went", "first", "think", "talk", "mean"]
stopwords = stopwords + sw

def tokenize(s):
    l = wordpunct_tokenize(s)
    return [lemmatizer.lemmatize(word) for word in l if len(word)>2 and word.isalpha()]

tokenize = wordpunct_tokenize #fix my screw up
stopwords = set(stopwords)

def lemmatize_and_cache(word, d={}):
    return d.setdefault(word,lemmatizer.lemmatize(word))

def read_input_file(quick_n_dirty=True, file_name="data/ml_dataset_train.csv", test_set=False):
    ''' 
    Expects there to by a data/ml_dataset_train.csv file
    Returns a a tuple of (sentence, category) with the sentence
    having been lemmatized, and all words containing non alpha characters or
    words of length less than 3 removed.
    '''
    lines = []
    targets = []
    with open(file_name, "r") as f:
        first = True
        count = 0
        for line in f:
            if first or not line.strip():
                first = False
                continue
            if "\"" in line:
                features = re.findall(pattern_test if test_set else pattern_train,line)[0]
            else:
                features = line.split(",")
            if not quick_n_dirty:
                sent = features[1].strip().lower()
                l = wordpunct_tokenize(sent)
                l = [lemmatize_and_cache(word) for word in l if len(word)>2 and word.isalpha() and word not in stopwords]
                sent = " ".join(l)
                lines.append(sent)
            else:
                count +=1
                if count>10000:
                    break
                lines.append(features[1].strip().lower())
            if not test_set:
                targets.append(int(features[2].strip()))
    output = np.zeros((len(targets),1))
    output[:,0] = targets
    return lines, output

lines, targets = [], []

def _generate_dataset(vect_args, file_name, save_to_file=True, d={}):
    if not d.get("lines") or not d.get("targets"):
        lines, targets = read_input_file()
        output = np.zeros((len(targets),1))
        output[:,0] = targets
        np.save("data/targets", output)
        d['lines'] = lines
        d['output'] = output
    else:
        lines = d['lines']
        output = d['output']
    count_vect = CountVectorizer(**vect_args)
    X = count_vect.fit_transform(lines)
    if save_to_file:
        np.save(file_name, X.data)
        f = open(file_name+"_headers.json","w")
        json.dump(count_vect.get_feature_names(), f)
        f.close()
    else:
        return X, output, count_vect

def generate_unigram_multinomial(save_to_file=True):
    return _generate_dataset({"decode_error":"ignore", "stop_words":stopwords, 
        "tokenizer":tokenize, "min_df":1}, 
            file_name="data/unigram_multinomial", save_to_file=save_to_file)


def generate_unigram_bernoulli(save_to_file=True):
    return _generate_dataset({"decode_error":"ignore", "stop_words":stopwords, 
        "binary":True, "tokenizer":tokenize, "min_df":1}, 
            file_name="data/unigram_bernoulli", save_to_file=save_to_file)

def generate_bigram_multinomial(save_to_file=True):
    return _generate_dataset({"decode_error":"ignore", "stop_words":stopwords, 
        "ngram_range":(1,2), "tokenizer":tokenize, "min_df":1}, 
            file_name="data/bigram_multinomial", save_to_file=save_to_file)

def generate_bigram_bernoulli(save_to_file=True):
    return _generate_dataset({"decode_error":"ignore", "stop_words":stopwords,
        "binary":True, "ngram_range":(1,2), "tokenizer":tokenize, "min_df":1},
            file_name="data/bigram_bernoulli", save_to_file=save_to_file)

def generate_test1():
    X = np.random.randn(100,1)
    Y = X>0
    return csr_matrix(X), (Y.astype(int)), None

def generate_test2():
    X = np.random.random_integers(0,1,(10000,2))
    Y = np.logical_and(X[:,0],X[:,1])
    return csr_matrix(X), Y.astype(int), None

def generate_test3():
    X = np.random.random_integers(0,5,(10000,3))
    Y = np.zeros((10000,1))
    Y[:,0] = X.sum(1)
    Y = (Y>5).astype(int) + (Y>11).astype(int) 
    return csr_matrix(X), Y.astype(int), None

if __name__=='__main__':
    generate_unigram_multinomial()
    generate_unigram_bernoulli()
    generate_bigram_multinomial()
    generate_bigram_bernoulli()
