from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import re, pdb, os, json
import numpy as np
from collections import Counter
import nltk

lemmatizer = WordNetLemmatizer()
pattern = r"(\d*),\"(.*)\",(\d*)"

def read_input_file():
    lines = []
    targets = []
    with open("data/ml_dataset_train.csv", "r") as f:
        first = True
        for line in f:
            if first or not line.strip():
                first = False
                continue
            if "\"" in line:
                features = re.findall(pattern,line)[0]
            else:
                features = line.split(",")
            lines.append(features[1].strip().lower())
            targets.append(int(features[2].strip()))
    return lines, targets

lines, targets = read_input_file()

output = np.zeros((len(targets),1))
output[:,0] = targets
np.save("data/targets", output)

def generate_unigram_multinomial():
    count_vect = CountVectorizer(decode_error="ignore", stop_words="english")
    X = count_vect.fit_transform(lines)
    np.save("data/unigram_multinomial",X.data)
    f = open("data/unigram_multinomial_headers.json","w")
    json.dump(count_vect.get_feature_names(), f)
    f.close()

def generate_unigram_bernoulli():
    count_vect = CountVectorizer(decode_error="ignore", stop_words="english", binary=True)
    X = count_vect.fit_transform(lines)
    np.save("data/unigram_bernoulli",X.data)
    f = open("data/unigram_bernoulli_headers.json","w")
    json.dump(count_vect.get_feature_names(), f)
    f.close()

def generate_bigram_multinomial():
    count_vect = CountVectorizer(decode_error="ignore", stop_words="english", ngram_range=(1,2))
    X = count_vect.fit_transform(lines)
    np.save("data/bigram_multinomial",X.data)
    f = open("data/bigram_multinomial_headers.json","w")
    json.dump(count_vect.get_feature_names(), f)
    f.close()

def generate_bigram_bernoulli():
    count_vect = CountVectorizer(decode_error="ignore", stop_words="english", binary=True, ngram_range=(1,2))
    X = count_vect.fit_transform(lines)
    np.save("data/bigram_bernoulli",X.data)
    f = open("data/bigram_bernoulli_headers.json","w")
    json.dump(count_vect.get_feature_names(), f)
    f.close()
    

generate_unigram_multinomial()
generate_unigram_bernoulli()
generate_bigram_multinomial()
generate_bigram_bernoulli()
