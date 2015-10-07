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
stopwords = list(stopwords.words())
stopwords.append("__EOS__")

def tokenize(s):
    l = wordpunct_tokenize(s)
    return [word for word in l if len(word)>2]
tokenize = wordpunct_tokenize #undo my screw up

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

def _generate_dataset(vect_args, file_name, save_to_file=True):
    count_vect = CountVectorizer(**vect_args)
    X = count_vect.fit_transform(lines)
    if save_to_file:
        np.save(file_name, X.data)
        f = open(file_name+"_headers.json","w")
        json.dump(count_vect.get_feature_names(), f)
        f.close()
    else:
        indices = []
        for word,index in count_vect.vocabulary_.items():
            if len(word)>2:
                indices.append(index)
        X = X[:,indices]
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

if __name__=='__main__':
    generate_unigram_multinomial()
    generate_unigram_bernoulli()
    generate_bigram_multinomial()
    generate_bigram_bernoulli()
