from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, SelectPercentile
import numpy as np
from gensim.models import word2vec
import pdb

DEFAULT_KEYWORDS = ["movie", "author", "music"]
model = word2vec.Word2Vec.load_word2vec_format("./data/vectors.bin", binary=True)

class Word2VecVectorizer(CountVectorizer):
    
    def __init__(self, k=50, filter_type="k_percent", keywords=None,
            input='content', encoding='utf-8',
             decode_error='ignore', strip_accents=None,
             lowercase=True, preprocessor=None, tokenizer=None,
             stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
             ngram_range=(1, 1), analyzer='word',
             max_df=1.0, min_df=1, max_features=None,
             vocabulary=None, binary=False, dtype=np.int64):
        '''
        A vectorizer which selects features based on word2vec similarity to a set of keywords
        input: k: If filter_type is k_percent then it is the percentile of top features to accept, if filter_type is 
                  "top_k" then k is the number of features to take. Can be "all"
               filter_type: "top_k" or "k_percent"
               keywords: An optional list of keywords to compare similarity to

        '''
        self.k = k
        self.filter_type = filter_type
        if not keywords:
            self.keywords = DEFAULT_KEYWORDS
        else:
            self.keywords = keywords
        self.model = model 
        super(Word2VecVectorizer, self).__init__(input, encoding,
                                 decode_error, strip_accents, lowercase, preprocessor, tokenizer,
                                stop_words, token_pattern, ngram_range, analyzer,
                                max_df, min_df, max_features, vocabulary, binary, dtype)

    def fit_transform(self, raw, y=None):
        self.fit(raw, y)
        return self.transform(raw)
        
        '''
        X = super(Word2VecVectorizer, self).fit_transform(raw, y)
        
        feature_names = self.get_feature_names()
        values = np.zeros(feature_names.size())
        for cnt,feature in enumerate(feature_names):
            l = []
            for keyword in self.keywords:
                try:
                    l.append(self.model.similarity(keyword, feature))
                except:
                    l.append(0)
            values[cnt] = max(l)
        self.values = values
        def score_fun(X,y):
            return values, np.zeros(values.size())
        if self.filter_type == "k_percent":
            self.selector = SelectPercentile(score_func=score_fun, percentile=self.k)
            self.selector.fit(X)
            return self.selector.transform(X)
        elif self.filter_type == "top_k":
            self.selector = SelectKBest(score_fun=score_fun, k=self.k)
            self.selector.fit(X)
            return self.selector.transform(X)
        '''

        

    def fit(self, raw, y=None):
        X = super(Word2VecVectorizer, self).fit_transform(raw, y)

        feature_names = self.get_feature_names()
        values = np.zeros(len(feature_names))
        for cnt,feature in enumerate(feature_names):
            l = []
            feature = feature.split()
            for keyword in self.keywords:
                try:
                    l.append(self.model.n_similarity([keyword], feature))
                except:
                    l.append(0)
            values[cnt] = max(l)
        self.values = values
        def score_fun(X,y):
            return values, np.zeros(values.size)
        if self.filter_type == "k_percent":
            self.selector = SelectPercentile(score_func=score_fun, percentile=self.k)
        elif self.filter_type == "top_k":
            self.selector = SelectKBest(score_func=score_fun, k=self.k)
        self.selector.fit(X, y)

    def transform(self, raw):
        X = super(Word2VecVectorizer, self).transform(raw)
        return self.selector.transform(X)

    def generate_most_common_list(self, raw_docs):
        X = super(Word2VecVectorizer, self).fit_transform(raw_docs)
        feature_names = self.get_feature_names()
        values = []
        num_not_found = 0
        for cnt, feature in enumerate(feature_names):
            l = []
            feature = feature.split()
            for keyword in self.keywords:
                try: 
                    l.append(self.model.n_similarity([keyword], feature), keyword)
                except:
                    num_not_found += 1
                    break
            m = max(l, key= lambda s: s[0])
            values.append(" ".join(feature), m[0], m[1], X[:, cnt].sum())
        return num_not_found, values
