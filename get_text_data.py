from generate_dataset import read_input_file
from sklearn.feature_extraction.text import CountVectorizer
import json
import numpy as np
lines, targets = read_input_file(False)
data= {}

u = CountVectorizer()
X = u.fit_transform(lines)
data['num_unigrams'] = X.shape[1]
u = CountVectorizer(min_df=2)
X = u.fit_transform(lines)
data['num_unigrams_df_1'] = X.shape[1] - data['num_unigrams']

word_counts = zip(u.get_feature_names(),X.sum(0).view(np.ndarray).ravel())
word_counts.sort(key= lambda x: -x[1])
data['top_100_unigrams'] = word_counts[:100]
s = set()
for i in range(4):
    f = X[(targets==i).ravel(), :].sum(0).view(np.ndarray)
    wc = zip(u.get_feature_names(),f.ravel())
    wc.sort(key=lambda x: -x[1])
    wc = [k[0] for k in wc]
    data["top_100_unigrams_for_class_"+str(i)] = wc[:100]
    s = s | set(wc[:100])

data['top_common_unigrams'] = list(s)

for i in range(4):
    data['unique_unigrams_for_'+str(i)] = list(set([x[0] for x in data['top_100_unigrams_for_class_'+str(i)]]) -s)

f = open("text_data.txt","w")
json.dump(data,f)

