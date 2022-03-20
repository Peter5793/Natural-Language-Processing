%pip install spacy
%pip install texthero
%pip install nltk

import spacy
import texthero as hero 
import pandas as pd
import numpy as np
import matplotli.pyplot as plt
import seaborn as sns
import nltk

import warnings
warnings.filterwarnings("ignore")

language_model = "en_core_web_sm"
spacy.cli.download(language_model)
nlp = spacy.load(language_model)


df = pd.read_fwf("D:/Data_Analysis/Data_Analysis/Ruto_speech.txt")
print(df)

# renaming the column
df.columns
new_columns  = ["actual", "None"]
df.columns = new_columns

# text preprocessing to remove the stop words
# we shall use the english stopwords list from spacy for this part
import string
from nltk.corpus import stopwords
nltk.download('stopwords')
stop = stopwords.words('english')
print(stop)
df['clean_speech'] = df['actual'].apply(lambda x: ' '.join([word for word in str(x).split() if word not in (stop)]))
print(df['clean_speech'])

# punctuation mark removal
string.punctuation
# function to remove the punctuation

def remove_punctuation(text):
    free_text = "".join([i for i in text if i  not in string.punctuation])
    return free_text
# applying the function on the column
df['clean_speech'] = df['clean_speech'].apply(lambda x :remove_punctuation(x))
print(df['clean_speech'])
# tokenization
# splitting the text into smalller units

# using spacy to perform lemmatization
df['clean_speech'] = df['clean_speech'].apply(lambda row : " ".join([w.lemma_ for w in nlp(row)]))
import re
def tokenization(text):
    tokens = re.split('W+', text)
    return tokens

#applying the function to the column
df['clean_speech'] = df['clean_speech'].apply(lambda x : tokenization(x))
print(df["clean_speech"][0])
# vectorization for the creation of the TFIDF model
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(
    lowercase= True,
    max_features=150,
    max_df= 80,
    min_df= 5,
    ngram_range= (1,1)
    )

vectors = vectorizer.fit_transform(df['clean_speech'])
feature_names = vectorizer.get_feature_names()

print(feature_names)

# clustering
from sklearn.cluster import KMeans
# using a cluster of 20
true_k = 10
model = KMeans(n_clusters= true_k, init = 'k-means++', max_iter = 100, n_init = 1)
model.fit(vectors)


order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()

with open ("D:/Data_Analysis/Data_Analysis/size.txt", "w", encoding ="utf-8") as f:
    for i in range(true_k):
        f.write(f"Cluster{i}")
        
        f.write("\n")
        for ind in order_centroids[i, :10]:
            f.write(' %s' % terms[ind],)
            f.write('\n')
        f.write("\n")
        f.write("\n")
        
# for visualization we shal use a word cloud and pyLDAvis
from wordcloud import WordCloud, ImageColorGenerator, STOPWORDS
import matplotlib.cm as cm 
import collections

text = df['clean_speech'].str.split()
text

text_spam = [" ".join(x) for x in text]
final = ' '.join(text_spam)
final[: 300]

stopwords = set(STOPWORDS)
stopwords.update(['want','us'])
wordcloud_spam = WordCloud(stopwords=stopwords, background_color="white", max_font_size = 80,
max_words = 100).generate(final)
    
plt.figure(figsize=(20, 20))
plt.imshow(wordcloud_spam, interpolation = "bilinear")
plt.axis('off')
plt.show()