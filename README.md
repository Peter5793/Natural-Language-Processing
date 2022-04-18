# Natural-Language-Processing
### Presidential Candidates Speech Categorization
#### Motivation.
 My motivation for this project was to perfectly understand the content of speech for the presidential candidates in kenya and what was contained in there speech as they accepted the nomination to be Presidential Flag Bearer.
 
 ## Process
 1. Data Importation
 2. Data Cleaning
 3. Tokenization
 4. Lemmatization
 5. Visualization
 
 
 ### Data importation
A pdf transcript was not available however i took the transcript from the Youtube channels from the media houses that were streaming the speeches. <strong>NB</strong> one had a short speech that was 3X shorter than the other however, this will not impact the result as we want to investigate the content of their speech.
The infromation was transferred to a txt file and later loaded to Spyder IDE.
Neccesary libaries were loaded  for data preprocessing:
```
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


df = pd.read_fwf("D:/Data_Analysis/Data_Analysis/")
print(df)
````



#### Data pre_processing

The standdard procedure was to remove terms such as stop words, punctuation marks  and any charcter that is not relevant in the processing of the information. This was done using the texthero library and modules in NLTK.
```
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
    
  ```
#### Tokenization and Lemmmatization
This is the step where the text was brokendown into specific sections (into specific words) NLTK has a module that can do this well, other classes can be found using Spacy library.
```
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
```



    
## Resources
[Culstering Visualization](https://towardsdatascience.com/a-complete-exploratory-data-analysis-and-visualization-for-text-data-29fb1b96fb6a)

