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



####
## Resources
[Culstering Visualization](https://towardsdatascience.com/a-complete-exploratory-data-analysis-and-visualization-for-text-data-29fb1b96fb6a)

