# Natural-Language-Processing

Machine learning algorithms for NLP

## METHODS
### Using Spacy

This is a jupyter notebook that is aimed at familiralization with the use of spacy for natural language processing.
Spacy is a Python library that uses Natural Language Processing to extract relevat information from unstructured texts.

There is use of trained pipelines that are being used for English and other languages to predict:
* Parts of SPeech 
* Syntactic Dependencies
* Named Entities

### Installation and running

```
$ python -m spacy download en_core_web_sm
```
```
import spacy
```
Below is the english pipeline
```
nlp = spacy.load("en_core_web_sm")
```
## Resources
(Culstering Visualization)[https://towardsdatascience.com/a-complete-exploratory-data-analysis-and-visualization-for-text-data-29fb1b96fb6a]
