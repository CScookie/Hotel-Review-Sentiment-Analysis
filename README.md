# Hotel-Review-Sentiment-Analysis

## Introduction:
Predict sentiment of a textual review using only the raw textual data from the review.

The data can be found here: https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe

For any textual review, our challenge is to predict if it correspond to a positive or negative review. 

In the dataset, review ratings are ranged from 0 to 10. 

We will categorize ratings as such:
* Ratings < 5 as negative review
* Ratings >= 5 as positive review

# Loading the data:
```python
import numpy as np
import re
import pickle 
import nltk
import pandas as pd
from nltk.corpus import stopwords
from sklearn.datasets import load_files
nltk.download('stopwords')
```

### Copy original dataframe to another variable to ensure the original dataframe will not be modified
```python
df = raw_data.copy()
```

### Combining two columns to simplify the data
```python
df['Review'] = df.Negative_Review + df.Positive_Review
df['Review'][2]
```
Output:
```
' Rooms are nice but for elderly a bit difficult as most rooms are two story with narrow steps So ask for single level Inside the rooms are very very basic just tea coffee and boiler and no bar empty fridge  Location was good and staff were ok It is cute hotel the breakfast range is nice Will go back '
```

### Create the target and drop all columns that are not relevant

* Reviewer_score lower than 5 will be considered negative
* Reviewer_score equal or higher than 5 will be consider positive
``` python
# create the target
df['Target'] = df["Reviewer_Score"].apply(lambda x: 0 if x < 5 else 1)

df = df[['Review', 'Target']]

X, y = df.Review, df.Target
```

```python
print('Total number of rows: ', df.shape[0])
print('Total number of positive reviews: ', y.sum())
print('Percentage of positive reivews:', y.sum()/df.shape[0])
```
Output:
```
Total number of rows:  515738
Total number of positive reviews:  493457
Percentage of positive reivews: 0.9567978314570577
```
### From the above output, the dataset is highly imbalance. We will have to balance in such a way it is approximately 50% positive and negative

```python
#remove excess 0s
one_counter = 0
counter = 0
indices_to_remove =[]


for index, row in df.iterrows():
    if row['Target'] == 1:
        one_counter+=1
        if one_counter >= (df.shape[0] - df.Target.sum()):
            indices_to_remove.append(index)
    
df_balanced = df.drop(indices_to_remove)
df_balanced.reset_index(inplace=True, drop=True)

#check if targets are balance (approx. 50%)

print(df_balanced['Target'].sum())
print(df_balanced['Target'].shape[0])
print(df_balanced['Target'].sum()/df_balanced['Target'].shape[0])

X = df_balanced['Review']
y = df_balanced['Target']
```
Output:
```
22280
44561
0.49998877942595543
```

### Text cleaning
* Removing white spaces, punctuations, single letter words
* A function to clean text

```python
def text_cleaner(X):
    corpus = []
    for i in range(0, len(X)):
        review = re.sub(r'\W', ' ', str(X[i]))
        review = review.lower()
        review = re.sub(r'^br$', ' ', review)
        review = re.sub(r'\s+br\s+',' ',review)
        review = re.sub(r'\s+[a-z]\s+', ' ',review)
        review = re.sub(r'^b\s+', '', review)
        review = re.sub(r'\s+', ' ', review)
        corpus.append(review)
    return X

corpus = text_cleaner(X)
```
### Prior to lemmatizing, we have to take a part of speech parameter, “pos”. If not supplied, the default is “noun”. We will make a function to solve it

```python
# return the wordnet object value corresponding to the POS tag
from nltk.corpus import wordnet

def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
```
### In lemmatisation, the part of speech of a word should be first determined and will return the dictionary form of a word, which must be a valid word.
```python
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
lemmatizer = WordNetLemmatizer()

# Lemmatization
pos_tags = pos_tag(corpus)
corpus = [WordNetLemmatizer().lemmatize(text[0], get_wordnet_pos(text[1])) for text in pos_tags]
```

### Creating the Tf-Idf model
```python
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features = 6000, min_df = 3, max_df = 0.6, stop_words = stopwords.words('english'))
X = vectorizer.fit_transform(corpus).toarray()
```

### Splitting the dataset into training and test set

```python
from sklearn.model_selection import train_test_split
text_train, text_test, sent_train, sent_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
```

### Training the model
```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(text_train,sent_train)
```

### Testing model performance with an accuracy of 86%
```python
sent_pred = model.predict(text_test)
model.score(text_test, sent_test)
```
Output:
```
0.8633456748569506
```
# Testing the model with textual reviews.
* 1 means positive
* 0 means negative

```python
sample = [text_cleaner("""The room was simple, quite small and had basic equipments. The room was also clean. 
The main advantage of the room was the balcony which offers a really nice view on the street. 
The breakfast was also simple. We preferred to take it outside as the continental breakfast was not really suitable for us. 
At last, the location is really great, in the Chinatown Food Street and close to the Chinatown train station. 
The only problem is the street which is quite noisy until late night.""")]

sample = vectorizer.transform(sample).toarray()
sentiment = model.predict(sample)
sentiment[0]
```
Output:
```
1
```
```python
sample = [text_cleaner("""The receptionist has a serious attitude problem. 
the room is not what you expected from the picture. room was small, aircon was not cold. 
the aircon and bed was at two different location. This hotel is definitely not worth the money will never come back"""
)]

sample = vectorizer.transform(sample).toarray()
sentiment = model.predict(sample)
sentiment[0]
```
Output:
```
0
```
