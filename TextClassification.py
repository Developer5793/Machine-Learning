import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np

import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score


data = pd.read_csv("https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv", encoding='latin-1')
print(data.columns)

data = data[["v1","v2"]]
data.columns = ["label","text"]


print(data.isna().sum())

#data['label'].value_counts(normalize = True).plot.bar()

#data.label.value_counts(normalize=True).plot(kind='bar')
#plt.show()
# text preprocessing

# create a list text

text = list(data['text'])

# preprocessing loop
lemmatizer = WordNetLemmatizer()
corpus = []

# 1. Remove all nonalphabetcial signs
# 2. Lower and Tokenize
# 3. remove stopwords
# 4. lemmatize words 


for i in range(len(text)):
    r = re.sub('[^a-zA-Z]', ' ', text[i])

    r = r.lower()

    r = r.split()

    r = [word for word in r if word not in stopwords.words('english')]

    r = [lemmatizer.lemmatize(word) for word in r]

    r = ' '.join(r)
    corpus.append(r)



#assign corpus to data['text']

data['text'] = corpus


X1 = data['text']

X2 = data['label']




# train test split (70% train - 30% test)


X_train, X_test, y_train, y_test = train_test_split(X1, X2, test_size=0.30, random_state=100)


print('Training Data :', y_train.shape)

print('Testing Data : ', y_test.shape)


vectorizer = CountVectorizer()


X_train_vectorized = vectorizer.fit_transform(X_train)

print(X_train_vectorized.shape)



model = LogisticRegression()

model.fit(X_train_vectorized, y_train)

X_test_vectorized = vectorizer.transform(X_test)

# generate predictions
predictions = model.predict(X_test_vectorized)

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, pos_label="ham")

df = pd.DataFrame(metrics.confusion_matrix(y_test,predictions), index=['ham','spam'], columns=['ham','spam'])

print("Confusion Matrix:", df)
print("Accuracy is:", accuracy)
print("Precision is:", precision)