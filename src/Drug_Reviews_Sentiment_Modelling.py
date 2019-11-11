#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import datetime
import sklearn.datasets as datasets
from scipy.stats import chisquare
from sklearn import preprocessing


import os
import numpy as np
import pandas as pd
import numpy.random as rand
from itertools import islice
from io import StringIO
#Models
import sklearn.model_selection as cv
from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (GradientBoostingRegressor, 
                              GradientBoostingClassifier, 
                              AdaBoostClassifier,
                              RandomForestClassifier,
                             RandomForestRegressor)
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import log_loss,roc_curve, make_scorer, confusion_matrix,roc_auc_score,classification_report,f1_score


from pylab import rcParams
rcParams['figure.figsize'] = (12, 8)
#NLP
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
import unicodedata
import string
import nltk
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer




# ## Load Training and test Data

def load_df():
    df_train = pd.read_csv('drugsComTrain_raw.tsv',sep='\t',index_col=0)
    df_test = pd.read_csv('drugsComTest_raw.tsv',sep='\t',index_col=0)
    return df_train,df_test



def create_rating_label(df):
    if df['rating'] >= 7: 
        return 'high'
    elif (df['rating'] < 4):
        return 'low'
    else :
        return 'neutral'


def apply_label(df_train,df_test):
    df_train['label']=df_train.apply(create_rating_label, axis=1)
    df_test['label']=df_test.apply(create_rating_label, axis=1)
    return df_train,df_test


# # 2. NLP

### 2.1 Sentiment Analysis
#### 2.1.1 Vectorizerization of words using tf-idf
### Step -1 Change all the text to ASCII , Remove stop words and punctuation and Lemmatise

### Creating tfidf for review data 

#### NLP pipeline



punctuation_ = string.punctuation+'``'+"''"+'...'+''
stopwords_ = set(stopwords.words('english'))
stemmer_porter = PorterStemmer()
stemmer = SnowballStemmer('english')

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    text = "".join([ch for ch in text if ch not in punctuation_])
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems
def create_tf_idf_vectoriser(df_train,df_test):
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf = TfidfVectorizer(stop_words='english',tokenizer=tokenize,max_features= 10000)
    X_train = tfidf.fit_transform(df_train['review'])
    X_test = tfidf.transform(df_test['review'])
    return X_train,X_test


### Modelling 

####  Random Forest 



def randomforestmodel(X_train,y_train,X_test, y_test):
    rf = RandomForestClassifier(oob_score=True,n_estimators=60,random_state=0)
    rf.fit(X_train,y_train)
    rf_pred = rf.predict(X_test)
    rfcm=confusion_matrix(Y_test,rf_pred)
    rf_score=rf.score(X_test, y_test)
    return rf_score,rf_pred,rfcm





## Grid Search On Random Forest Classifier 



def grid_serach_cv(X_train,y_train,X_test,y_test):
    rfmodel = RandomForestClassifier()
    bootstrap=[True, False],
    n_estimators=[60,100]
    parameters = {'n_estimators': n_estimators,
                   'bootstrap': bootstrap}

    scorer = make_scorer(log_loss,
                         greater_is_better=False,
                         needs_proba=True)
    clf = GridSearchCV(rfmodel,
                       parameters,
                       cv=10,
                       scoring=scorer)
    clf.fit(X_train,y_train)
    accuracy=(clf.predict(X_test) == y_test).mean()
    logloss=clf.score(X_test, y_test)
    best_params=clf.best_params_
    return accuracy,logloss,rfcm,best_params



# Logistic Regression
def model_logistic_regression(X_train,y_train,X_test,y_test):
    lg = LogisticRegression()
    lg.fit(X_train,y_train)
    lg_pred = lg.predict(X_test)
    accuracyscore=lg.score(X_test, y_test)
    return accuracyscore



# Gradient Boost
def model_gradient_boosting_regrssor(X_train,y_train,X_test,y_test):
    N_ESTIMATORS = 40
    gb = GradientBoostingClassifier(learning_rate=0.01, 
                                       n_estimators=N_ESTIMATORS, 
                                       min_samples_leaf=10)
    gb.fit(X_train,y_train)
    gb_pred = gb.predict(X_test)
    gb_accuracy=gb.score(X_test, y_test)
    return gb_accuracy ##0.6591526243350817



# Naiyes_bayes
def model_naiyes_bayes(X_train,y_train,X_test,y_test):
    nb =  MultinomialNB()
    nb.fit(X_train, y_train)
    nb_pred = nb.predict(X_test)
    nb_accuracy=nb.score(nb_pred, y_test)
    return nb_accuracy

df_train,df_test=load_df()
df_train,df_test=apply_label(df_train,df_test)
X_train,X_test=create_tf_idf_vectoriser(df_train,df_test)
Y_train = df_train['label']
Y_test = df_test['label']
X_train,X_test=create_tf_idf_vectoriser(df_train,df_test)
lg_score=model_logistic_regression(X_train,Y_train,X_test,Y_test)
rf_model=randomforestmodel(X_train,Y_train,X_test, Y_test)
gb_model=model_gradient_boosting_regrssor(X_train,Y_train,X_test, Y_test)
nb_model=model_naiyes_bayes(X_train,Y_train,X_test, Y_test)
