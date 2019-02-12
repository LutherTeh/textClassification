from nltk.corpus import stopwords
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer ,TfidfVectorizer
#from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import pandas as pd
import glob
import re
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib


df = pd.read_csv('nstp_article_sample.xlsx')


#### Custom Function #############

def train_test (df):

	## change the column name below according (df.body = X  , df.sect = Y output)
    x_train, x_test, y_train, y_test = train_test_split( df.body, df.sect, test_size=0.15, random_state=42)
    x_train.index = range(0, len(x_train))
    x_test.index = range(0, len(x_test))
    y_train.index = range(0, len(y_train))
    y_test.index = range(0, len(y_test))
    
    return {"x_train":x_train,"x_test":x_test,"y_train":y_train,"y_test":y_test}

def clean( raw_text ):
    letters_only = re.sub("[^a-zA-Z]", " ", raw_text) # Remove non-letters
    words = letters_only.lower().split()  # Convert to lower case, split into individual words                           
    stops = set(stopwords.words("english"))   # In Python, searching a set is much faster than searching a list, so convert the stop words to a set             
    meaningful_words = [w for w in words if not w in stops]   # Remove stop words
    #meaningful_words = [stemmer.stem(x) for x in meaningful_words ]  #stemmiing malay words
    return ( " ".join( meaningful_words )) # Join the words back into one string separated by space, and return the result.


def process (text):
    clean_text=[]
    for i in range (0, len(text)):
        clean_text.append(clean(text[i]))    
    return (clean_text)


################################### Execute The Code ####################################################

data_split = train_test(df)
train = process(data_split['x_train'])
test = process (data_split['x_test'])


################################################## NB Classifier ################################################

## build a data processing pipeline
text_clf = Pipeline([('vect', CountVectorizer(max_features = 5000, ngram_range=(1, 2))),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB(alpha=0.01)),])

text_clf = text_clf.fit(train, data_split['y_train'])
predicted = text_clf.predict(test)
NB_accuracy = np.mean(predicted == data_split['y_test'])


## Gridsearch on finding optimized model
parameters = {'vect__ngram_range': [(1, 1), (1, 2),(1,3)],
                  'tfidf__use_idf': (True, False),
                  'clf__alpha': (1e-2, 1e-3),}
gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(train, data_split['y_train'])
predicted = gs_clf.predict(test)
gs_clf.best_score_
gs_clf.best_params_
gs_NB_accuracy = np.mean(predicted == data_split['y_test'])

## save the predictive model
joblib.dump(gs_clf, "NB_classifier.sav")

###################################################### SVM Classidier #############################
text_clf_svm = Pipeline([('vect', CountVectorizer(max_features = 5000, ngram_range=(1, 2))),
                         ('tfidf', TfidfTransformer()),
                         ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',alpha=0.001, n_iter=5, random_state=42)),])
    #SGDClassifier(loss='hinge',penalty='l2',alpha=0.001, n_iter=5,random_state=42)),
text_clf_svm = text_clf_svm.fit(train, data_split['y_train'])
predicted_svm = text_clf_svm.predict(test)
normal_accuracy = np.mean(predicted_svm == data_split['y_test']) 
joblib.dump(text_clf_svm, "SVM_classifier.sav")

parameters_svm = {'vect__ngram_range': [(1, 1), (1, 2)],
                  'tfidf__use_idf': (True, False),
                 'clf-svm__alpha': (1e-2, 1e-3),}
gs_clf_svm = GridSearchCV(text_clf_svm, parameters_svm, n_jobs=-1)
gs_clf_svm = gs_clf_svm.fit(train, data_split['y_train'])
gs_clf_svm.best_score_
gs_clf_svm.best_params_ 
gs_predicted_svm = gs_clf_svm.predict(test)
gs_accuracy = np.mean(gs_predicted_svm == data_split['y_test']) 
