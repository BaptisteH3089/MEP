#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 11:43:57 2021

@author: baptistehessel

Code to build a classification model able to predict the type of an article.
There are 4 types:
    - minor;
    - tertiary;
    - secondary;
    - main.

The key point is that our data are published pages, it means that we have the
areas of the articles as well as their location in the page. Both of these
information are very important to determine the type of the article.
Once all the data are labelled, we use only the content of the article to
determine the type.
We should have enough data (around 10.000 pages) to build an efficient model.

Results : The f1-score is only of 50%. It is a bit disappointed. Try to mix
the predictions of this model with the predictions with the features and it
lowers the f1-score of the model with the features.

This model with the content should be left behind.

"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC
import xgboost as xgb
import numpy as np
import pickle
import time
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('path_customer',
                    type=str,
                    help='The path to the data.')
parser.add_argument('path_objects',
                    type=str,
                    help='The path to the objects to lemmatize.')
parser.add_argument('--max_features',
                    type=int,
                    help='The max number of features for tfidf vect.',
                    default=None)
parser.add_argument('--xgb',
                    type=int,
                    help='Whether to use XGBoost.',
                    default=False)
args = parser.parse_args()

if args.path_customer[-1] == '/':
    path_customer = args.path_customer
else:
    path_customer = args.path_customer + '/'
if args.path_objects[-1] == '/':
    path_objects = args.path_objects
else:
    path_objects = args.path_objects + '/'

#path_customer = '/Users/baptistehessel/Documents/DAJ/MEP/montageIA/data/CM2/'
#path_objects = '/Users/baptistehessel/Documents/DAJ/SIM/fichiers_pickle/'

with open(path_customer + 'dict_pages', 'rb') as file:
    dict_pages = pickle.load(file)
# List of stop-words
with open(path_objects + 'Avoid', 'rb') as file:
    avoid = pickle.load(file)
# Dictionary uses to remplace words by their lemma
with open(path_objects + 'Dico', 'rb') as file:
    dict_lem = pickle.load(file)


def Clean(txt):
    """
    Remplace les caractères qui ne sont pas alpha-numériques par des espaces.
    """
    return re.sub(re.compile('\W+'), ' ', txt)


def CleanCar(txt):
    """
    Enlève les caractères entre deux espaces.
    """
    return ' '.join([word for word in txt.split() if len(word) > 1])


def LemTxt(txt, dict_lem):
    return ' '.join([dict_lem[word] if word in dict_lem.keys() else word
                     for word in txt.split()])


def CreationCorpusLabels(dict_pages):
    init = False
    for id_page, dict_page in dict_pages.items():
        for id_article, dict_article in dict_page['articles'].items():
            if dict_article['isPrinc'] == 1:
                label = 0
            elif dict_article['isSec'] == 1:
                label = 1
            elif dict_article['isSub'] == 1:
                label = 1
            elif dict_article['isTer'] == 1:
                label = 2
            elif dict_article['isMinor'] == 1:
                label = 3
            else:
                # Article with no label
                print("Article with no label")
                continue
            if len(dict_article['content']) > 10:
                if init is False:
                    corpus = [dict_article['content']]
                    Y = [label]
                    init = True
                else:
                    corpus.append(dict_article['content'])
                    Y.append(label)
    return corpus, Y


def CleanCorpus(corpus):
    # Remove the non-alphanumeric characters
    corpus = list(map(lambda x: CleanCar(Clean(x.lower())), corpus))
    # Replace words by their lemma
    corpus = list(map(lambda x: LemTxt(x, dict_lem), corpus))
    return corpus


def MatricesXYArticleContent(dict_pages, max_features=args.max_features):
    corpus, Y = CreationCorpusLabels(dict_pages)
    print(f"The length of corpus: {len(corpus)}.")
    print(f"The length of Y: {len(Y)}.")
    corpus = CleanCorpus(corpus)
    # Converting text into vector with tf-idf
    # Comparison of the models and cross-validation
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(corpus)
    Y = np.array(Y)
    return X, Y


X, Y = MatricesXYArticleContent(dict_pages)
ss = ShuffleSplit(n_splits=4)
lr = LogisticRegression()
sgd = SGDClassifier()
svc = LinearSVC()
mean_lr = []
mean_sgd = []
mean_svc = []
mean_xgb = []
for fold, (train, test) in enumerate(ss.split(X, Y)):
    list_rep = [(label, list(Y).count(label)) for label in set(Y[test])]
    print(f"The repartition of the test set: {list_rep}")
    # XGBoost
    if args.xgb:
        dtrain = xgb.DMatrix(X[train], label=Y[train])
        dtest = xgb.DMatrix(X[test], label=Y[test])
        param = {'objective': 'multi:softmax', 'num_class': max(set(Y)) + 1,
                 'eval_metric': 'mlogloss'}
        t_xgb = time.time()
        bst = xgb.train(param, dtrain, num_boost_round=15)
        duration_xgb = time.time() - t_xgb
        preds_xgb = bst.predict(dtest)
        score_xgb = f1_score(Y[test], preds_xgb, average='macro')
        mean_xgb.append(score_xgb)
        conf_xgb = confusion_matrix(Y[test], preds_xgb)
        str_xgb = "score XGB"
        print(f"{str_xgb:<12} fold {fold + 1}: {score_xgb:>8.2f}")
        print(f"The confusion matrix XGB: \n{conf_xgb}")
        print(f"Duration XGB: {duration_xgb}")
    t_lr = time.time()
    lr.fit(X[train], Y[train])
    duration_lr = time.time() - t_lr
    t_sgd = time.time()
    sgd.fit(X[train], Y[train])
    duration_sgd = time.time() - t_sgd
    t_svc = time.time()
    svc.fit(X[train], Y[train])
    duration_svc = time.time() - t_svc
    preds_lr = lr.predict(X[test])
    preds_sgd = sgd.predict(X[test])
    preds_svc = svc.predict(X[test])
    score_lr = f1_score(Y[test], preds_lr, average='macro')
    score_sgd = f1_score(Y[test], preds_sgd, average='macro')
    score_svc = f1_score(Y[test], preds_svc, average='macro')
    str_lr = "score LR"
    str_sgd = "score SGD"
    str_svc = "score SVC"
    print(f"{str_lr:<12} fold {fold + 1}: {score_lr:>8.2f}")
    print(f"{str_sgd:<12} fold {fold + 1}: {score_sgd:>8.2f}")
    print(f"{str_svc:<12} fold {fold + 1}: {score_svc:>8.2f}")
    mean_lr.append(score_lr)
    mean_sgd.append(score_sgd)
    mean_svc.append(score_svc)
    conf_svc = confusion_matrix(Y[test], preds_svc)
    print(f"The confusion matrix SVC: \n{conf_svc}")
    print(f"Duration SGD: {duration_sgd:.2f}")
    print(f"Duration SVC: {duration_svc:.2f}")
    print(f"Duration LRE: {duration_lr:.2f}")
print(f"The mean score LR: {np.mean(mean_lr):.3f}")
print(f"The mean score SGD: {np.mean(mean_sgd):.3f}")
print(f"The mean score SVC: {np.mean(mean_svc):.3f}")
if args.xgb:
    print(f"The mean score XGB: {np.mean(mean_xgb):.3f}")

# SVC gives the best results, but still not incredible ones (53% f1-score)
