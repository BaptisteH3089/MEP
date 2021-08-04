#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 09:25:36 2021

@author: baptistehessel

Small script to save the model for the prediction of the type of an article.
The user is supposed to have already cross-validated the model and to know
the best parameters.
For this script, you just need the object:
    - dict_pages
And the script:
    - module_art_type

Here, the best params are:
    - 'n_estimators': range(125, 251, 25)
    - 'criterion': ['gini', 'entropy']
    - 'max_depth': [None, 5, 6, 7]

By default, the model is saved under the name

"""
import pickle
import module_art_type
import argparse
from sklearn.ensemble import RandomForestClassifier

parser = argparse.ArgumentParser()
parser.add_argument('path_customer',
                    type=str,
                    help='The path to the data of the customer')
args = parser.parse_args()

if args.path_customer[-1] == '/':
    path_customer = args.path_customer
else:
    path_customer = args.path_customer + '/'

with open(path_customer, 'rb') as f:
    dict_pages = pickle.load(f)


list_features = ['nbSign', 'nbBlock', 'abstract', 'syn']
list_features += ['exergue', 'title', 'secTitle', 'supTitle']
list_features += ['subTitle', 'nbPhoto', 'aireImg', 'aireTot']
list_features += ['petittitre', 'quest_rep', 'intertitre']

# We build the matrices X and Y
X, Y = module_art_type.MatricesTypeArticle(dict_pages, list_features)
best_params = {'n_estimators': 150,
               'criterion': 'entropy',
               'min_samples_split': 8,
               'max_features': 9}

clf = RandomForestClassifier(**best_params)
clf.fit(X, Y)
print(f"The mean accuracy on the train data: {clf.score(X, Y):->30.3f}")

# We save the model in the same repertory as the data and the other models.
with open(path_customer + 'rfc_pred_type_art', 'wb') as f:
    pickle.dump(clf, f)




