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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC
import xgboost as xgb
import numpy as np
import pickle
import time
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import module_art_type


parser = argparse.ArgumentParser()
parser.add_argument('path_customer',
                    type=str,
                    help='The path to the data.')
parser.add_argument('path_objects',
                    type=str,
                    help='The path to the objects to lemmatize.')
parser.add_argument('--content',
                    help='Whether to validate the model with the content.',
                    action='store_true')
parser.add_argument('--features',
                    help='Whether to validate the model with the features.',
                    action='store_true')
parser.add_argument('--max_features',
                    type=int,
                    help='The max number of features for tfidf vect.',
                    default=10000)
parser.add_argument('--xgb',
                    help='Whether to use XGBoost.',
                    action='store_true')
parser.add_argument('--blend',
                    help='Whether to validate blended models.',
                    action='store_true')
parser.add_argument('--opti',
                    help='Whether to optimize blended models.',
                    action='store_true')
parser.add_argument('--concat',
                    help='Whether to validate the model concatenation.',
                    action='store_true')
parser.add_argument('--tuning',
                    help='Whether to perform a grid-search.',
                    action='store_true')
args = parser.parse_args()
print("{:-^80}".format("All the arguments"))
print(f"args.content: {args.content}")
print(f"args.features: {args.features}")
print(f"args.max_features: {args.max_features}")
print(f"args.xgb: {args.xgb}")
print(f"args.blend: {args.blend}")
print(f"args.opti: {args.opti}")
print(f"args.concat: {args.concat}")
print(f"args.tuning: {args.tuning}")

if args.path_customer[-1] == '/':
    path_customer = args.path_customer
else:
    path_customer = args.path_customer + '/'
if args.path_objects[-1] == '/':
    path_objects = args.path_objects
else:
    path_objects = args.path_objects + '/'


# List of stop-words.
with open(path_objects + 'Avoid', 'rb') as file:
    avoid = pickle.load(file)
# Dictionary uses to remplace words by their lemma.
with open(path_objects + 'Dico', 'rb') as file:
    dict_lem = pickle.load(file)
# Usual dictionary with all the pages.
with open(path_customer + 'dict_pages', 'rb') as file:
    dict_pages = pickle.load(file)


##############################################################################
#                                                                            #
#              Validation of the model with the content                      #
#                                                                            #
##############################################################################

param = {'dict_pages': dict_pages, 'dict_lem': dict_lem,
         'max_features': args.max_features}
Xc, Yc = module_art_type.MatricesXYArticleContent(**param)
if args.content:
    print("{:-^80}".format("CONTENT"))
    print(f"args content: {args.content}")
    ss = ShuffleSplit(n_splits=3)
    lr = LogisticRegression()
    sgd = SGDClassifier()
    svc = LinearSVC()
    mean_lr = []
    mean_sgd = []
    mean_svc = []
    mean_xgb = []
    duration_lr = 0
    duration_xgb = 0
    duration_svc = 0
    duration_sgd = 0
    for fold, (train, test) in enumerate(ss.split(Xc, Yc)):
        list_rep = [(label, list(Yc).count(label)) for label in set(Yc[test])]
        print(f"The repartition of the test set: {list_rep}")
        # XGBoost
        if args.xgb:
            dtrain = xgb.DMatrix(Xc[train], label=Yc[train])
            dtest = xgb.DMatrix(Xc[test], label=Yc[test])
            param = {'objective': 'multi:softmax', 'num_class': max(set(Yc)) + 1,
                     'eval_metric': 'mlogloss'}
            t_xgb = time.time()
            bst = xgb.train(param, dtrain, num_boost_round=15)
            duration_xgb += time.time() - t_xgb
            preds_xgb = bst.predict(dtest)
            score_xgb = f1_score(Yc[test], preds_xgb, average='macro')
            mean_xgb.append(score_xgb)
            conf_xgb = confusion_matrix(Yc[test], preds_xgb)
            str_xgb = "score XGB"
            print(f"{str_xgb:<12} fold {fold + 1}: {score_xgb:>8.2f}")
            print(f"The confusion matrix XGB: \n{conf_xgb}")

        t_lr = time.time()
        lr.fit(Xc[train], Yc[train])
        duration_lr += time.time() - t_lr

        t_sgd = time.time()
        sgd.fit(Xc[train], Yc[train])
        duration_sgd += time.time() - t_sgd

        t_svc = time.time()
        svc.fit(Xc[train], Yc[train])
        duration_svc += time.time() - t_svc

        preds_lr = lr.predict(Xc[test])
        preds_sgd = sgd.predict(Xc[test])
        preds_svc = svc.predict(Xc[test])

        score_lr = f1_score(Yc[test], preds_lr, average='macro')
        score_sgd = f1_score(Yc[test], preds_sgd, average='macro')
        score_svc = f1_score(Yc[test], preds_svc, average='macro')

        str_lr = "score LR"
        str_sgd = "score SGD"
        str_svc = "score SVC"

        print(f"{str_lr:<12} fold {fold + 1}: {score_lr:>8.2f}")
        print(f"{str_sgd:<12} fold {fold + 1}: {score_sgd:>8.2f}")
        print(f"{str_svc:<12} fold {fold + 1}: {score_svc:>8.2f}")

        mean_lr.append(score_lr)
        mean_sgd.append(score_sgd)
        mean_svc.append(score_svc)
        conf_svc = confusion_matrix(Yc[test], preds_svc)
        args_rep = [Yc[test], preds_svc]
        list_classes = ['Main', 'Sec', 'Ter', 'Minor']
        report = classification_report(*args_rep, target_names=list_classes)
        print(f"The confusion matrix SVC: \n{conf_svc}\n")
        print(f"The classification report: \n{report}\n")
        print(f"Duration SGD: {duration_sgd/(fold+1):.2f}")
        print(f"Duration SVC: {duration_svc/(fold+1):.2f}")
        print(f"Duration LRE: {duration_lr/(fold+1):.2f}")
        if args.xgb:
            print(f"Duration XGB: {duration_xgb/(fold+1):.2f}")
    print(f"The mean score LR: {np.mean(mean_lr):.3f}")
    print(f"The mean score SGD: {np.mean(mean_sgd):.3f}")
    print(f"The mean score SVC: {np.mean(mean_svc):.3f}")
    if args.xgb:
        print(f"The mean score XGB: {np.mean(mean_xgb):.3f}")

# SVC gives the best results, but still not incredible ones (53% f1-score)


##############################################################################
#                                                                            #
#              Validation of the model with the features                     #
#                                                                            #
##############################################################################

list_features = ['nbSign', 'nbBlock', 'abstract', 'syn']
list_features += ['exergue', 'title', 'secTitle', 'supTitle']
list_features += ['subTitle', 'nbPhoto', 'aireImg', 'aireTot']
list_features += ['petittitre', 'quest_rep', 'intertitre']
X, Y = module_art_type.MatricesTypeArticle(dict_pages, list_features)
print(f"X.shape: {X.shape}")
print(f"Y.shape: {Y.shape}")
if args.features:
    print("{:-^80}".format("FEATURES"))
    print(f"args features: {args.features}")
    module_art_type.ModelFeatures(X, Y)

if args.blend:
    print("{:-^80}".format("BLENDED"))
    print(f"args blended: {args.blend}")
    module_art_type.ModelMix(X, Y, Xc, Yc)

if args.opti:
    print("{:-^80}".format("OPTI"))
    print(f"args opti: {args.opti}")
    best_params = module_art_type.OptiThresholds(X, Y, Xc, Yc)


##############################################################################
#                                                                            #
#                     TUNING PARAMS RFC - FEATURES                           #
#                                                                            #
##############################################################################

if args.tuning:
    print("{:-^80}".format("TUNING"))
    print(f"args tuning: {args.tuning}")
    # With default parameters f1-score = 80%
    rfc = RandomForestClassifier()
    parameters = {'n_estimators': range(150, 251, 20),
                  'criterion': ['gini', 'entropy']}
    clf = GridSearchCV(rfc, parameters, verbose=4, scoring='f1_macro')

    X, Y = module_art_type.MatricesTypeArticle(dict_pages, list_features)
    print(f"A line vector of X: {X[0]}")
    clf.fit(X, Y)
    print(f"The best parameters: {clf.best_params_}")
    print(f"The best scores: {clf.best_score_}")
    # {'criterion': 'entropy', 'n_estimators': 150}
    # 0.8477
    # A validation of the model with the best parameters
    list_classes = ['Main', 'Sec', 'Ter', 'Minor']
    print("{:-^80}".format("Validation of the model with best params"))
    rfc = RandomForestClassifier(**clf.best_params_)
    ss = ShuffleSplit(n_splits=5)
    mean_best = []
    for i, (train, test) in enumerate(ss.split(X, Y)):
        rfc.fit(X[train], Y[train])
        preds = rfc.predict(X[test])
        score = f1_score(Y[test], preds, average='macro')
        mean_best.append(score)
        conf_best = confusion_matrix(Y[test], preds)
        args_rep = [Y[test], preds]
        report_best = classification_report(*args_rep, target_names=list_classes)
        print(f"[fold {i}] F1-score best params RFC: {score:.3f}")
        print(f"The classification report: \n{report_best}\n")
        print(f"The confusion matrix: \n{conf_best}\n")
    print(f"The mean score of the best model: {np.mean(mean_best):.3f}")


if args.concat:
    print("{:-^80}".format("CONCAT"))
    print(f"args concat: {args.concat}")
    # Essai en concaténant les features et le contenu
    print(f"The shape of X content: {Xc.shape}")
    x = input("Try to contatenate with vectors features (long)? y/n ")
    if x == 'y':
        Xc = Xc.toarray()
        newX = np.concatenate((X, Xc), axis=1)
        rfc = RandomForestClassifier()
        ss = ShuffleSplit(n_splits=3)
        for i, (train, test) in enumerate(ss.split(newX, Y)):
            rfc.fit(newX[train], Y[train])
            preds = rfc.predict(newX[test])
            score = f1_score(Y[test], preds, average='macro')
            print(f"[fold {i}] F1-score concatenation RFC: {score:.3f}")
        # The results are not good, better than just the content but worst
        # than with only the features.
