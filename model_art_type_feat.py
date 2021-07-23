#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 17:05:52 2021

@author: baptistehessel

Script very similar to model_type_article but this time we use the usual
features.
Objective: being able to predict with a high accuracy the type of an article.

Then, model that combines the prediction of the to type of models to predict

"""
import pickle
import numpy as np
import os
os.chdir('/Users/baptistehessel/Documents/DAJ/MEP/montageIA/bin')
import model_type_article
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

class MyException(Exception):
    pass


list_features = ['nbSign', 'nbBlock', 'abstract', 'syn']
list_features += ['exergue', 'title', 'secTitle', 'supTitle']
list_features += ['subTitle', 'nbPhoto', 'aireImg', 'aireTot']
list_features += ['petittitre', 'quest_rep', 'intertitre']

path_customer = '/Users/baptistehessel/Documents/DAJ/MEP/montageIA/data/CM2/'

with open(path_customer + 'dict_pages', 'rb') as file:
    dict_pages = pickle.load(file)


# Creation of the vectors X and Y
def MatricesTypeArticle(dict_pages, list_features):
    X, Y = [], []
    for idp, dicop in dict_pages.items():
        for ida, dicta in dicop['articles'].items():
            if len(dicta['content']) > 10:
                dicta['aireTot'] = dicta['width'] * dicta['height']
                try:
                    x = np.array([dicta[ft] for ft in list_features])
                except Exception as e:
                    print(e)
                    print(dicta)
                    sys.exit()
                if dicta['isPrinc'] == 1:
                    y = 0
                elif max(dicta['isSec'], dicta['isSub']) == 1:
                    y = 1
                elif dicta['isTer'] == 1:
                    y = 2
                elif dicta['isMinor'] == 1:
                    y = 3
                else:
                    print(dicta)
                    raise MyException("No label for this article")
                X.append(x)
                Y.append(y)
    X = np.array(X)
    Y = np.array(Y)
    return X, Y


# print(validation_models.TrainValidateModel(X, Y))
# Y and Yc are the same
# For the content we use SVM
# For the features, we use RFC


##############################################################################
#                                                                            #
#                 MODEL PREDS CONTENT AND FEATURES                           #
#                                                                            #
##############################################################################


def ModelMix(X, Y, Xc, Yc):
    lr = make_pipeline(StandardScaler(with_mean=False),
                       LogisticRegression(max_iter=1000))
    rfc = RandomForestClassifier()
    lin_svc = LinearSVC()
    ss = ShuffleSplit(n_splits=3, random_state=0)
    mean_rfc = []
    mean_lr = []
    mean_lin_svc = []
    mean_mix = []
    list_classes = ['Main', 'Sec', 'Ter', 'Minor']
    for (train, test), (trainc, testc) in zip(ss.split(X, Y), ss.split(Xc, Yc)):
        # Model features
        rfc.fit(X[train], Y[train])
        preds_rfc = rfc.predict(X[test])
        probas_rfc = rfc.predict_proba(X[test])
        score_rfc = f1_score(Y[test], preds_rfc, average='macro')
        conf_svc = confusion_matrix(Y[test], preds_rfc)
        args_cl = [Y[testc], preds_rfc]
        report = classification_report(*args_cl, target_names=list_classes)
        print(f"F1-score features: {score_rfc}")
        print(f"Confusion Matrix RFC: \n{conf_svc}\n")
        print(report)
        # Model article content
        lr.fit(Xc[trainc], Yc[trainc])
        lin_svc.fit(Xc[trainc], Yc[trainc])
        preds_lin_svc = lin_svc.predict(Xc[testc])
        preds_lr = lr.predict(Xc[testc])
        probas_lr = lr.predict_proba(Xc[testc])
        # probas_svc_vs = svc_vs.predict_proba(Xc[testc])
        # Scores of the models
        score_lr = f1_score(Y[testc], preds_lr, average='macro')
        score_lin_svc = f1_score(Y[testc], preds_lin_svc, average='macro')
        conf_lr = confusion_matrix(Y[test], preds_lr)
        print(f"Confusion Matrix LR: \n{conf_lr}\n")
        print(f"F1-score LR content: {score_lr}")
        print(f"F1-score LINEAR SVC content: {score_lin_svc}")
        # Average of the probas
        print(f"probas_lr: {probas_lr.shape}")
        print(f"probas_rfc: {probas_rfc.shape}")
        mean_preds = np.mean((probas_rfc, probas_lr), axis=0)
        print(f"Shape mean_preds: {mean_preds.shape}")
        preds_avg = np.argmax(mean_preds, axis=1)
        score_avg = f1_score(Y[testc], preds_avg, average='macro')
        # Only 40%
        print(f"Score average: {score_avg}")
        # We only select the strong predictions of the model content.
        preds_mix = []
        for preds_ctt, preds_ft in zip(probas_lr, probas_rfc):
            if max(preds_ctt) > 0.85:
                if max(preds_ft) < 0.55:
                    preds_mix.append(np.argmax(preds_ctt, axis=0))
                else:
                    preds_mix.append(np.argmax(preds_ft, axis=0))
            else:
                preds_mix.append(np.argmax(preds_ft, axis=0))
        score_mix = f1_score(Y[testc], preds_mix, average='macro')
        print(f"Score mix: {score_mix}")
        mean_mix.append(score_mix)
        mean_rfc.append(score_rfc)
        mean_lr.append(score_lr)
        mean_lin_svc.append(score_lin_svc)
    # The mean predictions
    print(f"The mean score MIX: {np.mean(mean_mix)}")
    print(f"The mean score RFC: {np.mean(mean_rfc)}")
    print(f"The mean score LR: {np.mean(mean_lr)}")
    print(f"The mean score LINEAR SVC: {np.mean(mean_lin_svc)}")

mix = False
if mix:
    X, Y = MatricesTypeArticle(dict_pages, list_features)
    Xc, Yc = model_type_article.MatricesXYArticleContent(dict_pages)
    print(f"X.shape: {X.shape}")
    print(f"Y.shape: {Y.shape}")
    print(f"Xc.shape: {Xc.shape}")
    print(f"Yc.shape: {Yc.shape}")
    ModelMix(X, Y, Xc, Yc)


#%%

##############################################################################
#                                                                            #
#                          TUNING PARAMS RFC                                 #
#                                                                            #
##############################################################################

# With default parameters f1-score = 80%

rfc = RandomForestClassifier()
parameters = {
    'n_estimators': range(50, 251, 25),
    'criterion': ['gini', 'entropy'],
    }
clf = GridSearchCV(rfc, parameters, verbose=4)

X, Y = MatricesTypeArticle(dict_pages, list_features)
clf.fit(X, Y)

print(clf.best_params_)
# {'criterion': 'entropy', 'n_estimators': 150}
print(clf.best_score_)
# 0.8477
