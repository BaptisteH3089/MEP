#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 17:32:16 2021

@author: baptistehessel

Contains just the function to train and validates some multiclass
classification models

"""
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC
from sklearn import svm
import time
import numpy as np
import xgboost as xgb


def TrainValidateModel(X, Y):
    mean_rdc, mean_svc, mean_lsvc = [], [], []
    mean_sgd, mean_gnb = [], []
    mean_logrepnop = []
    mean_gbc = []
    mean_xgb = []
    dict_duration = {}
    ss = ShuffleSplit(n_splits=4)
    for i, (train, test) in enumerate(ss.split(X, Y)):
        # XGBoost
        t_xgb = time.time()
        dtrain = xgb.DMatrix(X[train], label=Y[train])
        dtest = xgb.DMatrix(X[test], label=Y[test])
        param = {'objective': 'multi:softmax',
                 'num_class': max(set(Y)) + 1,
                 'eval_metric': 'mlogloss'}
        print(f"param['num_class']: {param['num_class']}")
        bst = xgb.train(param, dtrain, num_boost_round=15)
        preds_xgb = bst.predict(dtest)
        score_xgb = f1_score(Y[test], preds_xgb, average='macro')
        mean_xgb.append(score_xgb)
        dict_duration['xgb'] = time.time() - t_xgb
        # Random Forest
        t_rfc = time.time()
        rdc = RandomForestClassifier().fit(X[train], Y[train])
        preds_rdc = rdc.predict(X[test])
        score_rdc = f1_score(Y[test], preds_rdc, average='macro')
        mean_rdc.append(score_rdc)
        dict_duration['rfc'] = time.time() - t_rfc
        # SVC
        t_svc = time.time()
        svc = svm.SVC().fit(X[train], Y[train])
        preds_svc = svc.predict(X[test])
        score_svc = f1_score(Y[test], preds_svc, average='macro')
        mean_svc.append(score_svc)
        dict_duration['svc'] = time.time() - t_svc
        # Linear SVC
        t_linsvc = time.time()
        linear_svc = make_pipeline(StandardScaler(), LinearSVC())
        linear_svc.fit(X[train], Y[train])
        preds_lsvc = linear_svc.predict(X[test])
        score_lsvc = f1_score(Y[test], preds_lsvc, average='macro')
        mean_lsvc.append(score_lsvc)
        dict_duration['lin_svc'] = time.time() - t_linsvc
        # SGD
        t_sgd = time.time()
        sgd = make_pipeline(StandardScaler(), SGDClassifier())
        sgd.fit(X[train], Y[train])
        preds_sgd = sgd.predict(X[test])
        score_sgd = f1_score(Y[test], preds_sgd, average='macro')
        mean_sgd.append(score_sgd)
        dict_duration['sgd'] = time.time() - t_sgd
        # Gaussian Naive Bayes
        t_gnb = time.time()
        gnb = GaussianNB().fit(X[train], Y[train])
        preds_gnb = gnb.predict(X[test])
        score_gnb = f1_score(Y[test], preds_gnb, average='macro')
        mean_gnb.append(score_gnb)
        dict_duration['gnb'] = time.time() - t_gnb
        # Logistic regression default - No penalty
        t_log = time.time()
        param = {'max_iter': 1000}
        param['penalty'] = 'none'
        logregnop = make_pipeline(StandardScaler(), LogisticRegression(**param))
        logregnop.fit(X[train], Y[train])
        preds_logregnop = logregnop.predict(X[test])
        score_logregnop = f1_score(Y[test], preds_logregnop, average='macro')
        mean_logrepnop.append(score_logregnop)
        dict_duration['log'] = time.time() - t_log
        # Gradient Boosting Classifier
        t_gbc = time.time()
        gbc = GradientBoostingClassifier().fit(X[train], Y[train])
        preds_gbc = gbc.predict(X[test])
        score_gbc = f1_score(Y[test], preds_gbc, average='macro')
        mean_gbc.append(score_gbc)
        dict_duration['gbc'] = time.time() - t_gbc
        print("FOLD: {}.".format(i))
        print("{:<50} {:>30.4f}.".format("score XGBoost", score_xgb))
        print("{:<50} {:>30.4f}.".format("score rdmForest", score_rdc))
        print("{:<50} {:>30.4f}.".format("score GBC", score_gbc))
        print("{:<50} {:>30.4f}.".format("score SVC", score_svc))
        print("{:<50} {:>30.4f}.".format("score Linear SVC", score_lsvc))
        print("{:<50} {:>30.4f}.".format("score SGD", score_sgd))
        print("{:<50} {:>30.4f}.".format("score GNB", score_gnb))
        print("{:<50} {:>30.4f}.".format("score Log", score_logregnop))
        print('\n')
    all_means = [mean_rdc, mean_svc, mean_lsvc, mean_sgd, mean_gnb]
    all_means += [mean_logrepnop, mean_gbc, mean_xgb]
    str_means = ['mean_rdc', 'mean_svc','mean_lsvc', 'mean_sgd', 'mean_gnb']
    str_means += ['mean_logrepnop', 'mean_gbc', 'mean_xgb']
    print(("{:-^80}".format("GLOBAL MEANS")))
    for list_mean, str_mean in zip(all_means, str_means):
        print("{:<35} {:>15.3f}".format(str_mean, np.mean(list_mean)))
    print(("{:-^80}".format("DURATION MODELS")))
    for key, value in dict_duration.items():
        print("{:<35} {:>15.3f}".format(key, value))
    return "End scores model"
