#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from operator import itemgetter
from matplotlib import patches
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, RepeatedKFold, StratifiedKFold
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.metrics import f1_score
import xml.etree.cElementTree as ET
from bs4 import BeautifulSoup as bs
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import collections as col
import numpy as np
import pickle, math, time, os, re

path_customer = '/Users/baptistehessel/Documents/DAJ/MEP/montageIA/data/CM/'

# The dict with all the pages available
with open(path_customer + 'dico_pages', 'rb') as file:
    dico_bdd = pickle.load(file)
# The dict {ida: dicoa, ...}
with open(path_customer + 'dico_arts', 'rb') as file:
    dict_arts = pickle.load(file)
# The list of triplet (nb_pages_using_mdp, array_mdp, list_ids)
with open(path_customer + 'list_mdp', 'rb') as file:
    list_mdp_data = pickle.load(file)


def SelectionModelesPages(l_model, nb_art, nb_min_pages):
    f = lambda x: (x[1].shape[0] == nb_art) & (x[0] >= nb_min_pages)
    return list(filter(f, l_model))


def GetYClasses(big_y):
    n = big_y.shape[0]
    return [(big_y[i], i) for i in range(n)]


def TransformArrayIntoClasses(big_y, list_classes):
    n = big_y.shape[0]
    little_y = np.zeros(n)
    for i in range(n):
        if max(abs(big_y[i] - list_classes[i][0])) < 20:
            little_y[i] = list_classes[i][1]
        else:
            for j in range(n):
                if max(abs(big_y[i] - list_classes[j][0])) < 20:
                    little_y[i] = list_classes[j][1]
                    break
    return little_y


def GetTrainableData(dico_bdd, dict_arts, list_id_pages, y_ref, list_features):
    # Transformation de big_y into simpler classes
    classes = GetYClasses(y_ref)
    for i, id_page in enumerate(list_id_pages):
        dicoarts = dico_bdd[id_page]['dico_page']['articles']
        for j, (ida, dicoa) in enumerate(dicoarts.items()):
            vect_art = [dict_arts[ida][x] for x in list_features]
            y_art = [dict_arts[ida][x] for x in ['x', 'y', 'width', 'height']]
            y_array = np.array(y_art)
            f = lambda x: np.allclose(x[0], y_array, atol=20)
            try:
                label_y = list(filter(f, classes))[0][1]
            except Exception as e:
                print(e)
                print("We didn't find the label of y")
                print(f"y_array: {y_array}")
                print(f"classes: {classes}")
                raise e
            if i + j == 0:
                very_big_x = np.array(vect_art, ndmin=2)
                very_big_y = [label_y]
            else:
                x = np.array(vect_art, ndmin=2)
                very_big_x = np.concatenate((very_big_x, x))
                very_big_y.append(label_y)
    return very_big_x, np.array(very_big_y)


#%%

def TrainValidateModels(dico_bdd,
                        list_id_pages,
                        y_ref):
    t0 = time.time()
    args = [dico_bdd, dict_arts, list_id_pages, y_ref, list_features]
    X, Y = GetTrainableData(*args)
    clf = RandomForestClassifier()
    ss = ShuffleSplit(n_splits=4)
    res_small_vector = []
    dict_duration = {}
    mean_rfc = []
    mean_svc = []
    mean_lsvc = []
    mean_sgd = []
    mean_gnb = []
    mean_logreg = []
    mean_gbc = []
    for k, (train, test) in enumerate(ss.split(X, Y)):
        print("FOLD: {}.".format(k))
        # Gradient Boosting Classifier
        t_gbc = time.time()
        gbc = GradientBoostingClassifier().fit(X[train], Y[train])
        preds_gbc = gbc.predict(X[test])
        score_gbc = f1_score(Y[test], preds_gbc, average='macro')
        mean_gbc.append(score_gbc)
        dict_duration['gbc'] = time.time() - t_gbc
        # RFC
        t_rfc = time.time()
        clf.fit(X[train], Y[train])
        y_preds = clf.predict(X[test])
        score_rfc = f1_score(Y[test], y_preds, average='macro')
        mean_rfc.append(score_rfc)
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
        # Logistic regression default 
        t_log = time.time()
        logreg = make_pipeline(StandardScaler(), LogisticRegression())
        logreg.fit(X[train], Y[train])
        preds_logreg = logreg.predict(X[test])
        score_logreg = f1_score(Y[test], preds_logreg, average='macro')
        mean_logreg.append(score_logreg)
        dict_duration['log'] = time.time() - t_log
    list_scores = [mean_gbc, mean_rfc, mean_svc, mean_lsvc, mean_sgd]
    list_scores += [mean_gnb, mean_logreg]
    list_means = list(map(lambda x: np.mean(x), list_scores))
    list_mins = list(map(lambda x: min(x), list_scores))
    print("{:<50} {:>30.4f}.".format("GBC", list_means[0]))
    print("{:<50} {:>30.4f}.".format("rdmForest", list_means[1]))
    print("{:<50} {:>30.4f}.".format("SVC", list_means[2]))
    print("{:<50} {:>30.4f}.".format("Linear SVC", list_means[3]))
    print("{:<50} {:>30.4f}.".format("SGD", list_means[4]))
    print("{:<50} {:>30.4f}.".format("GNB", list_means[5]))
    print("{:<50} {:>30.4f}.".format("Log Reg", list_means[6]))
    return list_mins, list_means, dict_duration


#%%

list_features = ['nbSign', 'nbBlock', 'abstract', 'syn']
list_features += ['exergue', 'title', 'secTitle', 'supTitle']
list_features += ['subTitle', 'nbPhoto', 'aireImg', 'aireTot']
list_features += ['petittitre', 'quest_rep', 'intertitre']
# On sélectionne le nb d'articles qu'on veut avec un nb min de pages pour
# entraîner le modèle
pages_p_art = SelectionModelesPages(list_mdp_data, 3, 20)

all_durations, all_means, all_mins = [], [], []
for nb_pages, big_y, list_id_pages in pages_p_art:
    args = [dico_bdd, list_id_pages, big_y]
    list_mins, list_means, dict_duration = TrainValidateModels(*args)
    all_durations.append(dict_duration)
    all_means.append(list_means)
    all_mins.append(list_mins)

print("Number of layout found : {}".format(len(pages_p_art)))
total_pages = sum((len(elt[2]) for elt in pages_p_art))
print("Number of pages found : {}".format(total_pages))
dict_means_duration = {}
for classifier in all_durations[0].keys():
    list_durat = [dict_d[classifier] for dict_d in all_durations]
    dict_means_duration[classifier] = np.mean(list_durat)
global_min = np.min(all_mins, axis=0)
global_means = np.mean(all_means, axis=0)
keys = all_durations[0].keys()
args_zip = [global_min, global_means, keys, dict_means_duration.values()]
for min_score, mean_score, clfr, duration in zip(*args_zip):
    print(f"{clfr:<10} {min_score:8.3f} {mean_score:8.3f} {duration:8.3f}")












