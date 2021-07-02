#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
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
import os
os.chdir('/Users/baptistehessel/Documents/DAJ/MEP/montageIA/bin')
import without_layout
import time
import numpy as np
import pickle


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


def AireTotArts(dico_bdd, nb_arts):
    list_artTot = []
    for key, dicop in dico_bdd.items():
        if len(dicop['dico_page']['articles']) == nb_arts:
            list_aireTot_art = []
            for ida, art in dicop['dico_page']['articles'].items():
                list_aireTot_art.append(art['aireTot'])
            list_artTot.append(list_aireTot_art)
    print("The number of pages with 3 articles: {}".format(len(list_artTot)))
    moy, nb_pages, nb_inf, nb_sup = 0, 0, 0, 0
    for list_pg in list_artTot:
        if sum(list_pg) < 20000: nb_inf += 1
        elif sum(list_pg) > 130000: nb_sup += 1
        else:
            moy += sum(list_pg)
            nb_pages += 1
    list_sums = [sum(list_pg) for list_pg in list_artTot]
    list_filter = list(filter(lambda x: 20000 < x < 130000, list_sums))
    mean_area = moy / nb_pages
    print("The mean aireTot for {} arts: {}".format(nb_arts, mean_area))
    print("The min area: {}".format(min(list_filter)))
    print("The max area: {}".format(max(list_filter)))
    return list_artTot


def SelectionArts(dico_bdd, nb_arts):
    list_dico_arts = []
    for key, dicop in dico_bdd.items():
        if len(dicop['dico_page']['articles']) == nb_arts:
            for ida, art in dicop['dico_page']['articles'].items():
                dict_elts_art = {}
                dict_elts_art['aireArt'] = art['width'] * art['height']
                dict_elts_art['nbPhoto'] = art['nbPhoto']
                dict_elts_art['score'] = art['score']
                dict_elts_art['melodyId'] = art['melodyId']
                list_dico_arts.append(dict_elts_art)
    return list_dico_arts


# if 90000 <= art['aireTot'] <= 120000:
# Si on commnence par les pages avec 3 articles
# A priori marge de tolérance de 5%
# On va plutôt générer tous les triplets possibles. Bourrin mais efficace.
def GenerationAllNTuple(list_dico_arts, len_sample, nb_arts):
    rdm_list = list(np.random.choice(list_dico_arts, size=len_sample))
    list_feat = ['aireArt', 'nbPhoto', 'score', 'melodyId']
    select_arts = [[dicoa[x] for x in list_feat] for dicoa in rdm_list]
    print("{:-^75}".format("The selection of arts"))
    for i, list_art in enumerate(select_arts):
        print("{:<15} {}".format(i + 1, list_art))
    print("{:-^75}".format("End of the selection of arts"))
    if nb_arts == 2:
        all_ntuple = [(x, y) for i, x in enumerate(select_arts)
                      for y in select_arts[i + 1:]]
    elif nb_arts == 3:
        all_ntuple = [(x, y, z) for i, x in enumerate(select_arts)
                      for j, y in enumerate(select_arts[i + 1:])
                      for z in select_arts[i + j + 2:]]
    elif nb_arts == 4:
        all_ntuple = [(a, b, c, d)
                      for i, a in enumerate(select_arts)
                      for j, b in enumerate(select_arts[i + 1:])
                      for k, c in enumerate(select_arts[i + j + 2:])
                      for d in select_arts[i + j + k + 3:]]
    elif nb_arts == 5:
        all_ntuple = [(a, b, c, d, e)
                      for i, a in enumerate(select_arts)
                      for j, b in enumerate(select_arts[i + 1:])
                      for k, c in enumerate(select_arts[i + j + 2:])
                      for l, d in enumerate(select_arts[i+j+k+3:])
                      for e in select_arts[i+j+k+l+4:]]
    else:
        str_exc = 'Wrong nb of arts: {}. Must be in [2, 5]'
        raise MyException(str_exc.format(nb_arts))
    print("The total nb of ntuples generated: {}".format(len(all_ntuple)))
    return all_ntuple


def TrainValidateModel(X, Y):
    mean_rdc, mean_svc, mean_lsvc = [], [], []
    mean_sgd, mean_gnb, mean_logregd, mean_logreg = [], [], [], []
    mean_logrepnop, mean_logregl1 = [], []
    mean_regel95, mean_regel85 = [], []
    mean_gbc = []
    dict_duration = {}
    ss = ShuffleSplit(n_splits=4)
    for i, (train, test) in enumerate(ss.split(X, Y)):
        # Random Forest
        t_rfc = time.time()
        rdc = RandomForestClassifier().fit(X[train], Y[train])
        preds_rdc = rdc.predict(X[test])
        score_rdc_w = f1_score(Y[test], preds_rdc, average='weighted')
        score_rdc = f1_score(Y[test], preds_rdc, average='macro')
        mean_rdc.append(score_rdc)
        dict_duration['rfc'] = time.time() - t_rfc
        # SVC
        t_svc = time.time()
        svc = svm.SVC().fit(X[train], Y[train])
        preds_svc = svc.predict(X[test])
        score_svc_w = f1_score(Y[test], preds_svc, average='weighted')
        score_svc = f1_score(Y[test], preds_svc, average='macro')
        mean_svc.append(score_svc)
        dict_duration['svc'] = time.time() - t_svc
        # Linear SVC
        t_linsvc = time.time()
        linear_svc = make_pipeline(StandardScaler(), LinearSVC())
        linear_svc.fit(X[train], Y[train])
        preds_lsvc = linear_svc.predict(X[test])
        score_lsvc_w = f1_score(Y[test], preds_lsvc, average='weighted')
        score_lsvc = f1_score(Y[test], preds_lsvc, average='macro')
        mean_lsvc.append(score_lsvc)
        dict_duration['lin_svc'] = time.time() - t_linsvc
        # SGD
        t_sgd = time.time()
        sgd = make_pipeline(StandardScaler(), SGDClassifier())
        sgd.fit(X[train], Y[train])
        preds_sgd = sgd.predict(X[test])
        score_sgd_w = f1_score(Y[test], preds_sgd, average='weighted')
        score_sgd = f1_score(Y[test], preds_sgd, average='macro')
        mean_sgd.append(score_sgd)
        dict_duration['sgd'] = time.time() - t_sgd
        # Gaussian Naive Bayes
        t_gnb = time.time()
        gnb = GaussianNB().fit(X[train], Y[train])
        preds_gnb = gnb.predict(X[test])
        score_gnb_w = f1_score(Y[test], preds_gnb, average='weighted')
        score_gnb = f1_score(Y[test], preds_gnb, average='macro')
        mean_gnb.append(score_gnb)
        dict_duration['gnb'] = time.time() - t_gnb
        # Logistic regression default - slightly better max_ter=1000
        # t_log1000 = time.time()
        # param = {'max_iter': 1000}
        # logreg = make_pipeline(StandardScaler(), LogisticRegression(**param))
        # logreg.fit(X[train], Y[train])
        # preds_logreg = logreg.predict(X[test])
        # score_logreg_w = f1_score(Y[test], preds_logreg, average='weighted')
        # score_logreg = f1_score(Y[test], preds_logreg, average='macro')
        # mean_logreg.append(score_logreg)
        # dict_duration['log_1000'] = time.time() - t_log1000
        # Logistic regression default - No penalty
        t_log = time.time()
        param = {'max_iter': 1000}
        param['penalty'] = 'none'
        logregnop = make_pipeline(StandardScaler(), LogisticRegression(**param))
        logregnop.fit(X[train], Y[train])
        preds_logregnop = logregnop.predict(X[test])
        score_logregnop_w = f1_score(Y[test], preds_logregnop, average='weighted')
        score_logregnop = f1_score(Y[test], preds_logregnop, average='macro')
        mean_logrepnop.append(score_logregnop)
        dict_duration['log'] = time.time() - t_log
        # Logistic regression elasticnet
        # t_l95 = time.time()
        # param['l1_ratio'] = 0.95
        # param['penalty'] = 'elasticnet'
        # param['solver'] = 'saga'
        # regel95 = make_pipeline(StandardScaler(), LogisticRegression(**param))
        # regel95.fit(X[train], Y[train])
        # preds_regel95 = regel95.predict(X[test])
        # score_regel95_w = f1_score(Y[test], preds_regel95, average='weighted')
        # score_regel95 = f1_score(Y[test], preds_regel95, average='macro')
        # mean_regel95.append(score_regel95)
        # dict_duration['l95'] = time.time() - t_l95
        # Logistic regression elasticnet 0.85
        # t_l85 = time.time()
        # param['l1_ratio'] = 0.85
        # regel85 = make_pipeline(StandardScaler(), LogisticRegression(**param))
        # regel85.fit(X[train], Y[train])
        # preds_regel85 = regel85.predict(X[test])
        # score_regel85_w = f1_score(Y[test], preds_regel85, average='weighted')
        # score_regel85 = f1_score(Y[test], preds_regel85, average='macro')
        # mean_regel85.append(score_regel85)
        # dict_duration['l85'] = time.time() - t_l85
        # Gradient Boosting Classifier
        t_gbc = time.time()
        gbc = GradientBoostingClassifier().fit(X[train], Y[train])
        preds_gbc = gbc.predict(X[test])
        score_gbc_w = f1_score(Y[test], preds_gbc, average='weighted')
        score_gbc = f1_score(Y[test], preds_gbc, average='macro')
        mean_gbc.append(score_gbc)
        dict_duration['gbc'] = time.time() - t_gbc
        print("FOLD: {}.".format(i))
        print("{:<50} {:>30.4f}.".format("score rdmForest", score_rdc))
        print("{:<50} {:>30.4f}.".format("score GBC", score_gbc))
        print("{:<50} {:>30.4f}.".format("score SVC", score_svc))
        print("{:<50} {:>30.4f}.".format("score Linear SVC", score_lsvc))
        print("{:<50} {:>30.4f}.".format("score SGD", score_sgd))
        print("{:<50} {:>30.4f}.".format("score GNB", score_gnb))
        print("{:<50} {:>30.4f}.".format("score Log", score_logregnop))
        # str_sc = "score LogReg max_iter=1000"
        # print("{:<50} {:>30.4f}.".format(str_sc, score_logreg))
        # str_sc1 = str_sc + " elasticnet 0.95"
        # print("{:<50} {:>30.4f}.".format(str_sc1, score_regel95))
        # str_sc2 = str_sc + " elasticnet 0.85"
        # print("{:<50} {:>30.4f}.".format(str_sc2, score_regel85))
        print('\n')
    all_means = [mean_rdc, mean_svc, mean_lsvc, mean_sgd, mean_gnb]
    all_means += [mean_logrepnop, mean_gbc]
    str_means = ['mean_rdc', 'mean_svc','mean_lsvc', 'mean_sgd', 'mean_gnb']
    str_means += ['mean_logrepnop', 'mean_gbc']
    print(("{:-^80}".format("GLOBAL MEANS")))
    for list_mean, str_mean in zip(all_means, str_means):
        print("{:<35} {:>15.3f}".format(str_mean, np.mean(list_mean)))
    print(("{:-^80}".format("DURATION MODELS")))
    for key, value in dict_duration.items():
        print("{:<35} {:>15.3f}".format(key, value))
    return "End scores model"


def GlobalValidateModel(dico_bdd, dict_arts, list_mdp_data, nb_arts, nb_pages_min):
    dico_narts = SelectionPagesNarts(dico_bdd, nb_arts)
    list_mdp_narts = SelectionMDPNarts(list_mdp_data, nb_arts)
    print(VerificationPageMDP(dico_narts, list_mdp_narts))
    args_xy = [dico_narts, dict_arts, list_mdp_narts, list_features, nb_pages_min]
    vect_XY = CreationVectXY(*args_xy)
    dict_labels = CreationDictLabels(vect_XY)
    X, Y = CreationXY(vect_XY, dict_labels)
    print(TrainValidateModel(X, Y))
    return "End of the Global Validation of the models"


def TuningHyperParamGBC(dico_bdd, dict_arts, list_mdp_data, nb_arts, nb_pages_min, list_feats):
    dico_narts = SelectionPagesNarts(dico_bdd, nb_arts)
    list_mdp_narts = SelectionMDPNarts(list_mdp_data, nb_arts)
    print(VerificationPageMDP(dico_narts, list_mdp_narts))
    args_xy = [dico_narts, dict_arts, list_mdp_narts, list_feats, nb_pages_min]
    vect_XY = CreationVectXY(*args_xy)
    dict_labels = CreationDictLabels(vect_XY)
    X, Y = CreationXY(vect_XY, dict_labels)
    parameters = {'n_estimators': range(50, 200, 10),
                  'max_depth': range(3, 7),
                  'criterion': ['mse', 'friedman_mse'],
                  'learning_rate': np.arange(0.05, .2, 0.05),
                  'subsample': np.arange(.5, 1, .25),
                  'min_samples_split': np.arange(.5, 1, .25),
                  'min_samples_leaf': np.arange(.15, .5, .15)}
    parameters = {'n_estimators': range(50, 200, 30),
                  'max_depth': range(3, 6)}
    gbc = GradientBoostingClassifier()
    clf = RandomizedSearchCV(gbc, parameters, verbose=3, n_iter=100,
                             scoring='f1_weighted')
    # clf = GridSearchCV(gbc, parameters, verbose=3, scoring='f1_weighted')
    print(clf.fit(X, Y))
    print('clf.best_params_', clf.best_params_)
    print('clf.best_score_', clf.best_score_)
    return True


def GetSelectionOfArticles(dico_bdd, nb_arts, len_sample):
    """
    Only used to validate the models with our data.
    """
    list_dico_arts = SelectionArts(dico_bdd, nb_arts)
    print("The number of articles selected: {}".format(len(list_dico_arts)))
    all_ntuples = GenerationAllNTuple(list_dico_arts, len_sample, nb_arts)
    # Use of the constraints 
    possib_area = without_layout.ConstraintArea(all_ntuples)
    possib_imgs = without_layout.ConstraintImgs(all_ntuples)
    possib_score = without_layout.ConstraintScore(all_ntuples)
    # Intersection of each result obtained
    final_obj = set(possib_score) & set(possib_imgs) & set(possib_area)
    str_prt = "The number of triplets that respect the constraints: {}"
    print(str_prt.format(len(final_obj)))
    return final_obj


# I need to find the vectors article associated to the ids I just got
# Creation of the vectors page for each triplet
def CreationListVectorsPage(final_obj, dict_arts, list_feat):
    """
    Only to validate the model with the labelled data
    """
    list_vect_page = []
    for triplet in final_obj:
        for i, ida in enumerate(triplet):
            vect_art = np.array([dict_arts[ida][x] for x in list_feat], ndmin=2)
            if i == 0:
                vect_page = vect_art
            else:
                vect_page = np.concatenate((vect_page, vect_art), axis=1)
        list_vect_page.append(vect_page)
    return list_vect_page


def FilterSmallClasses(X, Y, min_nb):
    # Il faudrait enlever les classes où il y a moins de X représentants
    init = False
    for i, (line_X, elt_Y) in enumerate(zip(X, Y)):
        if i == 0:
            if dict_nblabels[elt_Y] > min_nb:
                Yn = [elt_Y]
                Xn = np.array(line_X, ndmin=2)
                init = True
        else:
            if dict_nblabels[elt_Y] > min_nb:
                if init is True:
                    Yn.append(elt_Y)
                    Xn = np.concatenate((Xn, np.array(line_X, ndmin=2)), axis=0)
                else:
                    Yn = [elt_Y]
                    Xn = np.array(line_X, ndmin=2)
                    init = True
    Yn = np.array(Yn)
    return Xn, Yn


list_features = ['nbSign', 'nbBlock', 'abstract', 'syn']
list_features += ['exergue', 'title', 'secTitle', 'supTitle']
list_features += ['subTitle', 'nbPhoto', 'aireImg', 'aireTot']
list_features += ['petittitre', 'quest_rep', 'intertitre']


# GlobalValidateModel(dico_bdd, list_mdp_data, 3, 20)
# print(TuningHyperParamGBC(dico_bdd, dict_arts, list_mdp_data, 4, 30, list_features))
t0 = time.time()
len_sample, nb_arts, min_nb = 30, 2, 30
final_obj = GetSelectionOfArticles(dico_bdd, nb_arts, len_sample)
list_vect_page = CreationListVectorsPage(final_obj, dict_arts, list_features)

# Now, I train the model. Well, first the matrices
args_cr = [dico_bdd, dict_arts, list_mdp_data, nb_arts, min_nb, list_features]
X, Y, dict_labels = without_layout.CreateXYFromScratch(*args_cr)

dict_nblabels = {x: list(Y).count(x) for x in set(Y)}
# print("The data labelled", dict_nblabels)

Xn, Yn = FilterSmallClasses(X, Y, min_nb)
dict_nblabelsn = {x: list(Yn).count(x) for x in set(Yn)}
print("The data labelled", dict_nblabelsn)

print("{:-^70}".format("Nb of classes: {}".format(len(dict_nblabelsn))))
print("{:-^70}".format("Nb of pages: {}".format(sum(dict_nblabelsn.values()))))

TrainValidateModel(Xn, Yn)
args = ["The duration for", nb_arts, min_nb, time.time() - t0]
print("{} {} articles and min_nb={}. {} sec.".format(*args))

