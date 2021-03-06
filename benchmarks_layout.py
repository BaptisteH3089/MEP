#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: baptistehessel

Script that shows the results obtained with the model that predicts the layout
for an article.

Objects necessary:
    - dico_pages
    - dico_arts
    - list_mdp

"""
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
import time
import numpy as np
import pickle
import argparse
import warnings
import without_layout
import validation_models


class MyException(Exception):
    pass

str_desc = ('Script that prints the cross-validated results of the models '
            'tested to predict the best layout for a page.')
parser = argparse.ArgumentParser(description=str_desc)
parser.add_argument('path_customer',
                    type=str,
                    help='The repertory with the data of a customer.')
parser.add_argument('nb_arts_layout',
                    type=int,
                    choices=[2, 3, 4, 5],
                    help='Select the layouts with this number of articles.')
args = parser.parse_args()

# Standardize path_customer
if args.path_customer[-1] == '/':
    path_customer = args.path_customer
else:
    path_customer = args.path_customer + '/'

# The dict with all the pages available
with open(path_customer + 'dict_pages', 'rb') as file:
    dico_bdd = pickle.load(file)
# The dict {ida: dicoa, ...}
with open(path_customer + 'dict_arts', 'rb') as file:
    dict_arts = pickle.load(file)
# The list of triplet (nb_pages_using_mdp, array_mdp, list_ids)
with open(path_customer + 'list_mdp', 'rb') as file:
    list_mdp_data = pickle.load(file)


def AireTotArts(dico_bdd, nb_arts):
    list_artTot = []
    for key, dicop in dico_bdd.items():
        if len(dicop['articles']) == nb_arts:
            list_aireTot_art = []
            for ida, art in dicop['articles'].items():
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
        if len(dicop['articles']) == nb_arts:
            for ida, art in dicop['articles'].items():
                dict_elts_art = {}
                dict_elts_art['aireArt'] = art['width'] * art['height']
                dict_elts_art['nbPhoto'] = art['nbPhoto']
                dict_elts_art['score'] = art['score']
                dict_elts_art['melodyId'] = art['melodyId']
                list_dico_arts.append(dict_elts_art)
    return list_dico_arts


# if 90000 <= art['aireTot'] <= 120000:
# Si on commnence par les pages avec 3 articles
# A priori marge de tol??rance de 5%
# On va plut??t g??n??rer tous les triplets possibles. Bourrin mais efficace.
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





def GlobalValidateModel(dico_bdd,
                        dict_arts,
                        list_mdp_data,
                        nb_arts,
                        nb_pages_min):
    dico_narts = without_layout.SelectionPagesNarts(dico_bdd, nb_arts)
    list_mdp_narts = without_layout.SelectionMDPNarts(list_mdp_data, nb_arts)
    print(without_layout.VerificationPageMDP(dico_narts, list_mdp_narts))
    args_xy = [dico_narts, dict_arts, list_mdp_narts, list_features]
    vect_XY = without_layout.CreationVectXY(*args_xy, nb_pages_min)
    dict_labels = without_layout.CreationDictLabels(vect_XY)
    X, Y = without_layout.CreationXY(vect_XY, dict_labels)
    print(validation_models.TrainValidateModel(X, Y))
    return "End of the Global Validation of the models"


def TuningHyperParamGBC(dico_bdd,
                        dict_arts,
                        list_mdp_data,
                        nb_arts,
                        nb_pages_min,
                        list_feats):
    dico_narts = without_layout.SelectionPagesNarts(dico_bdd, nb_arts)
    list_mdp_narts = without_layout.SelectionMDPNarts(list_mdp_data, nb_arts)
    print(without_layout.VerificationPageMDP(dico_narts, list_mdp_narts))
    args_xy = [dico_narts, dict_arts, list_mdp_narts, list_feats, nb_pages_min]
    vect_XY = without_layout.CreationVectXY(*args_xy)
    dict_labels = without_layout.CreationDictLabels(vect_XY)
    X, Y = without_layout.CreationXY(vect_XY, dict_labels)
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


def FilterSmallClasses(X, Y, dict_nblabels, min_nb):
    # Il faudrait enlever les classes o?? il y a moins de X repr??sentants
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
    # Deletion of the holes in the labels
    dict_corres_labels = {label: i for i, label in enumerate(set(Yn))}
    better_Yn = [dict_corres_labels[label] for label in Yn]
    return Xn, np.array(better_Yn)


list_features = ['nbSign', 'nbBlock', 'abstract', 'syn']
list_features += ['exergue', 'title', 'secTitle', 'supTitle']
list_features += ['subTitle', 'nbPhoto', 'aireImg', 'aireTot']
list_features += ['petittitre', 'quest_rep', 'intertitre']
list_features += ['isPrinc', 'isSec', 'isTer', 'isMinor']


# GlobalValidateModel(dico_bdd, list_mdp_data, 3, 20)
# print(TuningHyperParamGBC(dico_bdd, dict_arts, list_mdp_data, 4, 30, list_features))
t0 = time.time()
len_sample = 30
nb_arts = args.nb_arts_layout
final_obj = GetSelectionOfArticles(dico_bdd, nb_arts, len_sample)
list_vect_page = CreationListVectorsPage(final_obj, dict_arts, list_features)

# Now, I train the model. Well, first the matrices
for min_nb in [10, 30, 50]:
    print("\n{:*^80}\n".format(f"min number of pages per layout={min_nb}"))
    args_cr = [dico_bdd, dict_arts, list_mdp_data, nb_arts, min_nb]
    args_cr += [list_features]
    X, Y, dict_labels = without_layout.CreateXYFromScratch(*args_cr)
    if len(X) == 0:
        print("{:-^80}".format("Not enough data"))
        continue
    # At least 10 pages per layout
    dict_nblabel = {x: list(Y).count(x) for x in set(Y)}
    Xn, Yn = FilterSmallClasses(X, Y, dict_nblabel, 10)
    dict_nblabelsn = {x: list(Yn).count(x) for x in set(Yn)}
    # print("The data labelled", dict_nblabelsn)
    print("{:-^70}".format("Nb of classes: {}".format(len(dict_nblabelsn))))
    tot_pages = sum(dict_nblabelsn.values())
    print("{:-^70}".format("Nb of pages: {}".format(tot_pages)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        validation_models.TrainValidateModel(Xn, Yn)
    args = ["The duration for", nb_arts, min_nb, time.time() - t0]
    print("{} {} articles and min_nb={}. {} sec.".format(*args))

