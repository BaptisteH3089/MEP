#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pickle
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import f1_score
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from pathlib import Path
from operator import itemgetter
import time
import os
os.chdir('/Users/baptistehessel/Documents/DAJ/MEP/montageIA/bin/')
import propositions

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


def SelectionPagesNarts(dico_bdd, nb_arts):
    dico_narts = {}
    for key in dico_bdd.keys():
        if len(dico_bdd[key]['dico_page']['articles']) == nb_arts:
            dico_narts[key] = dico_bdd[key]['dico_page']['articles']
    return dico_narts


def SelectionMDPNarts(list_mdp_data, nb_arts):
    list_mdp_narts = []
    for nb, mdp, list_ids in list_mdp_data:
        if mdp.shape[0] == nb_arts:
            list_mdp_narts.append([mdp, list_ids])
    return list_mdp_narts


def VerificationPageMDP(dico_narts, list_mdp_narts):
    list_all_matches = []
    for key in dico_narts.keys():
        list_match = [1 for _, list_ids in list_mdp_narts if key in list_ids]
        list_all_matches.append(sum(list_match))
    str_prt = "The nb of 0 match found"
    print("{:<35} {:>35}".format(str_prt, list_all_matches.count(0)))
    print("{:<35} {:>35}".format("The total nb of pages", len(dico_narts)))
    return "{:^70}".format("End of the verifications")


# OK, there are some duplicates
# I need to create the vector page
# But what to put inside them ?
def CreationVectXY(dico_narts,
                   dict_arts,
                   list_mdp_narts,
                   list_features,
                   nb_pages_min):
    vect_XY = []
    for key in dico_narts.keys():
        for i, (ida, dicoa) in enumerate(dico_narts[key].items()):
            vect_art = [dict_arts[ida][x] for x in list_features]
            """
            if 'petittitre' in dicoa['typeBlock']:
                vect_art.append(1)
            else:
                vect_art.append(0)
            if 'question' in dicoa['typeBlock']:
                vect_art.append(1)
            else:
                vect_art.append(0)
            if 'intertitre' in dicoa['typeBlock']:
                vect_art.append(1)
            else:
                vect_art.append(0)
            """
            vect_art = np.array(vect_art, ndmin=2)
            if i == 0:
                vect_page = vect_art
            else:
                vect_page = np.concatenate((vect_page, vect_art), axis=1)
        for mdp, list_ids in list_mdp_narts:
            if len(list_ids) >= nb_pages_min:
                if key in list_ids:
                    vect_XY.append((vect_page, mdp))
                    break
    print("{:<35} {:>35}".format("The number of pages kept", len(vect_XY)))
    return vect_XY


# Now, we create a dico with the mdp
def CreationDictLabels(vect_XY):
    """
    Return {0: np.array([...]),
            1: np.array([...]),
            ...}
    The values correspond to the layouts
    """
    label = 0
    dict_labels = {}
    for _, mdp in vect_XY:
        find = False
        for mdp_val in dict_labels.values():
            if np.allclose(mdp, mdp_val):
                find = True
        if find is False:
            dict_labels[label] = mdp
            label += 1
    print("There are {} labels in that dict.".format(label + 1))
    return dict_labels


def CreationXY(vect_XY, dict_labels):
    """
    Creation of the matrix X and Y used to train a model.
    """
    for i, (vect_X, vect_Y) in enumerate(vect_XY):
        if i == 0:
            big_X = vect_X
            for label, vect_label in dict_labels.items():
                if np.allclose(vect_label, vect_Y):
                    big_Y = [label]
        else:
            big_X = np.concatenate((big_X, vect_X))
            for label, vect_label in dict_labels.items():
                if np.allclose(vect_label, vect_Y):
                    try:
                        big_Y.append(label)
                    except:
                        print("ERROR")
                        print(big_Y, label)
    return big_X, np.array(big_Y)


def TrainValidateModel(X, Y):
    skf = StratifiedKFold(n_splits=3)
    ss = ShuffleSplit(n_splits=3)
    for i, (train, test) in enumerate(ss.split(X, Y)):
        # Random Forest
        clf = RandomForestClassifier().fit(X[train], Y[train])
        preds_clf = clf.predict(X[test])
        score_clf = f1_score(Y[test], preds_clf, average='weighted')
        # SVC
        svc = svm.SVC().fit(X[train], Y[train])
        preds_svc = svc.predict(X[test])
        score_svc = f1_score(Y[test], preds_svc, average='weighted')
        # Linear SVC
        linear_svc = make_pipeline(StandardScaler(), LinearSVC())
        linear_svc.fit(X[train], Y[train])
        preds_lsvc = linear_svc.predict(X[test])
        score_lsvc = f1_score(Y[test], preds_lsvc, average='weighted')
        # SGD
        sgd = make_pipeline(StandardScaler(),
                            SGDClassifier(max_iter=1000, tol=1e-3))
        sgd.fit(X[train], Y[train])
        preds_sgd = sgd.predict(X[test])
        score_sgd = f1_score(Y[test], preds_sgd, average='weighted')
        # Gaussian Naive Bayes
        gnb = GaussianNB().fit(X[train], Y[train])
        preds_gnb = gnb.predict(X[test])
        score_gnb = f1_score(Y[test], preds_gnb, average='weighted')
        # Logistic regression default
        Xtrain_scaled = StandardScaler().fit_transform(X[train])
        Xtest_scaled = StandardScaler().fit_transform(X[test])
        logreg = LogisticRegression(max_iter=1000).fit(Xtrain_scaled, Y[train])
        preds_logreg = logreg.predict(Xtest_scaled)
        score_logreg = f1_score(Y[test], preds_logreg, average='weighted')
        # Logistic regression elasticnet
        logreg1 = LogisticRegression(penalty='elasticnet',
                                     solver='saga',
                                     l1_ratio=.95,
                                     max_iter=1000)
        logreg1.fit(Xtrain_scaled, Y[train])
        preds_logreg1 = logreg1.predict(Xtest_scaled)
        score_logreg1 = f1_score(Y[test], preds_logreg1, average='weighted')
        # Logistic regression elasticnet 0.85
        logreg2 = LogisticRegression(penalty='elasticnet',
                                     solver='saga',
                                     l1_ratio=.85,
                                     max_iter=1000)
        logreg2.fit(Xtrain_scaled, Y[train])
        preds_logreg2 = logreg2.predict(Xtest_scaled)
        score_logreg2 = f1_score(Y[test], preds_logreg2, average='weighted')
        # Gradient Boosting Classifier
        gbc = GradientBoostingClassifier().fit(X[train], Y[train])
        preds_gbc = gbc.predict(X[test])
        score_gbc = f1_score(Y[test], preds_gbc, average='weighted')
        print("FOLD: {}.".format(i))
        print("{:<30} {:>35.4f}.".format("score rdmForest", score_clf))
        print("{:<30} {:>35.4f}.".format("score GBC", score_gbc))
        print("{:<30} {:>35.4f}.".format("score SVC", score_svc))
        print("{:<30} {:>35.4f}.".format("score Linear SVC", score_lsvc))
        print("{:<30} {:>35.4f}.".format("score SGD", score_sgd))
        print("{:<30} {:>35.4f}.".format("score GNB", score_gnb))
        str_sc = "score LogReg"
        print("{:<30} {:>35.4f}.".format(str_sc, score_logreg))
        str_sc1 = str_sc + " elasticnet"
        print("{:<30} {:>35.4f}.".format(str_sc1, score_logreg1))
        str_sc2 = str_sc + " elasticnet 0.85"
        print("{:<30} {:>35.4f}.".format(str_sc2, score_logreg2))
        print('\n')
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
    #clf = GridSearchCV(gbc, parameters, verbose=3, scoring='f1_weighted')
    print(clf.fit(X, Y))
    print('clf.best_params_', clf.best_params_)
    print('clf.best_score_', clf.best_score_)
    return True


# Dans le cas où l'on a des xml input, il faut faire une sélection de ces xml
# pour créer une page
# Il faut jouer sur l'aireTot de chaque article
# But how to select articles such that sum(aireTot_i for i in I) ~ airePage
# This selection seems difficult
# Perhaps, possible to do it iteratively

# On va parcourir le dico_bdd pour trouver une aire moyenne de la somme des 
# aires des articles
# D'abord pour des pages avec 3 articles
# -> mais il y aura le problème des pubs
# bon... pour l'instant tant pis
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
                      for j, y in enumerate(select_arts[i + 1:])]
    elif nb_arts == 3:
        all_ntuple = [(x, y, z) for i, x in enumerate(select_arts)
                      for j, y in enumerate(select_arts[i + 1:])
                      for z in select_arts[j+2:]]
    elif nb_arts == 4:
        all_ntuple = [(a, b, c, d) for i, a in enumerate(select_arts)
                      for j, b in enumerate(select_arts[i + 1:])
                      for k, c in select_arts[j + 2:]
                      for d in select_arts[k + 3:]]
    elif nb_arts == 5:
        all_ntuple = [(a, b, c, d, e) for i, a in enumerate(select_arts)
                      for j, b in enumerate(select_arts[i + 1:])
                      for k, c in enumerate(select_arts[j + 2:])
                      for l, d in enumerate(select_arts[k + 3:])
                      for e in enumerate(select_arts[l + 4:])]
    else:
        str_exc = 'Wrong nb of arts: {}. Must be in [2, 5]'
        raise MyException(str_exc.format(nb_arts))
    print("The total nb of ntuples generated: {}".format(len(all_ntuple)))
    return all_ntuple


def ConstraintArea(all_ntuple):
    # Now we select the ones that respect the constraint
    mean_area = 95000
    possib_area = []
    all_sums = []
    for ntuple in all_ntuple:
        sum_art = sum((art[0] for art in ntuple))
        all_sums.append(sum_art)
        if 0.92 * mean_area <= sum_art <= 1.08 * mean_area:
            # If constraint ok, we add the id of the article
            possib_area.append(tuple([art[-1] for art in ntuple]))
    print("{:^35} {:>15}".format("Constraint area", len(possib_area)))
    print("{:^35} {:>15}".format("Mean score max", np.mean(all_sums)))
    return possib_area


# What's the next constraint ? -> Number of images ? (> 2)
# Now we select the ones that respect the constraint
def ConstraintImgs(all_ntuple):
    possib_imgs, all_nb_imgs = [], []
    for ntuple in all_ntuple:
        try:
            nb_imgs = sum((art[1] for art in ntuple))
        except Exception as e:
            print("Error with the sum of the nbImg")
            print(e)
            print("ntuple", ntuple)
            print("nb_imgs", nb_imgs)
        all_nb_imgs.append(nb_imgs)
        if nb_imgs >= 1:
            possib_imgs.append(tuple([art[-1] for art in ntuple]))
    print("{:^35} {:>15}".format("Constraint images", len(possib_imgs)))
    print("{:^35} {:>15}".format("Mean score max", np.mean(all_nb_imgs)))
    return possib_imgs


def ConstraintScore(all_ntuple):
    # Importance of the articles
    # -> At least one article with score > 0.5
    possib_score, all_max_score = [], []
    for ntuple in all_ntuple:
        max_score = max((art[2] for art in ntuple))
        all_max_score.append(max_score)
        if max_score >= .5:
            possib_score.append(tuple([art[-1] for art in ntuple]))
    print("{:^35} {:>15}".format("Constraint score", len(possib_score)))
    print("{:^35} {:>15}".format("Mean score max", np.mean(all_max_score)))
    return possib_score


def GetSelectionOfArticles(dico_bdd, nb_arts, len_sample):
    list_aireTot = AireTotArts(dico_bdd, nb_arts)
    list_dico_arts = SelectionArts(dico_bdd, nb_arts)
    print("The number of articles selected: {}".format(len(list_dico_arts)))
    all_ntuples = GenerationAllNTuple(list_dico_arts, len_sample, nb_arts)
    # Use of the constraints 
    possib_area = ConstraintArea(all_ntuples)
    possib_imgs = ConstraintImgs(all_ntuples)
    possib_score = ConstraintScore(all_ntuples)
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


def CreateXYFromScratch(dico_bdd,
                        dict_arts,
                        list_mdp,
                        nb_arts,
                        nbpg_min,
                        list_feats):
    dico_narts = SelectionPagesNarts(dico_bdd, nb_arts)
    list_mdp_narts = SelectionMDPNarts(list_mdp, nb_arts)
    print(VerificationPageMDP(dico_narts, list_mdp_narts))
    args_xy = [dico_narts, dict_arts, list_mdp_narts, list_feats, nbpg_min]
    vect_XY = CreationVectXY(*args_xy)
    dict_labels = CreationDictLabels(vect_XY)
    X, Y = CreationXY(vect_XY, dict_labels)
    return X, Y, dict_labels


def PredsModel(X, Y, list_vect_page):
    gbc = GradientBoostingClassifier().fit(X, Y)
    for i, vect_page in enumerate(list_vect_page):
        if i == 0:
            matrix_input = vect_page
        else:
            matrix_input = np.concatenate((matrix_input, vect_page))
    print("The shape of the matrix input: {}".format(matrix_input.shape))
    preds_gbc = gbc.predict_proba(matrix_input)
    preds = gbc.predict(matrix_input)
    print("The variety of the predictions")
    print([(x, list(preds).count(x)) for x in set(list(preds))])
    return preds_gbc


def CreationListDictArt(rep_data):
    """
    Parameters
    ----------
    rep_data : str
        The folder with the articles input

    Returns
    -------
    dict_arts_input : dict
        A dictionary of the form {ida: dicoa, ...}
    """
    dict_arts_input = {}
    for file_path in Path(rep_data).glob('./**/*'):
        if file_path.suffix == '.xml':
            dict_art = propositions.ExtractDicoArtInput(file_path)
            dict_arts_input[dict_art['melodyId']] = dict_art
    return dict_arts_input


def GenerationAllNTupleFromFiles(dict_arts_input, nb_arts):
    """
    I should add the score.
    """
    list_feat = ['aireTot', 'nbPhoto', 'melodyId']
    val_dict = dict_arts_input.values()
    select_arts = [[dicoa[x] for x in list_feat] for dicoa in val_dict]
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
        all_ntuple = [(a, b, c, d) for i, a in enumerate(select_arts)
                      for j, b in enumerate(select_arts[i + 1:])
                      for k, c in enumerate(select_arts[i + j + 2:])
                      for d in select_arts[i + j + k + 3:]]
    elif nb_arts == 5:
        all_ntuple = [(a, b, c, d, e) for i, a in enumerate(select_arts)
                      for j, b in enumerate(select_arts[i + 1:])
                      for k, c in enumerate(select_arts[i + j + 2:])
                      for l, d in enumerate(select_arts[i + j + k + 3:])
                      for e in select_arts[i + j + k + l + 4:]]
    else:
        str_exc = 'Wrong nb of arts: {}. Must be in [2, 5]'
        raise MyException(str_exc.format(nb_arts))
    print("The total nb of {}-tuple generated: {}.".format(nb_arts, len(all_ntuple)))
    return all_ntuple


def GetSelectionOfArticlesFromFiles(all_ntuples):
    # Use of the constraints 
    possib_area = ConstraintArea(all_ntuples)
    possib_imgs = ConstraintImgs(all_ntuples)
    # Intersection of each result obtained
    final_obj = set(possib_imgs) & set(possib_area)
    str_prt = "The number of ntuples that respect the constraints"
    print("{:<35} {:>35}".format(str_prt, len(final_obj)))
    return final_obj


def CreationListVectorsPageFromFiles(final_obj, artd_input, list_feat):
    """
    For the case with the files input.
    """
    list_vect_page = []
    for triplet in final_obj:
        for i, ida in enumerate(triplet):
            artv = np.array([artd_input[ida][x] for x in list_feat], ndmin=2)
            if i == 0: vect_page = artv
            else: vect_page = np.concatenate((vect_page, artv), axis=1)
        list_vect_page.append(vect_page)
    return list_vect_page


def SelectBestTriplet(all_triplets):
    """
    Computation of the variance of a proposition and then the final score of a
    triplet.
    """
    vect_var = []
    for triplet in all_triplets:
        mean_vect = np.mean([vect_pg for _, vect_pg in triplet], axis=0)
        # Computation of the variance
        # Need to determine the area
        # Size of the vectors article: 15
        ind_aireTot = list_features.index('aireTot')
        size_vect_pg = mean_vect.shape[1]
        array_weights = triplet[0][1][0][range(ind_aireTot, size_vect_pg, 15)]
        weights_tmp = list(array_weights / sum(array_weights))
        weights = []
        for weight in weights_tmp:
            weights += [weight] * 15
        if len(weights) != size_vect_pg:
            print("Something's wrong with the weights")
            print("weights\n", weights)
            print("weights_tmp\n", weights_tmp)
            print("array_weights", array_weights)
            print("triplet: {}".format(triplet))
            break
        variance_triplet = 0
        for _, vect_page in triplet:
            diff_vect = np.array(weights) * (vect_page - mean_vect)
            variance_triplet += int(np.sqrt(np.dot(diff_vect, diff_vect.T)))
        vect_var.append(variance_triplet)
    # We standardize the vect_var
    std_var = np.std(vect_var)
    mean_var = np.mean(vect_var)
    stand_vect_var = (vect_var - mean_var) / std_var
    all_triplets_scores = []
    for triplet, var in zip(all_triplets, stand_vect_var):
        score_prop = sum((sc for sc, _ in triplet))
        all_triplets_scores.append((score_prop + var, triplet))
    # Ultimately, we take the triplet with the best score
    all_triplets_scores.sort(key=itemgetter(0), reverse=True)
    best_triplet = all_triplets_scores[0]
    print("The best triplet has the score: {:>40.2f}.".format(best_triplet[0]))
    print("{:-^80}".format("The score of each page"))
    for sc, _ in best_triplet[1]:
        print("{:40.4f}".format(sc))
    print("{:-^80}".format("The vectors page"))
    for _, page in best_triplet[1]:
        print("{:^40}".format(str(page)))
    return best_triplet


def GenerationOfAllTriplets(preds_gbc, list_vect_page):
    pages_with_score = []
    for line, vect_pg in zip(preds_gbc, list_vect_page):
        pages_with_score.append((round(max(line), 4), vect_pg))
    # Generation of every triplet of pages
    # Eventually, the right formula
    all_triplets = [[p1, p2, p3] for i, p1 in enumerate(pages_with_score)
                    for j, p2 in enumerate(pages_with_score[i + 1:])
                    for p3 in pages_with_score[i + j +2:]]
    return all_triplets


def FillShortVectorPage(final_obj):
    list_short_vect_page = []
    for ntuple in final_obj:
        # Line vector
        short_vect_page = np.zeros((1, 15))
        # nbSign
        nbSign_page = sum((dict_arts_input[ida]['nbSign'] for ida in ntuple))
        short_vect_page[0][0] = nbSign_page
        # nbBlock
        nbBlock_page = sum((dict_arts_input[ida]['nbBlock'] for ida in ntuple))
        short_vect_page[0][1] = nbBlock_page
        # abstract
        abstract_page = sum((dict_arts_input[ida]['abstract'] for ida in ntuple))
        short_vect_page[0][2] = abstract_page
        # syn
        syn_page = sum((dict_arts_input[ida]['syn'] for ida in ntuple))
        short_vect_page[0][3] = syn_page
        # exergue
        exergue_page = sum((dict_arts_input[ida]['exergue'] for ida in ntuple))
        short_vect_page[0][4] = exergue_page
        # title - NOT INTERESTING
        title_page = sum((dict_arts_input[ida]['title'] for ida in ntuple))
        short_vect_page[0][5] = title_page
        # secTitle
        secTitle_page = sum((dict_arts_input[ida]['secTitle'] for ida in ntuple))
        short_vect_page[0][6] = secTitle_page
        # supTitle
        supTitle_page = sum((dict_arts_input[ida]['supTitle'] for ida in ntuple))
        short_vect_page[0][7] = supTitle_page
        # subTitle
        subTitle_page = sum((dict_arts_input[ida]['subTitle'] for ida in ntuple))
        short_vect_page[0][8] = subTitle_page
        # nbPhoto
        nbPhoto_page = sum((dict_arts_input[ida]['nbPhoto'] for ida in ntuple))
        short_vect_page[0][9] = nbPhoto_page
        # aireImg
        aireImg_page = sum((dict_arts_input[ida]['aireImg'] for ida in ntuple))
        short_vect_page[0][10] = aireImg_page
        # aireTot
        aireTot_page = sum((dict_arts_input[ida]['aireTot'] for ida in ntuple))
        short_vect_page[0][11] = aireTot_page
        # petittitre
        petittitre_page = sum((dict_arts_input[ida]['petittitre'] for ida in ntuple))
        short_vect_page[0][12] = petittitre_page
        # quest_rep
        quest_rep_page = sum((dict_arts_input[ida]['quest_rep'] for ida in ntuple))
        short_vect_page[0][13] = quest_rep_page
        # intertitre
        intertitre_page = sum((dict_arts_input[ida]['intertitre'] for ida in ntuple))
        short_vect_page[0][14] = intertitre_page
        # We add that vector to the list
        list_short_vect_page.append(short_vect_page)
    return list_short_vect_page


def SelectBestTripletAllNumbers(all_triplets, global_mean_vect, global_vect_std):
    """
    Computation of the variance of a proposition and then the final score of a
    triplet.
    """
    vect_var = []
    for triplet in all_triplets:
        list_vectors = [vect_pg for _, _, vect_pg in triplet]
        mean_local_vect = np.mean(list_vectors, axis=0)
        norm_mean_local_vect = (mean_local_vect - global_mean_vect) / global_vect_std
        variance_triplet = 0
        for _, _, vect_page in triplet:
            norm_vect = (vect_page - global_mean_vect) / global_vect_std
            try:
                diff_eucl = np.sqrt((norm_vect - norm_mean_local_vect) ** 2)
                diff_abs = np.abs(norm_vect - norm_mean_local_vect)
                variance_triplet += np.sum(diff_eucl)
            except Exception as e:
                args_n = ["norm_mean_local_vect", norm_mean_local_vect]
                args_t = ["type(norm_mean_loc_v)", type(norm_mean_local_vect)]
                print("Error with the computation of the variance \n", e)
                print("{:^35} {:>40}".format("norm_vect", str(norm_vect)))
                print("{:^35} {:>40}".format("type(norm_v)", type(norm_vect)))
                print("{:^35} {:>40}".format(*args_n))
                print("{:^35} {:>40}".format(*args_t))
                raise Exception
        vect_var.append(variance_triplet)
    # Here, I need to standardize the variance
    mean_var, std_var = np.mean(vect_var, axis=0), np.std(vect_var, axis=0)
    print("{:^40} {:>20.2f}".format("The mean var", mean_var))
    print("{:^40} {:>20.2f}".format("The std of the var", std_var))
    if np.isclose(std_var, 0): std_var = 1
    norm_vect_var = (vect_var - mean_var) / std_var
    all_triplets_scores = []
    for triplet, var in zip(all_triplets, norm_vect_var):
        score_prop = sum((sc for sc, _, _ in triplet))
        # We ponderate the variance by 0.5
        all_triplets_scores.append((score_prop + 0.5 * var, triplet))
    # Ultimately, we take the triplet with the best score
    all_triplets_scores.sort(key=itemgetter(0), reverse=True)
    best_triplet = all_triplets_scores[0]
    print("The best triplet has the score: {:>40.2f}.".format(best_triplet[0]))
    print("{:-^80}".format("The score of each page"))
    for sc, _, _ in best_triplet[1]: print("{:40.4f}".format(sc))
    print("{:-^80}".format("The vectors page"))
    for _, _, page in best_triplet[1]: print("{:^40}".format(str(page)))
    return all_triplets_scores


def GetMeanStdFromList(big_list_sc_label_vectsh):
    """
    Returns the mean and std vect of a list of vectors with dim = 2.
    Here, it is numpy vectors of dim (1, 15).
    """
    # Creation of a big matrix in order to compute the global mean and std
    for i, tuple_page in enumerate(big_list_sc_label_vectsh):
        if i == 0: full_matrix = tuple_page[-1]
        else: full_matrix = np.concatenate((full_matrix, tuple_page[-1]))
    global_vect_std = np.std(full_matrix, axis=0)
    for i in range(len(global_vect_std)):
        if np.isclose(global_vect_std[i], 0): global_vect_std[i] = 1
    global_mean_vect = np.mean(full_matrix, axis=0)
    return global_mean_vect, global_vect_std


#%%

start_time = time.time()
rep_data = '/Users/baptistehessel/Documents/DAJ/MEP/ArticlesOptions/'
dict_arts_input = CreationListDictArt(rep_data)
print("Duration loading input: {:.2f}".format(time.time() - start_time))

# Ok, now I should convert each page of the objects final_object into a vector
# page of size 15 with the following features
list_features = ['nbSign', 'nbBlock', 'abstract', 'syn']
list_features += ['exergue', 'title', 'secTitle', 'supTitle']
list_features += ['subTitle', 'nbPhoto', 'aireImg', 'aireTot']
list_features += ['petittitre', 'quest_rep', 'intertitre']

big_list_sc_label_vectsh = []
list_vectsh_ids = []
for nb_arts in range(2, 5):
    t1 = time.time()
    print("{:-^80}".format("PAGES WITH {} ARTS".format(nb_arts)))
    # The list all_ntuples corresponds to all nb_arts-tuple possible
    all_ntuples = GenerationAllNTupleFromFiles(dict_arts_input, nb_arts)
    # The set set_ids_page is made of list of ids that respect the basic
    # constraints of a page
    set_ids_page = GetSelectionOfArticlesFromFiles(all_ntuples)
    # All the short vectors page. For now it is vectors of size 15 which are
    # the sum of all vectors articles in the page
    list_short_vect_page = FillShortVectorPage(set_ids_page)
    # We add the couple (sh_vect, ids) to all_list_couples_sh_vect_ids
    # This will be used to find back the ids of the articles which form the
    # page.
    for short_vect, ids_page in zip(list_short_vect_page, set_ids_page):
        list_vectsh_ids.append((short_vect, ids_page))
    # But I also need the long vectors page for the predictions
    args_files = [set_ids_page, dict_arts_input, list_features]
    list_long_vect_page = CreationListVectorsPageFromFiles(*args_files)
    # Now, I train the model. Well, first the matrices
    args_cr = [dico_bdd, dict_arts, list_mdp_data]
    X, Y, dict_labels = CreateXYFromScratch(*args_cr, nb_arts, 20, list_features)
    preds_gbc = PredsModel(X, Y, list_long_vect_page)
    # I add the score given by preds_gbc to the short vect_page and the index
    # of the max because it is the label predicted
    list_score_label_short_vect = []
    for short_vect_pg, line_pred in zip(list_short_vect_page, preds_gbc):
        label_pred = list(preds_gbc[0]).index(max(preds_gbc[0]))
        sc_page = max(preds_gbc[0])
        list_score_label_short_vect.append((sc_page, label_pred, short_vect_pg))
    big_list_sc_label_vectsh += list_score_label_short_vect
    print("Duration model {} arts: {:.2f}".format(nb_arts, time.time() - t1))
    print("{:-^80}".format(""))

t2 = time.time()
gbal_mean_vect, gbal_vect_std = GetMeanStdFromList(big_list_sc_label_vectsh)
all_triplets = [[p1, p2, p3] for i, p1 in enumerate(big_list_sc_label_vectsh)
                for j, p2 in enumerate(big_list_sc_label_vectsh[i + 1:])
                for p3 in big_list_sc_label_vectsh[i + j + 2:]]

args_all_nb = [all_triplets, gbal_mean_vect, gbal_vect_std]
all_triplets_scores = SelectBestTripletAllNumbers(*args_all_nb)
list_ids_found = []
for k in range(10):
    print("{:-^80}".format("PAGE {}".format(k + 1)))
    for i, (_, _, vect_page) in enumerate(all_triplets_scores[k][1]):
        try:
            f = lambda x: np.allclose(x[0], vect_page)
            ids_found = tuple(filter(f, list_vectsh_ids))
        except Exception as e:
            print("An exception occurs", e)
            print("The vect_page: {}".format(vect_page))
        list_ids_found.append(ids_found)
        print("The ids of the page {} result: {}".format(i, ids_found[0][1]))
print("Duration computation scores triplet: {:.2f}".format(time.time() - t2))
print("{:-^80}".format("TOTAL DURATION - {:.2f}".format(time.time() - start_time)))


#%%

# GlobalValidateModel(dico_bdd, list_mdp_data, 3, 20)
# print(TuningHyperParamGBC(dico_bdd, dict_arts, list_mdp_data, 4, 30, list_features))
len_sample, nb_arts = 25, 3
final_obj = GetSelectionOfArticles(dico_bdd, nb_arts, len_sample)
list_vect_page = CreationListVectorsPage(final_obj, dict_arts, list_features)
# Now, I train the model. Well, first the matrices
args_cr = [dico_bdd, dict_arts, list_mdp_data]
X, Y, dict_labels = CreateXYFromScratch(*args_cr, 3, 17, list_features)
preds_gbc = PredsModel(X, Y, list_vect_page)
# Use of the dict_labels to have the real modules
rep_data = '/Users/baptistehessel/Documents/DAJ/MEP/ArticlesOptions/'
list_dict_arts = CreationListDictArt(rep_data)
np_arts_pg = 4
# all_ntuples = GenerationAllNTupleFromFiles(list_dict_arts, 3)
all_couples = GenerationAllNTupleFromFiles(list_dict_arts, np_arts_pg)
final_obj = GetSelectionOfArticlesFromFiles(all_couples)
# We should check that the ntuples obtained respect the constraints
# final_obj = GetSelectionOfArticlesFromFiles(all_ntuple)
# Fine, now I need to train the model with the right nb of arts per page
# Before that, I need to convert the list_dict_arts into list_vect_page
args_files = [final_obj, list_dict_arts, list_features]
list_vect_page = CreationListVectorsPageFromFiles(*args_files)
# Now, I train the model. Well, first the matrices
args_cr = [dico_bdd, dict_arts, list_mdp_data]
X, Y, dict_labels = CreateXYFromScratch(*args_cr, np_arts_pg, 12, list_features)
preds_gbc = PredsModel(X, Y, list_vect_page)
# Use of the dict_labels to have the real modules
# I should do that for i in [2, 3, 4, 5] and use the predicted probas to keep
# the pages with the best scores
# First with the pages with 3 articles
all_triplets = GenerationOfAllTriplets(preds_gbc, list_vect_page)
best_triplet = SelectBestTriplet(all_triplets)

