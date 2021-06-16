#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pickle
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
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
import os
os.chdir('/Users/baptistehessel/Documents/DAJ/MEP/montageIA/bin/')
import propositions

path_customer = '/Users/baptistehessel/Documents/DAJ/MEP/montageIA/data/CM/'

with open(path_customer + 'dico_pages', 'rb') as file:
    dico_bdd = pickle.load(file)
    
with open(path_customer + 'dico_arts', 'rb') as file:
    dict_arts = pickle.load(file)

with open(path_customer + 'list_mdp', 'rb') as file:
    list_mdp_data = pickle.load(file)


# All of this are the features I must put in my vectors
list_features = ['nbSign', 'nbBlock', 'abstract', 'syn']
list_features += ['exergue', 'title', 'secTitle', 'supTitle']
list_features += ['subTitle', 'nbPhoto', 'aireImg', 'aireTot']
# Add: petittitre, question reponse, intertitre
list_features += ['petittitre', 'quest_rep', 'intertitre']


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
    print("The nb of 0 match found: {}.".format(list_all_matches.count(0)))
    print("The nb of 1 match found: {}.".format(list_all_matches.count(1)))
    print("The nb of 2 match found: {}.".format(list_all_matches.count(2)))
    print("The nb of 3 match found: {}.".format(list_all_matches.count(3)))
    print("The nb of 4 match found: {}.".format(list_all_matches.count(4)))
    print("The nb of 5 match found: {}.".format(list_all_matches.count(5)))
    print("The total number of pages: {}.".format(len(dico_narts)))
    return "End of the verifications"


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
    print("The number of pages kept: {}".format(len(vect_XY)))
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
    parameters = {'n_estimators': range(50, 200, 15),
                  'max_depth': range(3, 6),
                  'criterion': ['mse', 'friedman_mse']}
    gbc = GradientBoostingClassifier()
    clf = GridSearchCV(gbc, parameters, verbose=3, scoring='f1_weighted')
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
    possibilities_area = []
    all_sums = []
    for ntuple in all_ntuple:
        sum_art = sum((art[0] for art in ntuple))
        all_sums.append(sum_art)
        if 0.92 * mean_area <= sum_art <= 1.08 * mean_area:
            # If constraint ok, we add the id of the article
            possibilities_area.append(tuple([art[-1] for art in ntuple]))
    print("the number of possibilities that respect the "
          "constraint of the area: {}".format(len(possibilities_area)))
    print("The mean sum {}".format(np.mean(all_sums)))
    # It remains a lot of possibilities. Here with 1O articles, there are 4645
    # possibilities
    return possibilities_area


# What's the next constraint ? -> Number of images ? (> 2)
# Now we select the ones that respect the constraint
def ConstraintImgs(all_ntuple):
    possibilities_imgs = []
    all_nb_imgs = []
    for ntuple in all_ntuple:
        nb_imgs = sum((art[1] for art in ntuple))
        all_nb_imgs.append(nb_imgs)
        if nb_imgs >= 1:
            possibilities_imgs.append(tuple([art[-1] for art in ntuple]))
    str_prt = "the nb of possib that respect the constraint of nb images: {}"
    print(str_prt.format(len(possibilities_imgs)))
    print("The mean sum {}".format(np.mean(all_nb_imgs)))
    return possibilities_imgs


def ConstraintScore(all_ntuple):
    # Importance of the articles
    # -> At least one article with score > 0.5
    possibilities_score, all_max_score = [], []
    for ntuple in all_ntuple:
        max_score = max((art[2] for art in ntuple))
        all_max_score.append(max_score)
        if max_score >= .5:
            possibilities_score.append(tuple([art[-1] for art in ntuple]))
    str_prt = "the nb of possib that respect the constraint of the score: {}"
    print(str_prt.format(len(possibilities_score)))
    print("The mean max {}".format(np.mean(all_max_score)))
    return possibilities_score


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
    print("The length of the list_vect_page: {}".format(len(list_vect_page)))
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
    return preds_gbc


#%%

# GlobalValidateModel(dico_bdd, list_mdp_data, 3, 20)

print(TuningHyperParamGBC(dico_bdd, dict_arts, list_mdp_data, 4, 30, list_features))

## 2 - 50
# clf.best_params_ {'criterion': 'mse', 'max_depth': 3, 'n_estimators': 50}
# clf.best_score_ 0.9777991425805286
## 3 - 20
# clf.best_params_ {'criterion': 'mse', 'max_depth': 3, 'n_estimators': 65}
# clf.best_score_ 0.9925850340136055
## 4 - 20
# clf.best_params_ {'criterion': 'mse', 'max_depth': 3, 'n_estimators': 125}
# clf.best_score_ 0.8826390559919535
## 5 - 20
# clf.best_params_ {'criterion': 'mse', 'max_depth': 4, 'n_estimators': 65}
# clf.best_score_ 0.9148272642390289
## 2 - 30
# clf.best_params_ {'criterion': 'friedman_mse', 'max_depth': 5, 'n_estimators': 100}
# clf.best_score_ 0.9199479776464266
## 4 - 25
# clf.best_params_ {'criterion': 'friedman_mse', 'max_depth': 3, 'n_estimators': 50}
# clf.best_score_ 0.9202237704198488


#%%

nb_arts = 3
len_sample = 25
final_obj = GetSelectionOfArticles(dico_bdd, nb_arts, len_sample)
list_vect_page = CreationListVectorsPage(final_obj, dict_arts, list_features)


#%%

# Now, I train the model. Well, first the matrices
args_cr = [dico_bdd, dict_arts, list_mdp_data]
X, Y, dict_labels = CreateXYFromScratch(*args_cr, 3, 17, list_features)
preds_gbc = PredsModel(X, Y, list_vect_page)
# Use of the dict_labels to have the real modules


#%%

rep_data = '/Users/baptistehessel/Documents/DAJ/MEP/ArticlesOptions/'
#dict_art = propositions.ExtractDicoArtInput(file_path)


def CreationListDictArt(rep_data):
    list_dict_arts = []
    for file_path in Path(rep_data).glob('./**/*'):
        if file_path.suffix == '.xml':
            dict_art = propositions.ExtractDicoArtInput(file_path)
            list_dict_arts.append(dict_art)
    return list_dict_arts


def GenerationAllNTupleFromFiles(list_dict_arts, nb_arts):
    """
    I should add the score.
    """
    list_feat = ['aireTot', 'nbPhoto', 'melodyId']
    select_arts = [[dicoa[x] for x in list_feat] for dicoa in list_dict_arts]
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
    print("The total nb of triplets generated: {}".format(len(all_ntuple)))
    return all_ntuple


list_dict_arts = CreationListDictArt(rep_data)


#%%

all_ntuples = GenerationAllNTupleFromFiles(list_dict_arts, 3)
# We should check that the ntuples obtained respect the constraints

def GetSelectionOfArticlesFromFiles(all_ntuple):
    # Use of the constraints 
    possib_area = ConstraintArea(all_ntuples)
    possib_imgs = ConstraintImgs(all_ntuples)
    # Intersection of each result obtained
    final_obj = set(possib_imgs) & set(possib_area)
    str_prt = "The number of triplets that respect the constraints: {}"
    print(str_prt.format(len(final_obj)))
    return final_obj


final_obj = GetSelectionOfArticlesFromFiles(all_ntuple)
# Fine, now I need to train the model with the right number of articles per
# page

def CreationListVectorsPageFromFiles(final_obj, list_dict_arts, list_feat):
    """
    For the case with the files input
    """
    # Need to convert list-dict_arts into a dict
    dict_arts_input = {}
    for dicta in list_dict_arts:
        dict_arts_input[dicta['melodyId']] = dicta
    list_vect_page = []
    for triplet in final_obj:
        for i, ida in enumerate(triplet):
            vect_art = np.array([dict_arts_input[ida][x] for x in list_feat],
                                ndmin=2)
            if i == 0:
                vect_page = vect_art
            else:
                vect_page = np.concatenate((vect_page, vect_art), axis=1)
        list_vect_page.append(vect_page)
    print("The length of the list_vect_page: {}".format(len(list_vect_page)))
    return list_vect_page


# Before that, I need to convert the list_dict_arts into list_vect_page
args_files = [final_obj, list_dict_arts, list_features]
list_vect_page = CreationListVectorsPageFromFiles(*args_files)


#%%

# Now, I train the model. Well, first the matrices
args_cr = [dico_bdd, dict_arts, list_mdp_data]
X, Y, dict_labels = CreateXYFromScratch(*args_cr, 3, 12, list_features)
preds_gbc = PredsModel(X, Y, list_vect_page)
# Use of the dict_labels to have the real modules
# I should do that for i in [2, 3, 4, 5] and use the predicted probas to keep
# the pages with the best scores
# First with the pages with 3 articles

pages_with_score = []
for line, vect_pg in zip(preds_gbc, list_vect_page):
    pages_with_score.append((round(max(line), 4), vect_pg))

# Generation of every triplet of pages
# Eventually, the right formula
all_triplets = [[p1, p2, p3] for i, p1 in enumerate(pages_with_score)
                for j, p2 in enumerate(pages_with_score[i + 1:])
                for p3 in pages_with_score[i + j +2:]]


#%%

# Computation of the variance of a proposition
vect_var = []
for triplet in all_triplets:
    mean_vect = np.mean([vect_pg for _, vect_pg in triplet], axis=0)
    # Computation of the variance
    # Need to determine the area
    # Size of the vectors article: 15
    ind_aireTot = list_features.index('aireTot')
    array_weights = triplet[0][1][0][range(ind_aireTot, 45, 15)]
    weights_tmp = list(array_weights / sum(array_weights))
    weights = []
    for weight in weights_tmp:
        weights += [weight] * 15
    if len(weights) != 45:
        print("Something's wrong with the weights")
        print("weights\n", weights)
        print("weights_tmp\n", weights_tmp)
        print("array_weights", array_weights)
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

# final_result = list(filter(lambda x: x[0] == max((sc for sc, _ in all_triplets_scores)), all_triplets_scores))







        
















