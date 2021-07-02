#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
import pickle
import without_layout
import numpy as np
import argparse

"""
Eventually I can check if it's better when I increase the number of features.
"""

# ARG PARSER - SERVOR. Initilisation of the arguements of the script 
# montage_ia.xml for the start webservice montage IA.
parser = argparse.ArgumentParser(description='Models used to predict layouts.')
parser.add_argument('--selected_models',
                    help="The models to consider.",
                    type=int,
                    nargs='+',
                    default=3)
parser.add_argument('--save_model',
                    help="Whether to save or not the models.",
                    type=bool,
                    default=False)
parser.add_argument('--cross_val',
                    help="Whether to show cross-validation scores.",
                    type=bool,
                    default=False)
parser.add_argument('--grid_search',
                    help="Whether to search for best parameters.",
                    type=bool,
                    default=False)
args = parser.parse_args()

print("args.selected_models", args.selected_models)

if type(args.selected_models) is int:
    args.selected_models = [args.selected_models]

dict_model = {2: False, 3: False, 4: False, 5: False}
if 0 in args.selected_models:
    dict_model = {2: True, 3: True, 4: True, 5: True}
else:
    for num_model in args.selected_models:
        if num_model != 'all':
            dict_model[num_model] = True

list_features = ['nbSign', 'nbBlock', 'abstract', 'syn']
list_features += ['exergue', 'title', 'secTitle', 'supTitle']
list_features += ['subTitle', 'nbPhoto', 'aireImg', 'aireTot']
list_features += ['petittitre', 'quest_rep', 'intertitre']

path_cm = '/Users/baptistehessel/Documents/DAJ/MEP/montageIA/data/CM/'

# The dict with all the pages available
with open(path_cm + 'dico_pages', 'rb') as file:
    dico_bdd = pickle.load(file)
# The dict {ida: dicoa, ...}
with open(path_cm + 'dico_arts', 'rb') as file:
    dict_arts = pickle.load(file)
# The list of triplet (nb_pages_using_mdp, array_mdp, list_ids)
with open(path_cm + 'list_mdp', 'rb') as file:
    list_mdp_data = pickle.load(file)


def RemovingLabels(X, Y, nbmin):
    init = False
    dict_labels = {x: list(Y).count(x) for x in set(list(Y))}
    for line_X, label_Y in zip(X, Y):
        line_x = np.array(line_X, ndmin=2)
        if dict_labels[label_Y] < nbmin:
            pass
        else:
            if init is True:
                smaller_X = np.concatenate((smaller_X, line_x))
                smaller_Y.append(label_Y)
            else:
                smaller_X = line_x
                smaller_Y = [label_Y]
                init = True
    return smaller_X, np.array(smaller_Y)

            
##############################################################################
#                                                                            #
#                 MODEL - LAYOUTS WITH 2 ARTICLES                            #
#                                                                            #
##############################################################################

if dict_model[2] is True:
    nb_pgmin = 20
    nb_arts = 2
    args_cr = [dico_bdd, dict_arts, list_mdp_data, nb_arts, nb_pgmin]
    X, Y, dict_labels = without_layout.CreateXYFromScratch(*args_cr, list_features)
    # The model with the layouts with 2 articles is trained with 1136 and 
    # there are 24 different classes or layouts
    list_dec = [(x, list(Y).count(x)) for x in set(list(Y))]
    X, Y = RemovingLabels(X, Y, 10)
    list_dec = [(x, list(Y).count(x)) for x in set(list(Y))]
    print("{:-^80}".format("Cleaner Objects"))
    for k, (label, nbelts) in enumerate(list_dec):
        print(f"{k:<6} {nbelts:>6}")
    print(f"Total number of pages: {sum((nb for _, nb in list_dec)):->15}")

    ##### CROSS-VALIDATION SCORE #####
    if args.cross_val:
        mean_gbc2 = []
        ss = ShuffleSplit(n_splits=5)
        for k, (train, test) in enumerate(ss.split(X, Y)):
            print(f"FOLD [{k}]")
            gbc_cv = GradientBoostingClassifier().fit(X[train], Y[train])
            preds_gbc = gbc_cv.predict(X[test])
            score_gbc = f1_score(Y[test], preds_gbc, average='macro')
            mean_gbc2.append(score_gbc)
        print(f"The cross-val score: {np.mean(mean_gbc2)}")
        # 84%

    ##### TUNING PARAMETERS #####
    if args.grid_search is True:
        param_gbc = {'n_estimators': range(50, 200, 25),
                     'max_depth': range(2, 5)}
        gbc = GradientBoostingClassifier()
        clf = GridSearchCV(gbc, param_gbc, scoring='f1_macro', verbose=3)
        clf.fit(X, Y)
        print(f"best params: {clf.best_params_}")
        print(f"best score: {clf.best_score_}")
        
    ##### SAVE MODEL #####
    if args.save_model:
        if args.grid_search:
            gbc2 = GradientBoostingClassifier(**clf.best_params_).fit(X, Y)
            with open(path_cm + 'gbc2', 'wb') as f: pickle.dump(gbc2, f)
        else:
            gbc2 = GradientBoostingClassifier().fit(X, Y)
            with open(path_cm + 'gbc2', 'wb') as f: pickle.dump(gbc2, f)


##############################################################################
#                                                                            #
#                 MODEL - LAYOUTS WITH 3 ARTICLES                            #
#                                                                            #
##############################################################################


if dict_model[3]:
    nb_pgmin = 20
    nb_arts = 3
    args_cr = [dico_bdd, dict_arts, list_mdp_data, nb_arts, nb_pgmin]
    X, Y, dict_labels = without_layout.CreateXYFromScratch(*args_cr, list_features)
    # Only 141 pages in our data for 7 labels
    list_dec = [(x, list(Y).count(x)) for x in set(list(Y))]
    X, Y = RemovingLabels(X, Y, 10)
    list_dec = [(x, list(Y).count(x)) for x in set(list(Y))]
    print("{:-^80}".format("Cleaner Objects"))
    for k, (label, nbelts) in enumerate(list_dec):
        print(f"{k:<6} {nbelts:>6}")
    print(f"Total number of pages: {sum((nb for _, nb in list_dec)):->15}")
    
    ##### CROSS-VALIDATION SCORE #####
    if args.cross_val:
        mean_gbc3 = []
        ss = ShuffleSplit(n_splits=5)
        for k, (train, test) in enumerate(ss.split(X, Y)):
            print(f"FOLD [{k}]")
            gbc_cv = GradientBoostingClassifier().fit(X[train], Y[train])
            preds_gbc = gbc_cv.predict(X[test])
            score_gbc = f1_score(Y[test], preds_gbc, average='macro')
            mean_gbc3.append(score_gbc)
        print(f"The cross-val score: {np.mean(mean_gbc3)}")
    
    ##### TUNING PARAMETERS #####
    if args.grid_search:
        param_gbc = {'n_estimators': range(50, 200, 25),
                     'max_depth': range(2, 5)}
        gbc = GradientBoostingClassifier()
        clf = GridSearchCV(gbc, param_gbc, scoring='f1_macro', verbose=3)
        clf.fit(X, Y)
        print(f"best params: {clf.best_params_}")
        print(f"best score: {clf.best_score_}")
    
    ##### SAVE MODEL #####
    if args.save_model:
        if args.grid_search:
            gbc3 = GradientBoostingClassifier(**clf.best_params_).fit(X, Y)
            with open(path_cm + 'gbc3', 'wb') as f: pickle.dump(gbc3, f)
        else:
            gbc3 = GradientBoostingClassifier().fit(X, Y)
            with open(path_cm + 'gbc3', 'wb') as f: pickle.dump(gbc3, f)


##############################################################################
#                                                                            #
#                 MODEL - LAYOUTS WITH 4 ARTICLES                            #
#                                                                            #
##############################################################################


if dict_model[4]:
    nb_pgmin = 20
    nb_arts = 4
    args_cr = [dico_bdd, dict_arts, list_mdp_data, nb_arts, nb_pgmin]
    X, Y, dict_labels = without_layout.CreateXYFromScratch(*args_cr, list_features)
    # Only 274 pages in our data for 16 labels
    list_dec = [(x, list(Y).count(x)) for x in set(list(Y))]
    X, Y = RemovingLabels(X, Y, 10)
    list_dec = [(x, list(Y).count(x)) for x in set(list(Y))]
    print("{:-^80}".format("Cleaner Objects"))
    for k, (label, nbelts) in enumerate(list_dec):
        print(f"{k:<6} {nbelts:>6}")
    print(f"Total number of pages: {sum((nb for _, nb in list_dec)):->15}")

    ##### CROSS-VALIDATION SCORE #####
    if args.cross_val:
        mean_gbc4 = []
        ss = ShuffleSplit(n_splits=5)
        for k, (train, test) in enumerate(ss.split(X, Y)):
            print(f"FOLD [{k}]")
            gbc_cv = GradientBoostingClassifier().fit(X[train], Y[train])
            preds_gbc = gbc_cv.predict(X[test])
            score_gbc = f1_score(Y[test], preds_gbc, average='macro')
            mean_gbc4.append(score_gbc)
        print(f"The cross-val score: {np.mean(mean_gbc4)}")
    
    ##### TUNING PARAMETERS #####
    if args.grid_search:
        param_gbc = {'n_estimators': range(50, 200, 25),
                     'max_depth': range(2, 5)}
        gbc = GradientBoostingClassifier()
        clf = GridSearchCV(gbc, param_gbc, scoring='f1_macro', verbose=3)
        clf.fit(X, Y)
        print(f"best params: {clf.best_params_}")
        print(f"best score: {clf.best_score_}")
    
    ##### SAVE MODEL #####
    if args.save_model:
        if args.grid_search:
            gbc4 = GradientBoostingClassifier(**clf.best_params_).fit(X, Y)
            with open(path_cm + 'gbc4', 'wb') as f: pickle.dump(gbc4, f)
        else:
            gbc4 = GradientBoostingClassifier().fit(X, Y)
            with open(path_cm + 'gbc4', 'wb') as f: pickle.dump(gbc4, f)


##############################################################################
#                                                                            #
#                 MODEL - LAYOUTS WITH 5 ARTICLES                            #
#                                                                            #
##############################################################################


if dict_model[5]:
    nb_pgmin = 20
    nb_arts = 5
    args_cr = [dico_bdd, dict_arts, list_mdp_data, nb_arts, nb_pgmin]
    X, Y, dict_labels = without_layout.CreateXYFromScratch(*args_cr, list_features)
    # Only 88 pages in our data for 5 labels
    list_dec = [(x, list(Y).count(x)) for x in set(list(Y))]
    X, Y = RemovingLabels(X, Y, 10)
    list_dec = [(x, list(Y).count(x)) for x in set(list(Y))]
    print("{:-^80}".format("Cleaner Objects"))
    for k, (label, nbelts) in enumerate(list_dec):
        print(f"{k:<6} {nbelts:>6}")
    print(f"Total number of pages: {sum((nb for _, nb in list_dec)):->15}")

    ##### CROSS-VALIDATION SCORE #####
    if args.cross_val:
        mean_gbc5 = []
        ss = ShuffleSplit(n_splits=5)
        for k, (train, test) in enumerate(ss.split(X, Y)):
            print(f"FOLD [{k}]")
            gbc_cv = GradientBoostingClassifier().fit(X[train], Y[train])
            preds_gbc = gbc_cv.predict(X[test])
            score_gbc = f1_score(Y[test], preds_gbc, average='macro')
            mean_gbc5.append(score_gbc)
        print(f"The cross-val score: {np.mean(mean_gbc5)}")

    ##### TUNING PARAMETERS #####
    if args.grid_search:
        param_gbc = {'n_estimators': range(50, 200, 25),
                     'max_depth': range(2, 5)}
        gbc = GradientBoostingClassifier()
        clf = GridSearchCV(gbc, param_gbc, scoring='f1_macro', verbose=3)
        clf.fit(X, Y)
        print(f"best params: {clf.best_params_}")
        print(f"best score: {clf.best_score_}")

    #### SAVE MODEL #####
    if args.save_model:
        if args.grid_search:
            gbc5 = GradientBoostingClassifier(**clf.best_params_).fit(X, Y)
            with open(path_cm + 'gbc5', 'wb') as f: pickle.dump(gbc5, f)
        else:
            gbc5 = GradientBoostingClassifier().fit(X, Y)
            with open(path_cm + 'gbc5', 'wb') as f: pickle.dump(gbc5, f)

