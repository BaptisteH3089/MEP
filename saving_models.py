#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
import configparser
import numpy as np
import argparse
import pickle
import json
import without_layout

"""
@author: baptistehessel

Script that cross-validates and saves the wanted model. It is also possible
to perform a grid_search to find the best combinations of parameters.

We train 4 models in that scripts. They correspond to layouts with 2, 3, 4 or
5 articles in it. (Don't enough data to do more. Only 1 layout with more than
8 pages for the pages with 6 articles for instance.)

The models trained in that script are Gradient Boosting Classifiers that
predict the best layout for a given vector page.

Objects necessary:
    - dict_pages
    - dict_arts
    - list_mdp

If you save the models, they are stored in the same repertory as the data.

"""

# ARG PARSER - SERVOR. Initilisation of the arguements of the script
# montage_ia.xml for the start webservice montage IA.
parser = argparse.ArgumentParser(description='Models used to predict layouts.')
parser.add_argument('customer_code',
                    help='the code of the customer.',
                    type=str)
parser.add_argument('path_param_file',
                    help='the path to the parameters file.',
                    type=str)
parser.add_argument('--selected_models',
                    help="The model(s) to consider.",
                    type=int,
                    nargs='+',
                    default=0)
parser.add_argument('--save_model_false',
                    help="Whether to save or not the models.",
                    action='store_false')
parser.add_argument('--cross_val',
                    help="Whether to show cross-validation scores.",
                    action='store_true')
parser.add_argument('--grid_search',
                    help="Whether to search for best parameters.",
                    action='store_true')
args = parser.parse_args()

print("args.selected_models:", args.selected_models)
print("args.save_model_false:", args.save_model_false)

if type(args.selected_models) is int:
    args.selected_models = [args.selected_models]

dict_model = {2: False, 3: False, 4: False, 5: False}
if 0 in args.selected_models:
    dict_model = {2: True, 3: True, 4: True, 5: True}
else:
    for num_model in args.selected_models:
        dict_model[num_model] = True


# PARAM FILE. We use the param file of the app to get the features and the
# path to the data.
# Parsing of the parameters file.
config = configparser.RawConfigParser()
# Reading of the parameters file (param_montage_ia.py) of the appli montageIA.
config.read(args.path_param_file)
# The parameters extracted from the parameters file.
param = {'list_features': json.loads(config.get('FEATURES', 'list_features')),
         'path_data': config.get('DATA', 'path_data')}
list_features = param['list_features']
path_customer = param['path_data'] + '/' + args.customer_code + '/'

# The dict with all the pages available
with open(path_customer + 'dict_pages', 'rb') as file:
    dict_pages = pickle.load(file)
# The dict {ida: dicoa, ...}
with open(path_customer + 'dict_arts', 'rb') as file:
    dict_arts = pickle.load(file)
# The list of triplet (nb_pages_using_mdp, array_mdp, list_ids)
with open(path_customer + 'list_mdp', 'rb') as file:
    list_mdp_data = pickle.load(file)


def RemovingLabels(X, Y, nbmin):
    """
    Remove from X and Y the rows corresponding to a layout that appears less
    than "nbmin" times in our data.

    Parameters
    ----------
    X: numpy array
        Matrix with the vectors page.

    Y: numpy array
        Matrix with the labels of the layouts.

    nbmin: int
        The min number of pages using a layout. If a layout uses less than
        this number, we remove it from the matrices.

    Returns
    -------
    X, Y: numpy array
        The smaller matrices without elements with less than "nbmin"
        occurrences.

    """
    smaller_X = np.zeros(1)
    smaller_Y = []
    init = False
    dict_labels = {x: list(Y).count(x) for x in set(list(Y))}
    for line_X, label_Y in zip(X, Y):
        line_x = np.array(line_X, ndmin=2)
        # We check the condition
        if dict_labels[label_Y] <= nbmin:
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
#                    nb_pgmin = 20. Filter < 10                              #
#                                                                            #
##############################################################################


if dict_model[2]:
    print(f"\nLayouts with 2 articles.\n")
    nb_pgmin = 20
    nb_arts = 2
    args_cr = [dict_pages, dict_arts, list_mdp_data, nb_arts, nb_pgmin]
    args_cr += [list_features, 1]
    X, Y, dict_labels = without_layout.CreateXYFromScratch(*args_cr)
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
    if args.save_model_false:
        if args.grid_search:
            gbc2 = GradientBoostingClassifier(**clf.best_params_).fit(X, Y)
            with open(path_customer + 'gbc2', 'wb') as f:
                pickle.dump(gbc2, f)
        else:
            gbc2 = GradientBoostingClassifier().fit(X, Y)
            print(f"Training score: {gbc2.score(X, Y):.3f}")
            with open(path_customer + 'gbc2', 'wb') as f:
                pickle.dump(gbc2, f)


##############################################################################
#                                                                            #
#                 MODEL - LAYOUTS WITH 3 ARTICLES                            #
#                   nb_pgmin = 15. Filter < 10                               #
#                                                                            #
##############################################################################


if dict_model[3]:
    print(f"\nLayouts with 3 articles.\n")
    nb_pgmin = 15
    nb_arts = 3
    args_cr = [dict_pages, dict_arts, list_mdp_data, nb_arts, nb_pgmin]
    args_cr += [list_features, 1]
    X, Y, dict_labels = without_layout.CreateXYFromScratch(*args_cr)
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
    if args.save_model_false:
        if args.grid_search:
            gbc3 = GradientBoostingClassifier(**clf.best_params_).fit(X, Y)
            with open(path_customer + 'gbc3', 'wb') as f:
                pickle.dump(gbc3, f)
        else:
            gbc3 = GradientBoostingClassifier().fit(X, Y)
            print(f"Training score: {gbc3.score(X, Y):.3f}")
            with open(path_customer + 'gbc3', 'wb') as f:
                pickle.dump(gbc3, f)


##############################################################################
#                                                                            #
#                 MODEL - LAYOUTS WITH 4 ARTICLES                            #
#                   nb_pgmin = 13. Filter < 10                               #
#                                                                            #
##############################################################################


if dict_model[4]:
    print(f"\nLayouts with 4 articles.\n")
    nb_pgmin = 13
    nb_arts = 4
    args_cr = [dict_pages, dict_arts, list_mdp_data, nb_arts, nb_pgmin]
    args_cr += [list_features, 1]
    X, Y, dict_labels = without_layout.CreateXYFromScratch(*args_cr)
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
    if args.save_model_false:
        if args.grid_search:
            gbc4 = GradientBoostingClassifier(**clf.best_params_).fit(X, Y)
            with open(path_customer + 'gbc4', 'wb') as f:
                pickle.dump(gbc4, f)
        else:
            gbc4 = GradientBoostingClassifier().fit(X, Y)
            print(f"Training score: {gbc4.score(X, Y):.3f}")
            with open(path_customer + 'gbc4', 'wb') as f:
                pickle.dump(gbc4, f)


##############################################################################
#                                                                            #
#                 MODEL - LAYOUTS WITH 5 ARTICLES                            #
#                   nb_pgmin = 10. Filter < 10                               #
#                                                                            #
##############################################################################


if dict_model[5]:
    print(f"\nLayouts with 5 articles.\n")
    nb_pgmin = 10
    nb_arts = 5
    args_cr = [dict_pages, dict_arts, list_mdp_data, nb_arts, nb_pgmin]
    args_cr += [list_features, 1]
    X, Y, dict_labels = without_layout.CreateXYFromScratch(*args_cr)
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
    if args.save_model_false:
        if args.grid_search:
            gbc5 = GradientBoostingClassifier(**clf.best_params_).fit(X, Y)
            with open(path_customer + 'gbc5', 'wb') as f:
                pickle.dump(gbc5, f)
        else:
            gbc5 = GradientBoostingClassifier().fit(X, Y)
            print(f"Training score: {gbc5.score(X, Y):.3f}")
            with open(path_customer + 'gbc5', 'wb') as f:
                pickle.dump(gbc5, f)
