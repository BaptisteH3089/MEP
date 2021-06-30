#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import f1_score
import pickle
import without_layout
import numpy as np

"""
For now, I saved the models with default parameters. I should tune the
hyper-parameters to save the models with the optimal combinations of
parameters.
Eventually I can check if it's better when I increase the number of features.
"""

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

save = False

##############################################################################
#                                                                            #
#                 MODEL - LAYOUTS WITH 2 ARTICLES                            #
#                                                                            #
##############################################################################

nb_pgmin = 20
nb_arts = 2
args_cr = [dico_bdd, dict_arts, list_mdp_data, nb_arts, nb_pgmin]
X, Y, dict_labels = without_layout.CreateXYFromScratch(*args_cr, list_features)

# The model with the layouts with 2 articles is trained with 1136 and there
# are 24 different classes or layouts

list_dec = [(x, list(Y).count(x)) for x in set(list(Y))]
# The decomposition is 
# [(0, 41), (1, 2), (2, 173), (3, 111), (4, 34), (5, 30), (6, 34), (7, 98),
# (8, 36), (9, 66), (10, 183), (11, 27), (12, 16), (13, 59), (14, 24),
# (15, 46), (16, 46), (17, 61), (18, 21), (19, 11), (20, 15), (21, 1), (22, 1)] 

mean_gbc2 = []
ss = ShuffleSplit(n_splits=5)
for k, (train, test) in enumerate(ss.split(X, Y)):
    print(f"FOLD [{k}]")
    gbc_cv = GradientBoostingClassifier().fit(X[train], Y[train])
    preds_gbc = gbc_cv.predict(X[test])
    score_gbc = f1_score(Y[test], preds_gbc, average='macro')
    mean_gbc2.append(score_gbc)
print(f"The cross-val score: {np.mean(mean_gbc2)}")
    
if save is True:
    gbc2 = GradientBoostingClassifier().fit(X, Y)
    with open(path_cm + 'gbc2', 'wb') as f: pickle.dump(gbc2, f)

##############################################################################
#                                                                            #
#                 MODEL - LAYOUTS WITH 3 ARTICLES                            #
#                                                                            #
##############################################################################

nb_arts = 3
args_cr = [dico_bdd, dict_arts, list_mdp_data, nb_arts, nb_pgmin]
X, Y, dict_labels = without_layout.CreateXYFromScratch(*args_cr, list_features)

# Only 141 pages in our data for 7 labels
list_dec = [(x, list(Y).count(x)) for x in set(list(Y))]
# The decomposition is: [(0, 32), (1, 20), (2, 35), (3, 9), (4, 22), (5, 23)]

mean_gbc3 = []
ss = ShuffleSplit(n_splits=5)
for k, (train, test) in enumerate(ss.split(X, Y)):
    print(f"FOLD [{k}]")
    gbc_cv = GradientBoostingClassifier().fit(X[train], Y[train])
    preds_gbc = gbc_cv.predict(X[test])
    score_gbc = f1_score(Y[test], preds_gbc, average='macro')
    mean_gbc3.append(score_gbc)
print(f"The cross-val score: {np.mean(mean_gbc3)}")

if save is True:
    gbc3 = GradientBoostingClassifier().fit(X, Y)
    with open(path_cm + 'gbc3', 'wb') as f: pickle.dump(gbc3, f)

##############################################################################
#                                                                            #
#                 MODEL - LAYOUTS WITH 4 ARTICLES                            #
#                                                                            #
##############################################################################

nb_arts = 4
args_cr = [dico_bdd, dict_arts, list_mdp_data, nb_arts, nb_pgmin]
X, Y, dict_labels = without_layout.CreateXYFromScratch(*args_cr, list_features)

# Only 274 pages in our data for 16 labels
list_dec = [(x, list(Y).count(x)) for x in set(list(Y))]
# The decomposition is:
# [(0, 11), (1, 65), (2, 8), (3, 32), (4, 10), (5, 43), (6, 26), (7, 13),
# (8, 4), (9, 9), (10, 3), (11, 27), (12, 2), (13, 18), (14, 3)]

mean_gbc4 = []
ss = ShuffleSplit(n_splits=5)
for k, (train, test) in enumerate(ss.split(X, Y)):
    print(f"FOLD [{k}]")
    gbc_cv = GradientBoostingClassifier().fit(X[train], Y[train])
    preds_gbc = gbc_cv.predict(X[test])
    score_gbc = f1_score(Y[test], preds_gbc, average='macro')
    mean_gbc4.append(score_gbc)
print(f"The cross-val score: {np.mean(mean_gbc4)}")

if save is True:
    gbc4 = GradientBoostingClassifier().fit(X, Y)
    with open(path_cm + 'gbc4', 'wb') as f: pickle.dump(gbc4, f)

##############################################################################
#                                                                            #
#                 MODEL - LAYOUTS WITH 5 ARTICLES                            #
#                                                                            #
##############################################################################

nb_arts = 5
args_cr = [dico_bdd, dict_arts, list_mdp_data, nb_arts, nb_pgmin]
X, Y, dict_labels = without_layout.CreateXYFromScratch(*args_cr, list_features)

# Only 88 pages in our data for 5 labels
list_dec = [(x, list(Y).count(x)) for x in set(list(Y))]
# The decomposition is: [(0, 19), (1, 38), (2, 12), (3, 19)]

mean_gbc5 = []
ss = ShuffleSplit(n_splits=5)
for k, (train, test) in enumerate(ss.split(X, Y)):
    print(f"FOLD [{k}]")
    gbc_cv = GradientBoostingClassifier().fit(X[train], Y[train])
    preds_gbc = gbc_cv.predict(X[test])
    score_gbc = f1_score(Y[test], preds_gbc, average='macro')
    mean_gbc5.append(score_gbc)
print(f"The cross-val score: {np.mean(mean_gbc5)}")

if save is True:
    gbc5 = GradientBoostingClassifier().fit(X, Y)
    with open(path_cm + 'gbc5', 'wb') as f: pickle.dump(gbc5, f)






