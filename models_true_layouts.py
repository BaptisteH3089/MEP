#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 10:47:19 2021

@author: baptistehessel
"""

import pickle

path_cm = '/Users/baptistehessel/Documents/DAJ/MEP/montageIA/data/CM/'

# Loading dictionary with all the layouts and the pages that use them
with open(path_cm + 'dict_layouts', 'rb') as f:
    dict_layouts = pickle.load(f)
# Loading dictionary with all the pages
with open(path_cm + 'dico_pages', 'rb') as f:
    dict_pages = pickle.load(f)
# Loading dictionary with all the articles
with open(path_cm + 'dico_arts', 'rb') as f:
    dict_arts = pickle.load(f)

# First, we isolate layouts with the same number of modules (between 2 and 8)
# We create a dictionary of the form {2: list_ids_layouts_2_modules, ...}
dict_nbmodules = {i: [] for i in range(2, 9)}
for id_layout, dict_val in dict_layouts.items():
    nb_modules = len(dict_val['cartons'])
    try:
        dict_nbmodules[nb_modules].append(id_layout)
    except:
        print("Different number of modules", nb_modules)

# Now, I should have the same kind of object with the pages
# I have it with dict_layouts[id_layout][id_pages]


#%%

# Let first construct the matrixes with the layouts with 2 articles
list_features = ['nbSign', 'nbBlock', 'abstract', 'syn']
list_features += ['exergue', 'title', 'secTitle', 'supTitle']
list_features += ['subTitle', 'nbPhoto', 'aireImg', 'aireTot']
list_features += ['petittitre', 'quest_rep', 'intertitre']


def CreationVectorsPage(id_page, dico_bdd, dict_arts, list_feat):
    """
    With the id of a vector page:
        - we extract the ids of the articles in that page
        - we contruct each vector article
        - we concatenate these vectors
    """
    list_ids_art = list(dico_bdd[id_page]['dico_page']['articles'].keys())
    for i, ida in enumerate(list_ids_art):
        vect_art = np.array([dict_arts[ida][x] for x in list_feat], ndmin=2)
        if i == 0:
            vect_page = vect_art
        else:
            vect_page = np.concatenate((vect_page, vect_art), axis=1)
    return vect_page


def CreationValidationMatrixes(list_ids_layout,
                               dict_layouts,
                               dict_bdd,
                               dict_arts,
                               list_feat):
    for id_layout in list_ids_layout:
        # All the ids of pages using that layout
        list_ids_pages = dict_layouts[id_layout]['id_pages']
        init = False
        for i, id_page in enumerate(list_ids_pages):
            vect_page = CreationVectorsPage(id_page, dico_bdd, dict_arts, list_feat)
            if init is False:
                if vect_page.shape[1] == 45:
                    X = vect_page
                    Y = [id_layout]
                    init = True
            else:
                try:
                    X = np.concatenate((X, vect_page))
                    Y.append(id_layout)
                except Exception as e:
                    print(e)
                    print(dico_bdd[id_page]['dico_page']['articles'].keys())
    return X, np.array(Y)

list_ids_layout = dict_nbmodules[3]
X, Y = CreationValidationMatrixes(list_ids_layout, dict_layouts, dict_pages,
                                  dict_arts, list_features)

print(f"shape of X: {X.shape}")
print(f"shape of Y: {Y.shape}")

for id_layout in list_ids_layout:
    print(dict_layouts[id_layout]['cartons'])


