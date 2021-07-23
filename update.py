#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 14:58:08 2021

@author: baptistehessel
"""
import pickle
import matplotlib.pyplot as plt
import numpy as np

path_c = '/Users/baptistehessel/Documents/DAJ/MEP/montageIA/data/CM2/'
with open(path_c + 'dict_pages', 'rb') as f:
    dict_pages = pickle.load(f)

l_scores = []
list_all_labels = []
for idp, dicop in dict_pages.items():
    for dicta in dicop['articles'].values():
        l_scores.append(dicta['score'])
        l_labels = [dicta[x] for x in ['isPrinc', 'isSec', 'isSub', 'isTer',
                                       'isMinor']]
        list_all_labels.append(l_labels)

plt.hist(l_scores, bins=30)
np.median(l_scores)

for l in list_all_labels:
    if sum(l) > 1:
        print(l)

#%%

# Stats about the types of articles

labels = np.argmax(list_all_labels, axis=1)

count_labels = [(lab, list(labels).count(lab)) for lab in set(labels)]


#%%

dict_type = {'main': [], 'sec': [], 'ter': [], 'minor': []}
for dicop in dict_pages.values():
    for dicta in dicop['articles'].values():
        if dicta['isPrinc'] == 1:
            dict_type['main'].append(dicta)
        elif max(dicta['isSec'], dicta['isSub']) == 1:
            dict_type['sec'].append(dicta)
        elif dicta['isTer'] == 1:
            dict_type['ter'].append(dicta)
        elif dicta['isMinor'] == 1:
            dict_type['minor'].append(dicta)

#%%

# Average of the vectors
list_features = ['nbSign', 'nbBlock', 'abstract', 'syn']
list_features += ['exergue', 'title', 'secTitle', 'supTitle']
list_features += ['subTitle', 'nbPhoto', 'aireImg', 'aireTot']
list_features += ['petittitre', 'quest_rep', 'intertitre']

# petittitre, intertitre, syn, abstract and quest_rep not so useful to
# determine the type of the article

dict_vectors = {key: [[d[x] for x in list_features] for d in dict_type[key]]
                for key in ['main', 'sec', 'ter', 'minor']}


for key, val in dict_vectors.items():
    print(key)
    print(np.mean(val, axis=0))
























