#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 16:15:01 2021

@author: baptistehessel

Script used to create the dictionary dict_page_array

"""
import pickle
import numpy as np

path_cm = '/Users/baptistehessel/Documents/DAJ/MEP/montageIA/data/CM/'

# Loading dictionary with all the pages
with open(path_cm + 'dico_pages', 'rb') as f:
    dict_pages = pickle.load(f)

feat_module = ['x', 'y', 'width', 'height']
dict_page_array = {}
for id_page, dicop in dict_pages.items():
    modules_page = []
    for dicoa in dicop['dico_page']['articles'].values():
        modules_page.append([dicoa[x] for x in feat_module])
    dict_page_array[id_page] = np.array(modules_page)

# Check if this is OK
s = 0
for idp, arrayp in dict_page_array.items():
    print(idp)
    print(arrayp)
    if s == 10:
        break
    s += 1

save_dict = True
if save_dict:
    with open(path_cm + 'dict_page_array', 'wb') as f:
        pickle.dump(dict_page_array, f)

