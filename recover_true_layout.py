#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 16:31:44 2021

@author: baptistehessel

Just some code to see if we can associate a true layout to the array
associated to a page.

"""
import pickle
import creation_dict_layout

path_cm = '/Users/baptistehessel/Documents/DAJ/MEP/montageIA/data/CM/'

# Loading dictionary with all the pages
with open(path_cm + 'dict_page_array', 'rb') as f:
    dict_page_array = pickle.load(f)
# Loading dictionary with all the pages
with open(path_cm + 'dict_layouts_small', 'rb') as f:
    dict_layouts = pickle.load(f)

# It's long. Roughly 2 minutes.
nb_page_corr = 0
same_shape = 0
for idp, arrayp in dict_page_array.items():
    # We search for a correspondance in dict_layouts
    for idl, dictl in dict_layouts.items():
        true_layout = np.array([cart[:-2] for cart in dictl['cartons']])
        if true_layout.shape == arrayp.shape:
            same_shape += 1
            if CompareTwoLayouts(true_layout, arrayp):
                nb_page_corr += 1

