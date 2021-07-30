#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 16:31:44 2021

@author: baptistehessel

Just some code to see if we can associate a true layout to an array
associated to a page. Sometimes, we find a correspondance, but two layouts
with the same MelodyId can be significantly different.

"""
import pickle
import numpy as np
import module_montage

path_cm = '/Users/baptistehessel/Documents/DAJ/MEP/montageIA/data/CM/'

# Loading dictionary with all the pages
with open(path_cm + 'dict_page_array', 'rb') as f:
    dict_page_array = pickle.load(f)
# Loading dictionary with all the pages
with open(path_cm + 'dict_layouts_small', 'rb') as f:
    dict_layouts = pickle.load(f)


def NbOfCorrespondances(dict_page_array, dict_layouts):
    """
    Returns the number of layouts extracted from the articles in the database
    (which can be layouts modified by the user) can be associated by a
    true layout that has a MelodyId.
    For each layout in dict_page_array, we search in dict_layouts if we find a
    correspondance.

    """
    # It's long. Roughly 2 minutes.
    nb_page_corr = 0
    for idp, arrayp in dict_page_array.items():
        # We search for a correspondance in dict_layouts
        for idl, dictl in dict_layouts.items():
            true_layout = np.array([cart[:-2] for cart in dictl['cartons']])
            if true_layout.shape == arrayp.shape:
                if module_montage.CompareTwoLayouts(true_layout, arrayp):
                    nb_page_corr += 1
    return nb_page_corr

