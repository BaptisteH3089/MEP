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

path_cm = '/Users/baptistehessel/Documents/DAJ/MEP/montageIA/data/CM/'

# Loading dictionary with all the pages
with open(path_cm + 'dict_page_array', 'rb') as f:
    dict_page_array = pickle.load(f)
# Loading dictionary with all the pages
with open(path_cm + 'dict_layouts_small', 'rb') as f:
    dict_layouts = pickle.load(f)


def CompareTwoLayouts(layout1, layout2, tol=20):
    """

    Useful function in many cases.

    Parameters
    ----------
    layout1 : numpy array
        np.array([[x, y, w, h], [x, y, w, h], ...]).
    layout2 : numpy array
        np.array([[x, y, w, h], [x, y, w, h], ...]).

    Returns
    -------
    bool
        True if the two layouts are approximately the same.

    """
    if layout1.shape != layout2.shape:
        return False
    n = len(layout1)
    nb_correspondances = 0
    for i in range(n):
        for j in range(n):
            try:
                if np.allclose(layout1[j], layout2[i], atol=tol):
                    nb_correspondances += 1
            except Exception as e:
                str_exc = (f"An exception occurs with np.allclose: \n{e}\n"
                           f"layout1: {layout1}\n"
                           f"j: {j}, i: {i}\n"
                           f"layout2: \n{layout2}")
                print(str_exc)
    return True if nb_correspondances == n else False


def NbOfCorrespondances(dict_page_array, dict_layouts):
    """
    Returns the number of layouts extracted from the articles in the database
    (which can be layouts modified by the user) can be associated by a
    true layout that has a MelodyId.
    Pour chaque layout dans dict_page_array, on cherche dans dict_layouts
    si on trouve une correspondance.

    """
    # It's long. Roughly 2 minutes.
    nb_page_corr = 0
    for idp, arrayp in dict_page_array.items():
        # We search for a correspondance in dict_layouts
        for idl, dictl in dict_layouts.items():
            true_layout = np.array([cart[:-2] for cart in dictl['cartons']])
            if true_layout.shape == arrayp.shape:
                if CompareTwoLayouts(true_layout, arrayp):
                    nb_page_corr += 1
    return nb_page_corr



