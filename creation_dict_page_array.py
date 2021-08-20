#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 16:15:01 2021

@author: baptistehessel

Script used to create the dictionary dict_page_array

Object necessary:
    - dico_pages

Creates the dict_page_array which is of the form:
    - {id_page: numpy_array_layout, ...}.

"""
import pickle
import numpy as np


def CreationDictPageArray(dict_pages, path_customer, save_dict=True):
    feat_module = ['x', 'y', 'width', 'height']
    dict_page_array = {}
    for id_page, dicop in dict_pages.items():
        modules_page = []
        for dicoa in dicop['articles'].values():
            modules_page.append([dicoa[x] for x in feat_module])
        dict_page_array[id_page] = np.array(modules_page)
    # Check if this is OK
    s = 0
    print("{:-^80}".format("Visualisation of the results"))
    for idp, arrayp in dict_page_array.items():
        print(idp)
        print(arrayp)
        if s == 2:
            break
        s += 1
    # Save the dict
    if save_dict:
        with open(path_customer + 'dict_page_array', 'wb') as f:
            pickle.dump(dict_page_array, f)
