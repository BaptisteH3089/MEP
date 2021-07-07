#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 16:15:01 2021

@author: baptistehessel

Script used to create the dictionary dict_page_array

Object necessary:
    - dico_pages

"""
import pickle
import numpy as np
import argparse

str_desc = 'Creation dict id_page: array_layout.'
parser = argparse.ArgumentParser(description=str_desc)
parser.add_argument('path_customer',
                    help="The repertory where there is the dico_bdd",
                    type=str)
parser.add_argument('file_out',
                    help="The path where the dict will be created.",
                    type=str)
parser.add_argument('--save_dict',
                    help="Whether to save or not the dict.",
                    type=bool,
                    default=True)
args = parser.parse_args()

# Loading dictionary with all the pages
with open(args.path_customer + 'dict_pages', 'rb') as f:
    dict_pages = pickle.load(f)

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
    if s == 10:
        break
    s += 1

# Save the dict
if args.save_dict:
    with open(args.file_out, 'wb') as f:
        pickle.dump(dict_page_array, f)
