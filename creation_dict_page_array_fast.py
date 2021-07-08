#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 09:15:05 2021

@author: baptistehessel

Script that creates an object very similar to dict_page_array but used to
do faster searches. It is of the form:
    {nb_modules_layout: dict_layouts_nb_modules_layout, ...}

Objects necessary:
    - dict_page_array.

The arguments of the script are:
    - path_customer ('The path of the data.')
    - --save_dict (default=True)

"""
import pickle
import itertools


def CreationDictPageArrayFast(dict_page_array, path_customer, save_dict=True):
    # Add the number of modules
    for key, value in dict_page_array.items():
        dict_page_array[key] = (value, len(value))
    # Sort the dict by number of modules
    dict_page_array = dict(sorted(dict_page_array.items(), key=lambda x: x[1][1]))
    # Groupby the number of modules in the layout.
    new_dict = {}
    for unique_key, group in itertools.groupby(dict_page_array.items(),
                                               key=lambda x: x[1][1]):
        new_dict[unique_key] = dict(group)
    # Remove the number of modules in the value.
    for unique_key, dict_small in new_dict.items():
        for key, little_tuple in dict_small.items():
            dict_small[key] = little_tuple[0]
    # Print the results.
    for unique_key, dict_small in new_dict.items():
        if unique_key > 10:
            break
        print(f"The layouts with {unique_key} modules.")
        print((f"The 5 first elements:"))
        for i, (id_page, layout_array) in enumerate(dict_small.items()):
            print(f"{id_page} \n {layout_array}")
            if i == 4:
                break
    # Save the dict.
    if save_dict:
        with open(path_customer + 'dict_page_array_fast', 'wb') as f:
            pickle.dump(new_dict, f)
