#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 14:45:44 2021

@author: baptistehessel

Script that creates the object list_mdp_data, which is of the form
[(nb_pages_using_layout, np.array(layout), list_ids_page_using_layout), ...]

"""
import pickle
import recover_true_layout
import numpy as np
import time
import argparse

parser = argparse.ArgumentParser(description='Creation list with layouts.')
parser.add_argument('file_out',
                    help="The path where the list will be created.",
                    type=str)
parser.add_argument('--save_list',
                    help="Whether to save or not the list.",
                    type=bool,
                    default=True)
args = parser.parse_args()

path_customer = '/Users/baptistehessel/Documents/DAJ/MEP/montageIA/data/CM/'

# The list of triplet (nb_pages_using_mdp, array_mdp, list_ids)
with open(path_customer + 'list_mdp', 'rb') as file:
    list_mdp_data = pickle.load(file)

with open(path_customer + 'dico_pages', 'rb') as file:
    dico_bdd = pickle.load(file)
# The dictionary of the form {id_page: np.array(layout), ...}
with open(path_customer + 'dict_page_array', 'rb') as file:
    dict_page_array = pickle.load(file)


def ArrayLayoutInList(array_layout, list_already_added):
    """

    Parameters
    ----------
    array_layout : numpy array
        A layout input.
    list_already_added : list
        A list of layouts.

    Returns
    -------
    bool
        True if array_layout is in list_already_added, else otherwise.

    """
    for array_layout_added in list_already_added:
        args = [array_layout, array_layout_added, 1]
        if recover_true_layout.CompareTwoLayouts(*args):
            return True
    return False


def CreationListLayouts(dict_page_array):
    """

    Parameters
    ----------
    dict_page_array : dictionary
        Dictionary of the form {id_page: array_page, ...} that corresponds to
        all the data we have.

    Returns
    -------
    list_tuple_layout : list of list
        A list of the form [[nb_pages, array_page, list_ids], ...].
        The list_ids contains all the pages that use the layout array_page.

    This function is long.

    """
    list_tuple_layout = []
    list_already_added = []
    n = len(dict_page_array)
    print(f"Beggining of the function CreationListLayouts")
    for i, (id_page, array_layout) in enumerate(dict_page_array.items()):
        # We only compare with the pages that didn't go through the first loop.
        list_layout = [1, array_layout, [id_page]]
        for id_page2, array_layout2 in list(dict_page_array.items())[i:]:
            # We check if we did not already associate the page to a layout.
            # Good but too long.
            # if ArrayLayoutInList(array_layout2, list_already_added) is False:
            args = [array_layout, array_layout2, 10]
            if recover_true_layout.CompareTwoLayouts(*args):
                list_layout[0] += 1
                list_layout[2].append(array_layout2)
                list_already_added.append(array_layout2)
        if i % (n // 100) == 0:
            print(f"CreationListLayouts: {100*i/n}%")
    return list_tuple_layout


t0 = time.time()
list_tuple_layout = CreationListLayouts(dict_page_array)
print("Duration CreationListLayouts: {time.time() - t0} sec.")

for i, (nb_pages, array_lay, list_ids) in enumerate(list_tuple_layout):
    print(f"The number of pages: {nb_pages}")
    print(f"{array_lay}")
    print(f"An extract of the list_ids: {list_ids[:4]}")
    if i == 10:
        break


print(f"The length of list_tuple_layout: {len(list_tuple_layout)}")
total_pages = sum((elt for elt, _, _ in list_tuple_layout))
print(f"The total number of pages: {total_pages}")

if args.save_list:
    with open(args.file_out, 'wb') as f:
        pickle.dump(list_tuple_layout, f)
