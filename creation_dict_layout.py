#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 09:49:35 2021

@author: baptistehessel

Script used to extract the data about the layout in the 15.000 pages about CM
I have.
"""
import pickle
import numpy as np

# Loading of the dictionary with all information on pages
path_cm = '/Users/baptistehessel/Documents/DAJ/MEP/montageIA/data/CM'
with open(path_cm + '/dico_pages','rb') as f:
    dico_bdd = pickle.load(f)


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
    n = len(layout1)
    nb_correspondances = 0
    for i in range(n):
        for j in range(n):
            try:
                if np.allclose(layout1[j], layout2[i], atol=tol):
                    nb_correspondances += 1
            except Exception as e:
                str_exc = (f"An exception occurs with np.allclose: {e}\n"
                           f"layout1: {layout1}\n"
                           f"j: {j}, i: {i}"
                           f"layout2: {layout2}")
                print(str_exc)
    return True if nb_correspondances == n else False


##############################################################################
#                        PART I - BARE DICT LAYOUTS                          #
##############################################################################


def CreationDictLayoutSimple(dico_bdd):
    """

    Parameters
    ----------
    dico_bdd : dictionary
        The usual dictionary with all the info on the page.
        Of the form {id_page: {..., 'dico_page': {...}}, ...}

    Returns
    -------
    dict_layouts : dictionary
        A dictionary of the form {id_layout: {'nameTemplate': str,
                                              'cartons': list of tuples,
                                              'id_pages': list}} 

    """
    # Creation of the dictionary with all the layouts
    dict_layouts = {}
    cpt_error = 0
    cpt_layout = 0
    for id_page in dico_bdd.keys():
        id_layout = dico_bdd[id_page]['dico_page']['pageTemplate']
        # Check wether this layout is already in the dict
        if id_layout in dict_layouts.keys():
            cpt_layout += 1
            dict_layouts[id_layout]['id_pages'].append(id_page)
        else:
            name_template = dico_bdd[id_page]['dico_page']['nameTemplate']
            feat_carton = ['x', 'y', 'width', 'height', 'nbCol', 'nbPhoto']
            list_cartons = []
            for dict_carton in dico_bdd[id_page]['dico_page']['cartons'].values():
                list_cartons.append(tuple(dict_carton[x] for x in feat_carton))
            dict_layouts[id_layout] = {'nameTemplate': name_template,
                                       'cartons': list_cartons,
                                       'id_pages': [id_page]}
    f = lambda x: len(x[1]['id_pages'])
    dict_layouts = dict(sorted(dict_layouts.items(), key=f, reverse=True))
    return dict_layouts


##############################################################################
#               PART II - DICT LAYOUTS WITH VERIFICATIONS                    #
##############################################################################


def CreationDictLayoutVerif(dico_bdd):
    # Creation of the dictionary with all the layouts
    dd_layouts_sm = {}
    cpt_layout = 0
    for id_page in dico_bdd.keys():
            id_layout = dico_bdd[id_page]['dico_page']['pageTemplate']
            # Check wether this layout is already in the dict
            if id_layout in dd_layouts_sm.keys():
                cpt_layout += 1
                # HERE I SHOULD DO SOME VERIFS
                # 1. Check if the number of articles makes sense
                # 2. Verify if the zonnings of the articles correspond
                nb_arts_pg = len(dico_bdd[id_page]['dico_page']['articles'])
                nb_cartons = len(dd_layouts_sm[id_layout]['cartons'])
                if nb_cartons == nb_arts_pg:
                    modules_layout = []
                    for module in dd_layouts_sm[id_layout]['cartons']:
                        modules_layout.append(module[:-2])
                    layout_template = np.array(modules_layout)
                    cartons_page = []
                    for ida, dicoa in dico_bdd[id_page]['dico_page']['articles'].items():
                        cartons_page.append([dicoa[x] for x in ['x', 'y', 'width', 'height']])
                    layout_page = np.array(cartons_page)
                    if CompareTwoLayouts(layout_template, layout_page):
                        dd_layouts_sm[id_layout]['id_pages'].append(id_page)
            else:
                name_template = dico_bdd[id_page]['dico_page']['nameTemplate']
                feat_carton = ['x', 'y', 'width', 'height', 'nbCol', 'nbPhoto']
                list_cartons = []
                for dict_carton in dico_bdd[id_page]['dico_page']['cartons'].values():
                    list_cartons.append(tuple(dict_carton[x] for x in feat_carton))
                # Add the array associated to the layout
                array_layout = np.array([carton[:-2] for carton in list_cartons])
                dd_layouts_sm[id_layout] = {'nameTemplate': name_template,
                                            'cartons': list_cartons,
                                            'id_pages': [id_page],
                                            'array': array_layout}
    f = lambda x: len(x[1]['id_pages'])
    dd_layouts_sm = dict(sorted(dd_layouts_sm.items(), key=f, reverse=True))
    return dd_layouts_sm


# Wheter to save or not the big dict without verifications.
save_dict_big = False
if save_dict_big:
    dict_layouts_big = CreationDictLayoutSimple(dico_bdd)
    with open(path_cm + '/dict_layouts_big', 'wb') as f:
        pickle.dump(dict_layouts_big, f)

# Wheter to save or not the small dict.
save_dict_small = False
if save_dict_small:
    dd_layouts_sm = CreationDictLayoutVerif(dico_bdd)
    for i, (key, val) in enumerate(dd_layouts_sm.items()):
        print(key)
        print(val)
        if i == 10:
            break
    with open(path_cm + '/dict_layouts_small', 'wb') as f:
        pickle.dump(dd_layouts_sm, f)
