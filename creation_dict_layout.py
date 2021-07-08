#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 09:49:35 2021

@author: baptistehessel

Script used to extract the data about the layout in the 15.000 pages about CM
I have.

Object necessary:
    - dict_pages.

Imports the script recover_true_layout.

Creates the object /path_customer/dict_layouts_small.

dict_layouts_small is of the form:
    - {id_layout: {'nameTemplate': name_template,
                   'cartons': list_cartons,
                   'id_pages': list_ids_page,
                   'array': array_layout},
       ...}

"""
import pickle
import numpy as np
import recover_true_layout


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
    cpt_layout = 0
    for id_page in dico_bdd.keys():
        id_layout = dico_bdd[id_page]['pageTemplate']
        # Check wether this layout is already in the dict
        if id_layout in dict_layouts.keys():
            cpt_layout += 1
            dict_layouts[id_layout]['id_pages'].append(id_page)
        else:
            name_template = dico_bdd[id_page]['nameTemplate']
            feat_carton = ['x', 'y', 'width', 'height', 'nbCol', 'nbPhoto']
            list_cartons = []
            for dict_carton in dico_bdd[id_page]['cartons']:
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
    feat_carton = ['x', 'y', 'width', 'height', 'nbCol', 'nbPhoto']
    # Creation of the dictionary with all the layouts
    dd_layouts_sm = {}
    cpt_layout = 0
    n = len(dico_bdd)
    for i, id_page in enumerate(dico_bdd.keys()):
            id_layout = dico_bdd[id_page]['pageTemplate']
            # Check wether this layout is already in the dict
            if id_layout in dd_layouts_sm.keys():
                cpt_layout += 1
                # HERE I SHOULD DO SOME VERIFS
                # 1. Check if the number of articles makes sense
                # 2. Verify if the zonnings of the articles correspond
                nb_arts_pg = len(dico_bdd[id_page]['articles'])
                nb_cartons = len(dd_layouts_sm[id_layout]['cartons'])
                if nb_cartons == nb_arts_pg:
                    modules_layout = []
                    for module in dd_layouts_sm[id_layout]['cartons']:
                        modules_layout.append(module[:-2])
                    layout_template = np.array(modules_layout)
                    cartons = []
                    for ida, dicoa in dico_bdd[id_page]['articles'].items():
                        cartons.append([dicoa[x] for x in feat_carton[:-2]])
                    layout_page = np.array(cartons)
                    args_c = [layout_template, layout_page]
                    if recover_true_layout.CompareTwoLayouts(*args_c):
                        dd_layouts_sm[id_layout]['id_pages'].append(id_page)
            else:
                name_template = dico_bdd[id_page]['nameTemplate']
                list_cartons = []
                for dict_carton in dico_bdd[id_page]['cartons']:
                    list_cartons.append(tuple(dict_carton[x]
                                              for x in feat_carton))
                # Add the array associated to the layout
                array_layout = np.array([x[:-2] for x in list_cartons])
                dd_layouts_sm[id_layout] = {'nameTemplate': name_template,
                                            'cartons': list_cartons,
                                            'id_pages': [id_page],
                                            'array': array_layout}
            if i % (n//50) == 0:
                print(f"CreationDictLayoutVerif: {i/n:.2%}")
    f = lambda x: len(x[1]['id_pages'])
    dd_layouts_sm = dict(sorted(dd_layouts_sm.items(), key=f, reverse=True))
    return dd_layouts_sm


def CreationDictLayoutsSmall(dict_pages, path_customer, save_dict=True):
    # Show if the results make sense
    dd_layouts_sm = CreationDictLayoutVerif(dict_pages)
    print("{:-^80}".format("Visualisation of the results"))
    for i, (key, val) in enumerate(dd_layouts_sm.items()):
        print(key)
        print(val)
        if i == 10:
            break
    # Wheter to save or not the small dict.
    if save_dict:
        with open(path_customer + 'dict_layouts_small', 'wb') as f:
            pickle.dump(dd_layouts_sm, f)
