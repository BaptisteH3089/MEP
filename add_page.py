#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 11:28:25 2021

@author: baptistehessel

Script used to add the articles in a given page in the database.
Import by montage_ia.py.
It updates the elements:
    - dict_pages
    - dict_arts
    - list_mdp
    - dict_page_array
    - dict_page_array_fast
    - dict_layouts_small
If everything went well, it creates a xml file with:
    - <root><statut><'OK'></statut></root>

Scripts imported:
    - creation_dict_bdd
    - recover_true_layout

"""
from bs4 import BeautifulSoup as bs
import numpy as np
import creation_dict_bdd
import module_montage



def ExtrationDataOneFile(path_file):
    """
    Extraction des données d'un fichier xml.
    """
    with open(str(path_file), "r", encoding='utf-8') as file:
        content = file.readlines()
        content = "".join(content)
        soup = bs(content, "lxml")
        pg = soup.find('page')
        # Remplissage dico_page
        dict_page = creation_dict_bdd.FillDictPage(soup, pg, path_file)
        dict_page['cartons'] = []
        for carton_soup in soup.find_all('carton'):
            dict_carton = creation_dict_bdd.FillDictCarton(carton_soup)
            dict_page['cartons'].append(dict_carton)
        # Remplissage des dict ARTICLE
        dict_page['articles'] = {}
        for art_soup in soup.find_all('article'):
            dict_art = creation_dict_bdd.FillDictArticle(art_soup)
            l_dict_photos = []
            for bl_ph in art_soup.find_all('photo'):
                dict_photo = creation_dict_bdd.FillDictPhoto(bl_ph)
                l_dict_photos.append(dict_photo)
            dict_art['photo'] = l_dict_photos
            dict_page['articles'][dict_art['melodyId']] = dict_art
    return dict_page


def GetY(dict_page):
    def GetPositionArticle(dict_art):
        return [dict_art[x] for x in ['x', 'y', 'width', 'height']]
    for i, dict_art in enumerate(dict_page['articles'].values()):
        if i == 0:
            big_y = np.array([GetPositionArticle(dict_art)])
        else:
            y = np.array([GetPositionArticle(dict_art)])
            big_y = np.concatenate((big_y, y), axis=0)
    return big_y


def AddElementsToDictLayout(dict_layouts_small, dict_page, array_layout_page):
    id_page = dict_page['melodyId']
    id_layout = dict_page['pageTemplate']
    if id_layout == 'NoTemplate':
        print(f"The page to add has no template")
        return dict_layouts_small
    # We check if the layout is already in the base
    if id_layout in dict_layouts_small.keys():
        # We must check if the modules match
        layout_data = dict_layouts_small[id_layout]['array']
        args_c = [array_layout_page, layout_data]
        if module_montage.CompareTwoLayouts(*args_c):
            dict_layouts_small[id_layout]['id_pages'].append(id_page)
    # Otherwise we must fill all keys
    else:
        feat_carton = ['x', 'y', 'width', 'height', 'nbCol', 'nbPhoto']
        list_cartons = []
        for dict_carton in dict_page['cartons']:
            list_cartons.append(tuple(dict_carton[x] for x in feat_carton))
        dict_layouts_small[id_layout]['id_pages'] = [id_page]
        dict_layouts_small[id_layout]['nameTemplate'] = dict_page['nameTemplate']
        dict_layouts_small[id_layout]['cartons'] = list_cartons
        dict_layouts_small[id_layout]['array'] = [array_layout_page]
    return dict_layouts_small


def AddOnePageToDatabase(file_in,
                         file_out,
                         dict_pages,
                         dict_arts,
                         list_mdp,
                         dict_page_array,
                         dict_page_array_fast,
                         dict_layouts_small):
    # Extraction of the data
    dict_page = ExtrationDataOneFile(file_in)

    # Update some elements of the dict_page
    dict_page = creation_dict_bdd.Update(dict_page)
    id_page = dict_page['melodyId']

    if id_page in dict_pages.keys():
        print(f"There is already a page with that id in the data.")
        print(f"Under the name: {dict_pages[id_page]['pageName']}")
        database = [dict_pages, dict_arts, list_mdp, dict_page_array,
                    dict_page_array_fast, dict_layouts_small]
        return database

    # We add this dict to the global dict dict_pages
    dict_pages[id_page] = dict_page
    # We add the articles to the dict_arts
    for ida, dicta in dict_page['articles'].items():
        dict_arts[ida] = dicta
    # We add the array of the layout in dict_page_array(_fast)
    array_layout_page = GetY(dict_page)
    dict_page_array[id_page] = array_layout_page
    dict_page_array_fast[len(array_layout_page)][id_page] = array_layout_page
    # We add the info of the layout used for the page
    dict_layouts_small = AddElementsToDictLayout(dict_layouts_small,
                                                 dict_page,
                                                 array_layout_page)
    # We add the info about the layout in the important object list_mdp
    # Check if there is already a similar layout in the data
    corres = False
    for i, (nb, array_d, list_ids) in enumerate(list_mdp):
        if module_montage.CompareTwoLayouts(array_layout_page, array_d):
            new_list_ids = list_mdp[i][2] + [id_page]
            list_mdp[i] = (list_mdp[i][0] + 1, list_mdp[i][1], new_list_ids)
            corres = True
            break
    if corres is False:
        list_mdp.append((1, array_layout_page, [id_page]))
    database = [dict_pages, dict_arts, list_mdp, dict_page_array,
                dict_page_array_fast, dict_layouts_small]
    # Creation of the file_out with 'OK' to communicate to webservice
    module_montage.Creationxml("OK", file_out)
    return database
