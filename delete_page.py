#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 15:58:20 2021

@author: baptistehessel

I need to make a supposition about the kind of input I will receive. The most
natural should be an id of page.
Not so fast, in fact not to add another argument, the simplest thing is that I
also receive a file with the page to delete.
Same thing, but I just need to extract the id of the page from the xml.

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

"""
from bs4 import BeautifulSoup as bs
import module_montage


##############################################################################
#                                                                            #
#                       Script to delete page                                #
#                                                                            #
##############################################################################


def DeleteOnePageFromDatabase(path_file_in,
                              file_out,
                              dict_pages,
                              dict_arts,
                              list_mdp,
                              dict_page_array,
                              dict_page_array_fast,
                              dict_layouts):
    with open(str(path_file_in), "r", encoding='utf-8') as file:
        content = file.readlines()
    content = "".join(content)
    soup = bs(content, "lxml")
    infos_page = soup.find('page')
    id_page_to_delete = float(infos_page.get('melodyid'))
    # We check whether the id of the page is in the data.
    if id_page_to_delete not in dict_pages.keys():
        print(f"The page to delete is not in the data.")
        database = [dict_pages, dict_arts, list_mdp, dict_page_array]
        database += [dict_page_array_fast, dict_layouts]
        return database
    # We start to delete the elements from the data
    dict_page_array.pop(id_page_to_delete, -1)
    dict_page_array_fast.pop(id_page_to_delete, -1)
    # We remove from the dict_layouts
    for id_layout, dict_lay in dict_layouts.items():
        if id_page_to_delete in dict_lay['id_pages']:
            old_list_ids = dict_lay['id_pages']
            old_list_ids.remove(id_page_to_delete)
            if len(old_list_ids) > 0:
                new_list_ids = old_list_ids
                dict_layouts[id_layout]['id_pages'] = new_list_ids
            else:
                dict_layouts.pop(id_layout, -1)
            break
    # We remove from the list_mdp
    for i, (nbp, arrayp, list_ids) in enumerate(list_mdp):
        if id_page_to_delete in list_ids:
            new_nbp = list_mdp[i][0] - 1
            if new_nbp > 0:
                list_ids.remove(id_page_to_delete)
                list_mdp[i] = (new_nbp, list_mdp[i][1], list_ids)
            else:
                list_mdp.pop(i)
            break
    # We find the ids of the articles and we delete them
    list_ids_articles = dict_pages[id_page_to_delete]['articles']
    for id_art in list_ids_articles:
        dict_arts.pop(id_art, -1)
    # Finally, we delete the page from the dict_pages
    dict_pages.pop(id_page_to_delete, -1)
    database = [dict_pages, dict_arts, list_mdp, dict_page_array]
    database += [dict_page_array_fast, dict_layouts]
    # Creation of the xml file_out to communicate with the webservice
    module_montage.Creationxml("OK", file_out)
    return database
