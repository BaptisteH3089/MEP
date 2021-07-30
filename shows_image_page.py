#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 14:24:21 2021

@author: baptistehessel

Script used to represent the layouts or the pages.
Mainly used to see if the ids of the layouts associated to the pages are
coherent.
You can have a basic representation of the page/layout by giving an id of
page/layout in the database.
It is also possible to use RepPageSlide() with the zonning of the articles if
you don't have the id.

"""
import matplotlib.pyplot as plt
from matplotlib import patches
import pickle

path_cm = '/Users/baptistehessel/Documents/DAJ/MEP/montageIA/data/CM/'

loading_files = False
if loading_files:
    # Loading dictionary with all the layouts and the pages that use them
    with open(path_cm + 'dict_layouts', 'rb') as f:
        dict_layouts = pickle.load(f)
    # Loading dictionary with all the pages
    with open(path_cm + 'dict_pages', 'rb') as f:
        dict_pages = pickle.load(f)
    # Loading dictionary with all the articles
    with open(path_cm + 'dict_arts', 'rb') as f:
        dict_arts = pickle.load(f)

    dict_nbmodules = {i: [] for i in range(2, 9)}
    for id_layout, dict_val in dict_layouts.items():
        nb_modules = len(dict_val['cartons'])
        try:
            dict_nbmodules[nb_modules].append(id_layout)
        except:
            print("Different number of modules", nb_modules)


# Possibilité de mettre du texte dans les figures, mais pas d'imgs
def RepPageSlide(zonning_arts,
                 l_id_art,
                 txt=False,
                 l_car_art=None,
                 title=None,
                 dico_colors=None,
                 dico_colors_edge=None):
    figure = plt.figure(figsize = (10, 15))
    plt.gcf().subplots_adjust(0, 0, 1, 1)
    axes = figure.add_subplot(111)
    axes.set_frame_on(False)
    axes.xaxis.set_visible(False)
    axes.yaxis.set_visible(False)
    colors = ['green', 'darkred', 'midnightblue', 'darkgoldenrod']
    colors += ['darkseagreen', 'mediumslateblue', 'crimson', 'b']
    colors += ['rosybrown', 'orangered', 'lime', 'y', 'g', 'tan', 'c']
    middle_imgs = []
    # 1. On place les articles
    for i, (dim_art, id_art) in enumerate(zip(zonning_arts, l_id_art)):
        coord_img = (dim_art[0]/310, 1 - (dim_art[1]/470 + dim_art[3]/470))
        larg, haut = dim_art[2]/310, dim_art[3]/470
        midx, midy = coord_img[0] + larg/2, coord_img[1] + haut/2
        middle_imgs.append((midx, midy))
        # Le premier param est le coin inf gauche puis largeur et hauteur
        param_rect = {'xy': coord_img,
                      'width': larg,
                      'height': haut,
                      'edgecolor': colors[i],
                      'facecolor': 'lightgrey',
                      'fill': True,
                      'hatch': '/',
                      'linestyle': 'dashed',
                      'linewidth': 3,
                      'zorder': 1}
        if dico_colors is not None:
            try:
                param_rect['facecolor'] = dico_colors[id_art]
                param_rect['edgecolor'] = dico_colors_edge[dico_colors[id_art]]
            except Exception as e:
                print("An error with the dico_colors in RepPageSlide l. 1150")
                print(e)
        axes.add_artist(patches.Rectangle(**param_rect))
    param_txt = {'horizontalalignment': 'center',
                 'verticalalignment': 'center',
                 'fontsize': 30,
                 'color': 'black'}
    if txt is True:
        for mid, id_art, car_art in zip(middle_imgs, l_id_art, l_car_art):
            param_txt['x'] = mid[0]
            param_txt['y'] = mid[1]
            param_txt['s'] = str(id_art) + '\n'
            param_txt['s'] += "{:.0f} - {:.0f}".format(car_art[0], car_art[1])
            figure.text(**param_txt)
    if title is not None:
        param_txt['fontsize'] = 40
        param_txt['color'] = 'navy'
        param_txt['x'] = 0.5
        param_txt['y'] = 0.95
        param_txt['s'] = title
        figure.text(**param_txt)
    return "Page affichée"


def ShowLayout(id_layout, dict_layouts):
    # Creation of the object zonning_arts [(x, y, w, h), ...]
    zonning_arts = []
    list_for_rep = []
    for i, carton in enumerate(dict_layouts[id_layout]['cartons']):
        zonning_arts.append((carton[0], carton[1], carton[2], carton[3]))
        list_for_rep.append("carton " + str(i))
    title_rep = "Layout with " + str(len(list_for_rep)) + " modules."
    # Show the layout in the prompt
    RepPageSlide(zonning_arts, list_for_rep, title=title_rep)


def ShowPage(id_page, dict_pages):
    # Creation of the object zonning_arts [(x, y, w, h), ...]
    zonning_arts = []
    list_for_rep = []
    dict_arts_page = dict_pages[id_page]['articles']
    for i, (id_art, dict_art) in enumerate(dict_arts_page.items()):
        zonning = tuple(dict_art[x] for x in ['x', 'y', 'width', 'height'])
        zonning_arts.append(zonning)
        list_for_rep.append("article " + str(id_art))
    title_rep = "Page " + str(id_page)
    # Show the layout in the prompt
    RepPageSlide(zonning_arts, list_for_rep, title=title_rep)


show_layout = False
if show_layout:
    # The results are really strange. Most of these layouts aren't good.
    for key in dict_nbmodules.keys():
        for id_layout in dict_nbmodules[key]:
            ShowLayout(id_layout, dict_layouts)

show_pages = False
if show_pages:
    for id_layout in dict_nbmodules[2]:
        ShowLayout(id_layout, dict_layouts)
        for id_page in dict_layouts[id_layout]['id_pages']:
            ShowPage(id_page, dict_pages)
