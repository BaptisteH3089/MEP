#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Baptiste Hessel

Contains the main functions relative to the part with a layout input.

"""
from pathlib import Path
from bs4 import BeautifulSoup
from operator import itemgetter
import xml.etree.cElementTree as ET
import numpy as np
import re
import methods


# Classe pour écrire des exceptions personnalisées
class MyException(Exception):
    pass


def CleanBal(txt):
    """
    Removes the xml tags <...> and their content.

    Parameters
    ----------
    txt : str
        The text input.

    Returns
    -------
    str
        The text input without <tag>.

    """
    return re.sub(re.compile('<.*?>'), '', txt)


def OpeningArticle(file_path_article):
    """
    Opening of an article input and returns the tag article and the content.

    Parameters
    ----------
    file_path_article: str
        The absolute path to the file with an article.

    Raises
    ------
    MyException
        If there is an issue with the content.

    Returns
    -------
    art_soup: BeautifulSoup object
        All what's in the tag <article>.

    soup: BeautifulSoup object
        All the content of the file parsed with beautiful soup.

    """
    with open(file_path_article, "r", encoding='utf-8') as file:
        try:
            content = file.readlines()
        except Exception as e:
            str_exc = (f"Error in propositions:OpeningArticle \n"
                       f"An error while opening the file: {e} \n"
                       f"The path of the file: {str(file_path_article)}.")
            raise MyException(str_exc)

    content = "".join(content)
    soup = BeautifulSoup(content, "lxml")
    art_soup = soup.find('article')

    if art_soup is None:
        print("art_soup is None")
        str_exc = (f"propositions:ExtractDicoArtInput: No art. in that file."
                   f"file_path:\n{file_path_article}")
        raise MyException(str_exc)

    return art_soup, soup


def ExtractDicoArtInput(file_path):
    """
    Parsing of a xml file with an article.

    Parameters
    ----------
    file_path: str
        The absolute path to an xml file article.

    Raises
    ------
    MyException
        If the file doesn't have a tag 'article'.

    Returns
    -------
    dico_vector_input: dict
        dico_vector_input = {'aireImg': ...,
                             'melodyId': ...,
                             'nbPhoto': ...,
                             'nbBlock': ...,
                             'nbSign': ...,
                             'supTitle': ...,
                             'secTitle': ...,
                             'subTitle': ...,
                             'title': ...,
                             'abstract': ...,
                             'exergue': ...,
                             'syn': ...,
                             'label': ...,
                             'blocs': ...,
                             'aireTotImgs': ...,
                             'dimImgs': ...,
                             'nbEmpImg': ...,
                             'aireTot': ...}.
    """

    art_soup, soup = OpeningArticle(file_path)

    dico_vector_input = {}
    img_soup = art_soup.find('photos')
    aire_img = 0
    if img_soup is not None:
        widths = img_soup.find_all('originalwidth')
        heights = img_soup.find_all('originalheight')
        for width_bal, height_bal in zip(widths, heights):
            width, height = width_bal.text, height_bal.text
            try:
                aire_img += int(width)*int(height) / 10000
            except Exception as e:
                print("Error with width and height: {}".format(e))
                print("width", width, "type(width)", type(width))
                print("height", height, "type(height)", type(height))

    try:
        sup_title = CleanBal(art_soup.find('suptitle').text)
    except:
        sup_title = ""
    try:
        sec_title = CleanBal(art_soup.find('secondarytitle').text)
    except:
        sec_title = ""
    try:
        sub_title = CleanBal(art_soup.find('subtitle').text)
    except:
        sub_title = ""
    try:
        title = CleanBal(art_soup.find('title').text)
    except:
        title = ""
    try:
        abstract = CleanBal(art_soup.find('abstract').text)
    except:
        abstract = ""
    try:
        syn = CleanBal(art_soup.find('synopsis').text)
    except:
        syn = ""
    try:
        exergue = CleanBal(art_soup.find('exergue').text)
    except:
        exergue = ""

    # We use indicators functions.
    sup_title_indicatrice = 1 if len(sup_title) > 0 else 0
    sec_title_indicatrice = 1 if len(sec_title) > 0 else 0
    sub_title_indicatrice = 1 if len(sub_title) > 0 else 0
    title_indicatrice = 1 if len(title) > 0 else 0
    abstract_indicatrice = 1 if len(abstract) > 0 else 0
    syn_indicatrice = 1 if len(syn) > 0 else 0
    exergue_indicatrice = 1 if len(exergue) > 0 else 0

    # Data of the module.
    options_soup_txt = soup.find('options').text
    options_soup = BeautifulSoup(options_soup_txt, "lxml")
    liste_blocks = []
    aire_imgs = []
    dim_imgs = []
    aireTot = 0
    petittitre_indicator = 0
    quest_rep_indicator = 0
    intertitre_indicator = 0

    for bloc_soup in options_soup.find_all('bloc'):
        dico_block = {}
        dico_block['label'] = bloc_soup.get('label')
        str_height = bloc_soup.get('originalheight')
        str_width = bloc_soup.get('originalwidth')
        str_left = bloc_soup.get('originalleft')
        str_top = bloc_soup.get('originaltop')
        dico_block['originalheight'] = int(float(str_height))
        dico_block['originalwidth'] = int(float(str_width))
        try:
            dico_block['originalleft'] = int(float(str_left))
        except:
            dico_block['originalleft'] = -1
        try:
            dico_block['originaltop'] = int(float(str_top))
        except:
            dico_block['originaltop'] = -1

        aireTot += dico_block['originalheight'] * dico_block['originalwidth']

        if 'PHOTO' in dico_block['label']:
            height_img = dico_block['originalheight']
            width_img = dico_block['originalwidth']
            dim_imgs.append((height_img, width_img))
            aire_img = height_img * width_img
            aire_imgs.append(aire_img)
        elif 'PETITTITRE' in dico_block['label']:
            petittitre_indicator = 1
        elif 'INTERTITRE' in dico_block['label']:
            intertitre_indicator = 1
        elif 'QUESTION' in dico_block['label']:
            quest_rep_indicator = 1
        elif 'REPONSE' in dico_block['label']:
            quest_rep_indicator = 1

        liste_blocks.append(dico_block)

    dico_vector_input['aireImg'] = aire_img
    dico_vector_input['melodyId'] = int(art_soup.find('id').text)
    dico_vector_input['nbPhoto'] = len(aire_imgs)
    dico_vector_input['blocs'] = liste_blocks
    dico_vector_input['aireTotImgs'] = sum(aire_imgs)
    dico_vector_input['dimImgs'] = dim_imgs
    dico_vector_input['nbEmpImg'] = len(aire_imgs)
    dico_vector_input['aireTot'] = aireTot
    text_raw_soup = art_soup.find('text_raw')
    dico_vector_input['nbBlock'] = len(text_raw_soup.find_all('p'))
    dico_vector_input['nbSign'] = len(art_soup.find('text').text)
    dico_vector_input['supTitle'] = sup_title_indicatrice
    dico_vector_input['secTitle'] = sec_title_indicatrice
    dico_vector_input['subTitle'] = sub_title_indicatrice
    dico_vector_input['title'] = title_indicatrice
    dico_vector_input['abstract'] = abstract_indicatrice
    dico_vector_input['exergue'] = exergue_indicatrice
    dico_vector_input['syn'] = syn_indicatrice
    dico_vector_input['petittitre'] = petittitre_indicator
    dico_vector_input['intertitre'] = intertitre_indicator
    dico_vector_input['quest_rep'] = quest_rep_indicator

    return dico_vector_input


def OpeningMDP(file_path_layout):
    """
    Opening of the xml file with the info about the layout.

    Parameters
    ----------
    file_path_layout : str
        The absolute path to the file with the layout.

    Raises
    ------
    MyException
        If we can't open the file.

    Returns
    -------
    soup : Beautifulsoup object
        All the parsed content of the file.

    """
    with open(file_path_layout, "r", encoding='utf-8') as file:
        try:
            content = file.readlines()
        except Exception as e:
            str_exc = (f"Error in propositions:ExtractMDP \n"
                       f"An error while opening the file: {e} \n"
                       f"The path of the file: {str(file_path_layout)}.")
            raise MyException(str_exc)
    content = "".join(content)
    soup = BeautifulSoup(content, "lxml")
    return soup


def ExtractMDP(file_path):
    """
    Extracts the info in the xml file with the layout and put the info in a
    dict.

    Parameters
    ----------
    file_path: str
        The absolute path to the xml file.

    Returns
    -------
    dico_mdp: dict
        dico_mdp = {id_mdp: {id_carton_1: {'nbCol': ...,
                                           'nbImg': ...,
                                           'listeAireImgs': [...],
                                           'height': ...,
                                           'width': ...,
                                           'x': ...,
                                           'y': ...,
                                           'aireTotImgs': ...},
                             id_carton_2: {...}, ...}}.
    """
    soup = OpeningMDP(file_path)
    cartons_soup = soup.find_all('carton')
    page_template_soup = soup.find('pagetemplate')
    id_mdp = int(page_template_soup.find('id').text)
    dico_mdp = {id_mdp: {}}

    for carton_soup in cartons_soup:
        options_soup_txt = soup.find('options').text
        options_soup = BeautifulSoup(options_soup_txt, "lxml")
        liste_blocks = []
        aire_imgs, dim_imgs = [], []
        aire_tot = 0

        for bloc_soup in options_soup.find_all('bloc'):
            dico_block = {}
            dico_block['label'] = bloc_soup.get('label')
            str_height = bloc_soup.get('originalheight')
            str_width = bloc_soup.get('originalwidth')
            str_left = bloc_soup.get('originalleft')
            str_top = bloc_soup.get('originaltop')
            height_bloc = int(float(str_height))
            width_bloc = int(float(str_width))
            dico_block['originalheight'] = height_bloc
            dico_block['originalwidth'] = width_bloc
            dico_block['originalleft'] = int(float(str_left))
            dico_block['originaltop'] = int(float(str_top))
            aire_tot += height_bloc * width_bloc
            if 'PHOTO' in dico_block['label']:
                height_img = dico_block['originalheight']
                width_img = dico_block['originalwidth']
                dim_imgs.append((height_img, width_img))
                aire_img = height_img * width_img
                aire_imgs.append(aire_img)
            liste_blocks.append(dico_block)

        id_carton = carton_soup.find('id').text
        nb_col = carton_soup.find('nbcol')
        mm_height = carton_soup.find('mmheight')
        mm_width = carton_soup.find('mmwidth')
        x_precision = carton_soup.find('x_precision')
        y_precision = carton_soup.find('y_precision')

        dico_mdp[id_mdp][id_carton] = {}
        dico_mdp[id_mdp][id_carton]['blocs'] = liste_blocks
        dico_mdp[id_mdp][id_carton]['x'] = int(float(x_precision.text))
        dico_mdp[id_mdp][id_carton]['y'] = int(float(y_precision.text))
        dico_mdp[id_mdp][id_carton]['width'] = int(mm_width.text)
        dico_mdp[id_mdp][id_carton]['height'] = int(mm_height.text)
        dico_mdp[id_mdp][id_carton]['nbCol'] = int(nb_col.text)
        dico_mdp[id_mdp][id_carton]['nbImg'] = len(aire_imgs)
        dico_mdp[id_mdp][id_carton]['aireTotImgs'] = sum(aire_imgs)
        dico_mdp[id_mdp][id_carton]['listeAireImgs'] = aire_imgs
        dico_mdp[id_mdp][id_carton]['aireTot'] = aire_tot

    return dico_mdp


def ExtractCatName(file_path):
    """
    Returns the printCategory (rubric) associated to the files input.

    Parameters
    ----------
    file_path: str
        The absolute path to the file with the rubric or printCategory.

    Raises
    ------
    MyException
        If there is an issue while opening the file.

    Returns
    -------
    str
        The printCategory.

    """
    with open(file_path, "r", encoding='utf-8') as file:
        try:
            content = file.readlines()
        except Exception as e:
            str_exc = (f"Error in propositions:ExtractCatName \n"
                       f"An error while opening the file: {e} \n"
                       f"The path of the file: {str(file_path)}.")
            raise MyException(str_exc)
        content = "".join(content)
        soup = BeautifulSoup(content, "lxml")
    print_category_soup = soup.find('printcategory')
    return print_category_soup.find('name').text


def GetXInput(liste_dico_vector_input, list_features):
    """
    Creates a matrix with the articles input. A line of the matrix corresponds
    to an article and a column corresponds to one feature.

    Parameters
    ----------
    liste_dico_vector_input: list of dicts
        DESCRIPTION.

    list_features: list of strings
        The list with the features of each article.

    Returns
    -------
    big_x: numpy array
        Matrix with the articles input.
        dim(big_x) = (nb_arts_INPUT, nb_features).

    dico_id_artx: dict
        Dict with the articles input.
        dico_id_artx = {id_article: vect_article, ...}.

    """

    dico_id_artx = {}
    for i, d_vect_input in enumerate(liste_dico_vector_input):
        args = [d_vect_input, list_features]
        id_art, vect_art = methods.GetVectorArticleInput(*args)
        dico_id_artx[id_art] = vect_art
        # Initialisation of the matrix big_x with the 1er article
        if i == 0:
            big_x = np.array(vect_art, ndmin=2)
        # Addition vector article at the end of big_x
        else:
            x = np.array(vect_art, ndmin=2)
            big_x = np.concatenate((big_x, x))
    return big_x, dico_id_artx


def TrouveMDPSim(list_mdp_data, mdp_input):
    """
    Search for some similar layouts in the data.

    Parameters
    ----------
    list_mdp_data: list of tuples
        The data list of the form:
            list_mdp_data = [(nb, array_layout, list_ids), ...].

    mdp_input: numpy array
        The layout input.

    Raises
    ------
    MyException
        If there in an issue with np.allclose().
    MyException
        If there is no correspondance.

    Returns
    -------
    mdp_trouve: numpy array
        The similar layout found in the data.

    """

    n = len(mdp_input)
    mdp_trouve = []
    for nb_pages, mdp, liste_ids in list_mdp_data:
        nb_correspondances = 0
        for i in range(n):
            for j in range(n):
                try:
                    if np.allclose(mdp[j], mdp_input[i], atol=10):
                        nb_correspondances += 1
                except Exception as e:
                    str_exc = (f"An exception occurs with np.allclose: {e}\n"
                               f"mdp: {mdp}.\nj: {j}, i: {i} \n"
                               f"mdp_input:\n{mdp_input}")
                    raise MyException(str_exc)
        if nb_correspondances == n:
            mdp_trouve.append((nb_pages, mdp, liste_ids))
    if len(mdp_trouve) == 0:
        stc_exc = "No correspondance found for that MDP:\n{}\n in the BDD."
        raise MyException(stc_exc.format(mdp_input))
    mdp_trouve.sort(key=itemgetter(0), reverse=True)
    return mdp_trouve


def ExtractionDataInput(rep_data):
    """
    Extraction the data in the archive. The archive once dezipped must be of
    the form:
        name_archive/articles
                    /pageTemplate
                    /printCategory


    Parameters
    ----------
    rep_data: str
        The absolute path to the archive.

    Returns
    -------
    list_articles: list of dicts
        The list with the dictionary associated to the articles.

    dico_mdps: dict
        Dictionary with the layout input (there should be only one).
        dico_mdps = {id_layout: dict_layout, ...}

    rubric_out: str
        The rubric associated to the articles input.

    """

    list_articles = []
    # Extraction des articles
    rep_articles = rep_data + '/' + 'articles'
    print(f"For the articles, we search into: {rep_articles}")
    for file_path in Path(rep_articles).glob('*.xml'):
        try:
            dict_vector_input = ExtractDicoArtInput(file_path)
            list_articles.append(dict_vector_input)
        except Exception as e:
            str_exc = (f"Error in propositions:ExtractionDataInput \n"
                       f"{e}.\n The path of the file: {file_path}.")
            print(str_exc)

    # Extraction du MDP
    dico_mdps = {}
    rep_mdp = rep_data + '/' + 'pageTemplate'
    print(f"For the templates, we search into: {rep_mdp}")
    for file_path in Path(rep_mdp).glob('*.xml'):
        dico_one_mdp = ExtractMDP(file_path)
        id_mdp = list(dico_one_mdp.keys())[0]
        dico_mdps[id_mdp] = dico_one_mdp[id_mdp]

    # Extraction de la rubrique
    list_rub = []
    rep_rub = rep_data + '/' + 'printCategory'
    print(f"For the rubric, we search into: {rep_rub}")
    for file_path in Path(rep_rub).glob('*.xml'):
        list_rub.append(ExtractCatName(file_path))

    # Verification of the rubric
    if len(list_rub) > 1:
        print(f"len(list_rub): {len(list_rub)}. There should be only one")
        print(f"rubric. content: \n {list_rub}\n")
        rubric_out = list_rub[0]
        print(f"The rubric chosen: {rubric_out}")
    elif len(list_rub) == 0:
        rubric_out = "None"
    else:
        rubric_out = list_rub[0]

    return list_articles, dico_mdps, rubric_out


def SelectionModelesPages(list_mdp, nb_art, nb_min_pages):
    """
    Selects the layouts with "nb_arts" modules in our data and that are used
    by at least "nb_min_pages" pages.

    Parameters
    ----------
    list_mdp: list of tuples
        The usual list with the layouts.
        list_mdp = [(nb, array_layout, list_ids), ...]

    nb_art: int
        The number of modules in the layouts selected.

    nb_min_pages: int
        The min number of pages using each layout.

    Returns
    -------
    list
        The tuples_layout with "nb_arts" modules and used by at least
        "nb_min_pages" pages
        [(nb_pages, array_layout, list_ids), ...]

    """
    f = lambda x: (x[1].shape[0] == nb_art) & (x[0] >= nb_min_pages)
    return list(filter(f, list_mdp))


def CreateXmlPageProposals(liste_page_created, file_out):
    """
    Liste_page_created is of the form:
        [{MelodyId: X, printCategoryName: X, emp1: X, emp2: X, emp3: X}, ...]

    Creates a xml file in the repertory given by "file_out" with the proposals
    of pages. The architecture of the xml is:
        <PagesLayout>
            <PageLayout MelodyId="..." printCategoryName="...">
                <article x="..." y="..." MelodyId="..." CartonId="..."/>
                <article x="..." y="..." MelodyId="..." CartonId="..."/>
            </PageLayout>
                ...
            <PageLayout MelodyId="..." printCategoryName="...">
                <article x="..." y="..." MelodyId="..." CartonId="..."/>
                ...
            </PageLayout>
            ...
        </PagesLayout>

    Parameters
    ----------
    liste_page_created: list of dicts
        liste_page_created = [{MelodyId: X,
                               printCategoryName: X,
                               emp1: X,
                               emp2: X,
                               emp3: X,
                               ...}, ...]

    file_out: str
        The repertory where we write the file.

    Returns
    -------
    bool
        Just True if everything went well.

    """
    PagesLayout = ET.Element("PagesLayout")
    for i, created_page in enumerate(liste_page_created):
        PageLayout = ET.SubElement(PagesLayout, "PageLayout", name=str(i))
        for key, value in created_page.items():
            if key in ['printCategoryName', 'MelodyId']:
                PageLayout.set(str(key), str(value))
            else:
                Article = ET.SubElement(PageLayout, "article")
                Article.set("x", str(key[0]))
                Article.set("y", str(key[1]))
                Article.set("MelodyId", str(value[0]))
                Article.set("CartonId", str(value[1]))

    tree = ET.ElementTree(PagesLayout)
    tree.write(file_out, encoding="UTF-8")

    return True


def ProposalsWithScore(dict_mlv_filtered):
    """
    Computes the score of each possibility. Returns the 3 pages with the best
    score.

    Parameters
    ----------
    dict_mlv_filtered : dict
        dict_mlv_filtered = {clef_1: [(score, id_art), ...],
                             clef_2: ...,
                             ...}.

    Raises
    ------
    MyException
        if the type of dict_mlv_filtered isn't right.

    Returns
    -------
    list of tuples
        liste_possib_score = [(score, (id_1, id_2, id_3), ...), ...].

    """
    if type(dict_mlv_filtered) is not dict:
        str_exc = "The argument 'dict_mlv_filtered' of the function "
        str_exc += "ProposalsWithScore should be a dict and not "
        str_exc += "a {}".format(type(dict_mlv_filtered))
        str_exc += "dict_mlv_filtered: {}".format(dict_mlv_filtered)
        raise MyException(str_exc)
    dict_id_score = {}
    for i, list_score_id in enumerate(dict_mlv_filtered.values()):
        for score, id_art in list_score_id:
            if id_art not in dict_id_score.keys():
                dict_id_score[id_art] = [(score, i)]
            else:
                dict_id_score[id_art].append((score, i))
    dict_emp_score = {}
    for emp, values in dict_mlv_filtered.items():
        dict_emp_score[emp] = [val[1] for val in values]
    # On constitue la liste avec toutes les possibilités de ventilations
    l_args = [emp_poss for emp_poss in dict_emp_score.values()]
    every_possib = methods.ProductList(*l_args)
    # Calcul du score, on le rajoute à chaque possib
    liste_possib_score = []
    for possib in every_possib:
        score = 0
        for i, art_possib in enumerate(possib):
            list_score_numemp = dict_id_score[art_possib]
            try:
                f = lambda x: x[1] == i
                score_emp_art = list(filter(f, list_score_numemp))[0][0]
                score += score_emp_art
            except Exception as e:
                args_fmt = [e, art_possib, dict_id_score[art_possib]]
                print("{}\n art_possib {}\n res du dico {}".format(*args_fmt))
        liste_possib_score.append((score, possib))
    liste_possib_score.sort(key=itemgetter(0), reverse=True)
    return liste_possib_score[:3]


def CreateListeProposalsPage(dict_global_results, mdp_INPUT):
    """
    Prepares the results for the xml output.

    Parameters
    ----------
    dict_global_results: dict of dicts
        dict_global_results = {id_layout: {rubric: list_possib_score, ...},
                               ...}
        with list_possib_score = [(score, [ida, ida, ...]), ...]

    mdp_INPUT: dict
        mdp_INPUT = {id_layout: dict_layout, ...}.

    Returns
    -------
    liste_page_created : list of dicts
            [{'MelodyId': '25782',
            'printCategoryName': 'INFOS',
            ('15', '46'): (id_art, id_carton),
            ('15', '292'): (id_art, id_carton),
            ('204', '191'): (id_art, id_carton)}, ...].

    """
    liste_page_created = []
    for id_mdp, dict_poss_each_rub in dict_global_results.items():
        if len(dict_poss_each_rub) > 0:
            for rubr, liste_poss_score in dict_poss_each_rub.items():
                for score, liste_art_one_poss in liste_poss_score:
                    dict_mdp = mdp_INPUT[id_mdp]
                    dict_one_res = {}
                    dict_one_res['MelodyId'] = id_mdp
                    dict_one_res['printCategoryName'] = rubr
                    args_zip = [liste_art_one_poss, dict_mdp.items()]
                    for id_art, (id_carton, dict_carton) in zip(*args_zip):
                        tuple_emp = (dict_carton['x'], dict_carton['y'])
                        dict_one_res[tuple_emp] = (id_art, id_carton)
                    liste_page_created.append(dict_one_res)
    return liste_page_created


def SelectionProposalsNaive(vents_uniq, dico_id_artv, ind_features):
    """
    The filtering is made on the score and on the differences between articles
    weighted by the areas of the articles.

    Parameters
    ----------
    vents_uniq: list of tuples
        vents_uniq = [(88176, (32234, 28797)), ...].

    dico_id_artv: dict
        dico_id_artv = {id_article: vect_article, ...}.

    ind_features: list of strings
        ['nbSign', 'aireTot', 'aireImg'].

    Raises
    ------
    MyException
        If error with the weights.

    Returns
    -------
    list of tuples
        [(score, (id1, id2)), (score, (id1, id2)), (score, (id1, id2))].

    """
    # D'abord, on commence par extraire toutes les propositions avec 3
    # ventilations
    # Utiliser qch du style ProductList
    list_triplets = [(vent1, vent2, vent3) for vent1 in vents_uniq
                     for vent2 in vents_uniq if vent2 != vent1
                     for vent3 in vents_uniq if vent3 != vent2]
    # Maintenant il faut ajouter le score total de pertinence et le score
    # d'éloignement
    try:
        poids = [dico_id_artv[ida][0][ind_features[1]]
                 for ida in vents_uniq[0][1]]
        print("Les poids: {}".format(poids))
        length_vect = len(dico_id_artv[list(dico_id_artv.keys())[0]][0])
        # IL FAUT LES NORMALISER
        norm_weights = []
        for i in range(len(poids)):
            norm_weights += [poids[i]] * length_vect
        norm_weights = norm_weights / sum(norm_weights)
        # print('The norm_weights obtained: {}'.format(norm_weights))
    except Exception as e:
        str_exc = "Error with the weights: {}\n".format(e)
        str_exc += "dico_id_artv: {}\n".format(dico_id_artv)
        str_exc += "ind_features: {}\n".format(ind_features)
        str_exc += "vents_uniq: {}".format(vents_uniq)
        raise MyException(str_exc)
    # ICI IL FAUT FAIRE UN CALCUL DE DISTANCE PROPRE
    # EXTRACTION DES VECTEURS ARTICLE
    # CONCATENATION VECTEUR PAGE
    list_triplets_vect = []
    for triplet in list_triplets:
        triplet_prop = []
        for vent in triplet:
            try:
                vect_arts = [dico_id_artv[ida] for ida in vent[1]]
                vect_page = np.concatenate(vect_arts, axis=1)
            except Exception as e:
                str_exc = "Error with concatenation: {}\n".format(e)
                str_exc += "dico_id_artv: {}".format(dico_id_artv)
                str_exc += "vent: {}".format(vent)
                raise MyException(str_exc)
            triplet_prop.append(vect_page)
        list_triplets_vect.append(triplet_prop)
    weighted_list_triplets = []
    for triplet, triplet_vect in zip(list_triplets, list_triplets_vect):
        score_total = sum([vent[0] for vent in triplet])
        # Calcul du score far
        # Here I should do the variance criterion instead
        # 1. Computation of the vector mean page
        # 2. Distance between each vector page and the mean vector page
        score_far = 0
        mean_vector_pg = np.mean(triplet_vect, axis=0)
        for vect_page in triplet_vect:
            try:
                diff_vect = vect_page - mean_vector_pg
            except Exception as e:
                str_exc = "Error with diff vector: {}.\n".format(e)
                str_exc += "vect_page_1: {}.\n".format(vect_page)
                str_exc += "vect_page_2: {}.".format(mean_vector_pg)
                raise MyException(str_exc)
            try:
                diff_vect = norm_weights * diff_vect
            except Exception as e:
                str_exc = "Error product poids * diff_vect: {}\n".format(e)
                str_exc += "poids: {}.".format(poids)
                str_exc += "diff_vect: {}.".format(diff_vect)
                raise MyException(str_exc)
            dist_pages = np.sqrt(np.dot(diff_vect, diff_vect.T))
            score_far += int(dist_pages)
        weighted_list_triplets.append((score_total, score_far, triplet))
    print("{:-^75}".format("WEIGHTED LIST TRIPLETS"))
    print("{:<15} {:^15} {:>25}".format("score total", "score far", "triplet"))
    for i, (sc_tot, sc_far, triplet) in enumerate(weighted_list_triplets):
        print("{:<15} {:^15} {:>25}".format(str(sc_tot), str(sc_far), str(triplet)))
        if i == 3: break
    print("{:-^75}".format("END WEIGHTED LIST TRIPLETS"))
    # triplet = [(88176, (32234, 28797)), (sc, (id, id)), (sc, (id, id))]
    f = lambda x: (x[0] - x[1] * 1500, x[2])
    one_score_triplet = [f(x) for x in weighted_list_triplets]
    one_score_triplet.sort(key=itemgetter(0))
    best_prop = one_score_triplet[0]
    print("What is returned by SelectionProposalsNaive : ")
    for elt in best_prop[1]: print(elt)
    return best_prop[1]


def ShowsXDictidartv(big_x, dico_id_artv, list_features):
    """

    Parameters
    ----------
    big_x: numpy array
        The matrix with all articles input.

    dico_id_artv: dict
        dico_id_artv = {id_art: vect_art, ...}.

    list_features: list of strings
        The features used to create the vectors.

    Returns
    -------
    None.

    """
    print("{:-^75}".format("big_x and dico_id_artv"))
    nb_ft = len(list_features)
    print("The features: " + ("{} " * nb_ft).format(*list_features))
    str_print = "The matrix X associated to all articles INPUT:\n{}\n"
    print(str_print.format(big_x))
    print("Le dico de l'article : ")
    for clef, val in dico_id_artv.items():
        print("{} {}".format(clef, val))
    print("{:-^75}".format("END big_x and dico_id_artv"))


def ShowsResultsNAIVE(dico_pos_naive):
    """
    Parameters
    ----------
    dico_pos_naive: dict
        dico_pos_naive = {emp: poss, ...}.

    Returns
    -------
    None.

    """
    print("{:-^75}".format("NAIVE"))
    for emp, poss in dico_pos_naive.items():
        nice_emp = ["{:.0f}".format(x) for x in emp]
        nice_poss = ["{}".format(id_art) for id_art in poss]
        print("{:<35} {}".format(str(nice_emp), nice_poss))
    print("{:-^75}".format("END NAIVE"))


def ShowsResultsMLV(dico_possi_mlv):
    """
    Parameters
    ----------
    dico_possi_mlv: dict
        dico_possi_mlv = {emp: [(score, ida), ...]}.

    Returns
    -------
    None.

    """
    print("{:-^75}".format("DICO POSSI MLV"))
    for key, value in dico_possi_mlv.items():
        nice_val = [(round(sc, 2), ida) for sc, ida in value]
        print("{:<30} {:<35}".format(str(key), str(nice_val)))
    print("{:-^75}".format("END DICO POSSI MLV"))


def ShowsResultsFilteringMLV(dict_mlv_filtered):
    """
    Parameters
    ----------
    dict_mlv_filtered: dict
        dict_mlv_filtered = {emp: poss, ...}.

    Returns
    -------
    None.

    """
    print("{:-^75}".format("MLV FILTERED"))
    for emp, poss in dict_mlv_filtered.items():
        nice_emp = ["{:.0f}".format(x) for x in emp]
        nice_poss = ["{:.2f} {}".format(sc, i) for sc, i in poss]
        print("{:<35} {}".format(str(nice_emp), nice_poss))
    print("{:-^75}".format("END MLV FILTERED"))


def Shows3bestResults(first_results):
    """
    Parameters
    ----------
    first_results: list of tuples
        first_results = [(score, ids), ...].

    Returns
    -------
    None.

    """
    print("The 3 results with the best score")
    for score, ids in first_results:
        str_print = ""
        for id_art in ids: str_print += "--{}"
        print(("{:-<15.2f}" + str_print).format(score, *ids))
    print("{:-^75}".format("END MDP"))


def ShowsWhatsInMDP(mdp_INPUT):
    """

    Parameters
    ----------
    mdp_INPUT: dict
        mdp_INPUT = {id_mdp: dict_mdp, ...}.

    Returns
    -------
    None.

    """
    print("{:-^75}".format("MDP INPUT"))
    for id_mdp, dict_mdp_input in mdp_INPUT.items():
        print("id mdp: {:<25}".format(id_mdp))
        for key, val in dict_mdp_input.items():
            print("id carton : {}".format(key))
            for dict_block in val['blocs']:
                print("block: ", dict_block)
            for key, value in val.items():
                if key != 'blocs':
                    print(f"{key}: {value}")
    print("{:-^75}".format("END-INPUT"))


def ExtractAndComputeProposals(dico_bdd,
                               liste_mdp_bdd,
                               path_data_input,
                               file_out,
                               list_features,
                               verbose=2):
    """

    Steps of the function:
        - Creates the lists of dictionaries with the articles and the layout.
        - Determine the pages that can be made with these articles and that
        layout.
        - Create an xml file with the results. The output file associates ids
        of articles with ids of modules.

    Parameters
    ----------
    dico_bdd: dictionary
        Usual dict with all the info about the pages.

    liste_mdp_bdd: list of tuples
        liste_mdp_bdd = [(nb, array, list_ids), ...].

    path_data_input: str
        Corresponds to a directory with 2 or 3 folders which contain the xml
        files with the articles, the rubric and the layout.

    file_out: str
        The absolute path of the file that will be created by this function
        with the results obtained.

    list_features: list of strings
        The list with the features used to create the vectors article/page.

    verbose: int
        Whether to print info on what is going on. (default=2)

    Returns
    -------
    str
        Indicates that everything went well.

    """

    # Indexes of nbSign, nbPhoto, aireImg for the function FiltreArticles.
    list_to_index = ['nbSign', 'aireTot', 'aireImg']
    ind_features = [list_features.index(x) for x in list_to_index]
    print("The path_data_input: {}".format(path_data_input))

    # mdp_INPUT: dict (id_mdp: dict_mdp), rub_INPUT: list of strings
    list_arts_INPUT, mdp_INPUT, rub_INPUT = ExtractionDataInput(path_data_input)

    # Shows how the object mdp_INPUT looks like
    ShowsWhatsInMDP(mdp_INPUT)

    big_x, dico_id_artv = GetXInput(list_arts_INPUT, list_features)
    # L'obtention du dico_arts_rub_X est une étape important
    # Visualisation des résultats
    if verbose > 1:
        ShowsXDictidartv(big_x, dico_id_artv, list_features)

    dict_global_results = {}
    for id_mdp, dict_mdp_input in mdp_INPUT.items():
        dict_global_results[id_mdp] = {}

        # Transformation MDP into matrix
        list_keys = ['x', 'y', 'width', 'height']
        list_cartons = [[int(dict_carton[x]) for x in list_keys]
                        for dict_carton in dict_mdp_input.values()]
        list_feat = ['nbCol', 'nbImg', 'aireTotImgs', 'aireTot']
        X_nbImg = [[int(dic_carton[x]) for x in list_feat]
                   for dic_carton in dict_mdp_input.values()]
        mdp_loop = np.array(list_cartons)

        # Search for a correspondance of the MDP INPUT in the data.
        nb_art = mdp_loop.shape[0]
        min_nb_art = 15
        args_select_model = [liste_mdp_bdd, nb_art, min_nb_art]
        liste_tuple_mdp = SelectionModelesPages(*args_select_model)
        if liste_tuple_mdp == 0:
            str_info = 'Less than {} pages with '.format(min_nb_art)
            str_info += '{} arts in the database'.format(nb_art)
            correspondance = False
            print(str_info)
        else:
            try:
                res_found = TrouveMDPSim(liste_tuple_mdp, mdp_loop)
                mdp_ref, liste_ids_found = res_found[0][1], res_found[0][2]
                correspondance = True
            except Exception as e:
                mdp_ref = mdp_loop
                correspondance = False
                str_prt = "An exception occurs while searching"
                print(str_prt + " for correspdces: ", e)

        # Method NAIVE
        X_input = big_x
        args_naive = [mdp_ref, X_nbImg, X_input, dico_id_artv]
        args_naive += [ind_features, dict_mdp_input, list_arts_INPUT]
        try:
            dico_pos_naive, vents_uniq = methods.MethodeNaive(*args_naive)
        except Exception as e:
            print("\nError Method Naive: {}".format(e))
            print("All the arguments of methods.MethodeNaive: ")
            print(f"mdp_ref: \n{mdp_ref}\n")
            print(f"X_nbImg: \n{X_nbImg}\n")
            print(f"X_input: \n{X_input}\n")
            print(f"dico_id_artv: \n{dico_id_artv}\n")
            print(f"ind_features: \n{ind_features}\n")
            print(f"dict_mdp_input: \n{dict_mdp_input}\n")
            print(f"list_arts_INPUT: \n{list_arts_INPUT}\n\n\n")
            print("{:-^80}".format("END ARGUMENTS METHOD NAIVE"))
            continue

        # Verification of vents_uniq
        if len(vents_uniq) == 0:
            str_prt = "When we delete duplicates of the naive method, "
            str_prt += "there is nothing left."
            print(str_prt)
            continue

        # Affichage Results naive
        if verbose > 1:
            ShowsResultsNAIVE(dico_pos_naive)

        # Case where we have data about the layout input
        if correspondance == True:
            # Method MLV
            print("{:-^75}".format("CORRESPONDANCE FOUND"))
            args_mlv = [dico_bdd, liste_ids_found, mdp_ref, X_input]
            args_mlv += [dico_id_artv, list_features]
            try:
                dico_possi_mlv = methods.MethodeMLV(*args_mlv)
            except Exception as e:
                print("Error with MethodeMLV: {}".format(e))
                continue

            # Print the possibilities of the mthod MLV
            if verbose > 1:
                ShowsResultsMLV(dico_possi_mlv)

            # We use the dict result of the method NAIVE to filter results
            args_filt = [dico_pos_naive, dico_possi_mlv]
            try:
                dict_mlv_filtered = methods.ResultsFiltered(*args_filt)
            except Exception as e:
                print("Error with ResultsFiltered: {}".format(e))
                # Case where the mlv method didn't find anything, but the
                # naive method did find some possibilities.
                args_sel = [vents_uniq, dico_id_artv, ind_features]
                first_results = SelectionProposalsNaive(*args_sel)
                print("{:*^75}".format("NAIVE BIS"))
                for elt in first_results: print(elt)
                print("{:*^75}".format("END NAIVE BIS"))
                dict_global_results[id_mdp][rub_INPUT] = first_results
                continue

            # Shows results of the filtering
            if verbose > 1:
                ShowsResultsFilteringMLV(dict_mlv_filtered)

            # Génération des objets avec toutes les pages possibles,
            # s'il y a des articles pour chaque EMP.
            first_results = ProposalsWithScore(dict_mlv_filtered)
            dict_global_results[id_mdp][rub_INPUT] = first_results

            # Print the 3 best results
            if verbose > 1:
                Shows3bestResults(first_results)

        # Dans le cas où le MDP INPUT n'est pas dans la BDD
        else:
            print("{:-^75}".format("NO CORRESPONDANCE FOUND"))
            # On parcourt le(s) MDP INPUT, on applique la méthode naïve
            # sur les articles INPUT et ce MDP
            print("{:*^75}".format("VENTS UNIQUE"))
            for elt in vents_uniq:
                print(elt)
            print("{:-^75}".format("END VENTS UNIQUE"))

            args_sel = [vents_uniq, dico_id_artv, ind_features]
            first_results = SelectionProposalsNaive(*args_sel)
            print("{:*^75}".format("FIRST RES"))
            for elt in first_results:
                print(elt)
            print("{:*^75}".format("END FIRST RES"))
            dict_global_results[id_mdp][rub_INPUT] = first_results

    list_created_pg = CreateListeProposalsPage(dict_global_results, mdp_INPUT)
    print("The list of page created: {}".format(list_created_pg))
    # Writing of the output file with the results
    CreateXmlPageProposals(list_created_pg, file_out)

    return "Xml output created"
