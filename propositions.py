#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
from bs4 import BeautifulSoup
from sklearn.ensemble import RandomForestClassifier
from operator import itemgetter
import xml.etree.cElementTree as ET
import numpy as np
import methods
import zipfile
import pickle
import math
import time
import re
import os


# Classe pour écrire des exceptions personnalisées
class MyException(Exception):
    pass


def CleanBal(txt):
    """
    Enlève les balises xml <...> et leur contenu.
    """
    return re.sub(re.compile('<.*?>'), '', txt)


def ExtractDicoArtInput(file_path):
    """
    Anciennement : CreateListMDPInput(file_path) qui renvoyait dico identique.
    'file_path' correspond au chemin absolu d'un fichier xml qui va être parsé
    par cette fonction.
    PLus de clef 'nbCol'.
    Renvoie un dico avec comme clefs les quelques features voulues. Exemple :
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
                             'syn': ...}
    """
    with open(file_path, "r", encoding='utf-8') as file:
        try:
            content = file.readlines()
        except Exception as e:
            print(str(file))
        content = "".join(content)
        soup = BeautifulSoup(content, "lxml")
    art_soup = soup.find('article')
    dico_vector_input = {}
    img_soup = art_soup.find('photos')
    aire_img = 0
    if img_soup is not None:
        widths = img_soup.find_all('originalwidth')
        heights = img_soup.find_all('originalheight')
        for width_bal, height_bal in zip(widths, heights):
            width, height = width_bal.text, height_bal.text
            try:
                aire_img += int(width) * int(height) / 10000
            except Exception as e:
                print("Error with width and height: {}".format(e))
                print("width", width, "type(width)", type(width))
                print("height", height, "type(height)", type(height))
    try: sup_title = CleanBal(art_soup.find('suptitle').text)
    except: sup_title = ""
    try: sec_title = CleanBal(art_soup.find('secondarytitle').text)
    except: sec_title = ""
    try: sub_title = CleanBal(art_soup.find('subtitle').text)
    except: sub_title = ""
    try: title = CleanBal(art_soup.find('title').text)
    except: title = ""
    try: abstract = CleanBal(art_soup.find('abstract').text)
    except: abstract = ""
    try: syn = CleanBal(art_soup.find('synopsis').text)
    except: syn = ""
    try: exergue = CleanBal(art_soup.find('exergue').text)
    except: exergue = ""
    sup_title_indicatrice = 1 if len(sup_title) > 0 else 0
    sec_title_indicatrice = 1 if len(sec_title) > 0 else 0
    sub_title_indicatrice = 1 if len(sub_title) > 0 else 0
    title_indicatrice = 1 if len(title) > 0 else 0
    abstract_indicatrice = 1 if len(abstract) > 0 else 0
    syn_indicatrice = 1 if len(syn) > 0 else 0
    exergue_indicatrice = 1 if len(exergue) > 0 else 0
    # Données du carton. Plus fiable normalement.
    carton = soup.find('carton')
    options_soup_txt = soup.find('options').text
    options_soup = BeautifulSoup(options_soup_txt, "lxml")
    liste_blocks = []
    aire_imgs, dim_imgs = [], []
    aireTot = 0
    petittitre_indicator, quest_rep_indicator, intertitre_indicator = 0, 0, 0
    for bloc_soup in options_soup.find_all('bloc'):
        dico_block = {}
        try:
            kind = bloc_soup.get('kind')
            dico_block['kind'] = kind
        except:
            pass
        dico_block['label'] = bloc_soup.get('label')
        # Il faut créer des variables pour avoir la présence du Titre,
        # intertitre, question, réponse, exergue
        str_height = bloc_soup.get('originalheight')
        str_width = bloc_soup.get('originalwidth')
        str_left = bloc_soup.get('originalleft')
        str_top = bloc_soup.get('originaltop')
        dico_block['originalheight'] = int(float(str_height))
        dico_block['originalwidth'] = int(float(str_width))
        try: dico_block['originalleft'] = int(float(str_left))
        except: dico_block['originalleft'] = -1
        try: dico_block['originaltop'] = int(float(str_top))
        except: dico_block['originaltop'] = -1
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
    dico_vector_input['blocs'] = liste_blocks
    dico_vector_input['aireTotImgs'] = sum(aire_imgs)
    dico_vector_input['dimImgs'] = dim_imgs
    dico_vector_input['nbEmpImg'] = len(aire_imgs)
    dico_vector_input['aireTot'] = aireTot
    # Aussi dans l'ancienne version
    dico_vector_input['aireImg'] = aire_img
    try: dico_vector_input['melodyId'] = int(art_soup.find('id').text)
    except: dico_vector_input['melodyId'] = 8888
    nb_img_len = len(art_soup.find_all('photo'))
    # dico_vector_input['nbPhoto'] = nb_img_len
    try: dico_vector_input['nbPhoto'] = len(aire_imgs)
    except: dico_vector_input['nbPhoto'] = 0
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


def ExtractMDP(file_path):
    """
    Anciennement : CreateListMDPInput(file_path).
    Extrait les cartons d'un MDP contenus dans un fichier.
    Renvoie un dico de la forme :
        dico_mdp = {id_mdp: {id_carton_1: {'nbCol': xxx,
                                           'nbImg': xxx,
                                           'listeAireImgs': [xxx, xxx, ...],
                                           'height': xxx,
                                           'width': xxx,
                                           'x': xxx,
                                           'y': xxx,
                                           'aireTotImgs': xxx},
                             id_carton_2: {...}, ...}}
    """
    with open(file_path, "r", encoding='utf-8') as file:
        try:
            content = file.readlines()
        except Exception as e:
            print(str(file))
            raise e
        content = "".join(content)
    soup = BeautifulSoup(content, "lxml")
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
            try:
                kind = bloc_soup.get('kind')
                dico_block['kind'] = kind
            except:
                pass
            dico_block['label'] = bloc_soup.get('label')
            # Il faut créer des variables pour avoir la présence du Titre,
            # intertitre, question, réponse, exergue
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
    Extrait le nom de la rubrique dans une liste.
    Pour l'instant une seule rubrique.
    Renvoie une string.
    """
    with open(file_path, "r", encoding='utf-8') as file:
        try:
            content = file.readlines()
        except Exception as e:
            print(str(file))
        content = "".join(content)
        soup = BeautifulSoup(content, "lxml")
    print_category_soup = soup.find('printcategory')
    return print_category_soup.find('name').text


def GetXInput(liste_dico_vector_input, features):
    """
    Crée une matrice avec comme lignes les vecteurs associés aux articles et
    les colonnes correspondent aux variables dans features.
    Renvoie aussi le dico {id_art: vect_art} pour chaque article INPUT.
    big_x renvoyé : numpy array de dimensions (nb_arts_INPUT * nb_features)
    """
    dico_id_artx = {}
    for i, d_vect_input in enumerate(liste_dico_vector_input):
        args = [d_vect_input, features]
        id_art, vect_art = methods.GetVectorArticleInput(*args)
        dico_id_artx[id_art] = vect_art
        # Initialisation de la matrice big_x avec le 1er article
        if i == 0:
            big_x = np.array(vect_art, ndmin=2)
        # Ajout d'un vecteur article à la fin de big_x
        else:
            x = np.array(vect_art, ndmin=2)
            big_x = np.concatenate((big_x, x))
    return big_x, dico_id_artx


def TrouveMDPSim(liste_tuple_mdp, mdp_input):
    """
    Identification MDP similaire dans la liste l_model_new
    liste_tuple_mdp : l_model_new avec la sélection sur nb d'articles
    Renvoie la liste triée de tuples [(nbpages, mdp, ids), ...]
    Si aucune correspondance trouvée, cette fonction raise une exception.
    UPDATE 02/06 : Comme l'ordre est important on parcourt les tuples
    """
    n = len(mdp_input)
    mdp_trouve = []
    for nb_pages, mdp, liste_ids in liste_tuple_mdp:
        nb_correspondances = 0
        for i in range(n):
            for j in range(n):
                try:
                    if np.allclose(mdp[j], mdp_input[i], atol=10):
                        nb_correspondances += 1
                except Exception as e:
                    str_exc = (f"An exception occurs with np.allclose: {e}\n"
                               f"mdp: {mdp}\n"
                               f"j: {j}, i: {i}"
                               f"mdp_input: {mdp_input}")
                    print(str_exc)
        if nb_correspondances == n:
            mdp_trouve.append((nb_pages, mdp, liste_ids))
    if len(mdp_trouve) == 0:
        stc_exc = "No correspondance found for that MDP:\n{}\n in the BDD."
        raise MyException(stc_exc.format(mdp_input))
    mdp_trouve.sort(key=itemgetter(0), reverse=True)
    # print('The mdp found in TrouveMDPSim: \n{}\n'.format(mdp_trouve[0][1]))
    return mdp_trouve


def ExtrationDataInput(rep_data):
    """
    Extraction des données des xml INPUT.
    Archive de la forme :
        name_archive/articles
                    /pageTemplate
                    /printCategory
    """
    list_articles = []
    # Extraction des articles
    rep_articles = rep_data + '/' + 'articles'
    for file_path in Path(rep_articles).glob('./**/*'):
        if file_path.suffix == '.xml':
            dict_vector_input = ExtractDicoArtInput(file_path)
            list_articles.append(dict_vector_input)
    # Extraction du MDP
    dico_mdps = {}
    rep_mdp = rep_data + '/' + 'pageTemplate'
    for file_path in Path(rep_mdp).glob('./**/*'):
        if file_path.suffix == '.xml':
            dico_one_mdp = ExtractMDP(file_path)
            id_mdp = list(dico_one_mdp.keys())[0]
            # VOIR POUR AUTRE SOLUTION SI PLUSIEURS MDP
            dico_mdps[id_mdp] = dico_one_mdp[id_mdp]
    # Extraction de la/des rubrique(s)
    list_rub = []
    rep_rub = rep_data + '/' + 'printCategory'
    for file_path in Path(rep_rub).glob('./**/*'):
        if file_path.suffix == '.xml':
            list_rub.append(ExtractCatName(file_path))
    return list_articles, dico_mdps, list_rub


def SelectionModelesPages(l_model, nb_art, nb_min_pages):
    """
    Sélectionne parmi tous les MDP dans la BDD, ceux qui ont "nb_art" articles
    et qui sont utilisés pour au moins "nb_min_pages" pages
    """
    f = lambda x: (x[1].shape[0] == nb_art) & (x[0] >= nb_min_pages)
    return list(filter(f, l_model))


def CreateXmlPageProposals(liste_page_created, file_out):
    """
    Liste_page_created est de la forme : 
        [{MelodyId: X, printCategoryName: X, emp1: X, emp2: X, emp3: X}, ...]
    Crée un fichier xml à l'emplacement "file_out" avec les propositions de
    page. L'architecture du xml est :
        <PagesLayout>
            <PageLayout MelodyId="..." printCategoryName="...">
                <article x="..." y="..." MelodyId="..." />
                <article x="..." y="..." MelodyId="..." />
            </PageLayout>
                ...
            <PageLayout MelodyId="..." printCategoryName="...">
                <article x="..." y="..." MelodyId="..." />
                ...
            </PageLayout>
            ...
        </PagesLayout>
    # UPDATE 01/06/21, il faut ajouter l'élément CartonId dans article'
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
    dict_mlv_filtered = {clef_1: [(score, id_art), ...],
                         clef_2: ...,
                         ...}
    Création d'un dico avec id_art: score
    On enlève le score du dico_filtered
    On utilise la fonction pour faire le produit de listes
    On calcule le score pour chaque ventilation
    On renvoie les 3 ventilations avec les meilleurs scores.
    Returns an object of the form:
        liste_possib_score = [(score, (id_1, id_2, id_3), ...), ...]
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
    Crée une liste de la forme
    [
    {'MelodyId': '25782', 'printCategoryName': 'INFOS',
     ('15', '46'): 695558, ('15', '292'): 695641, ('204', '191'): 694396}, ...
    ]
    avec les pages proposées
    UPDATE 01/06 ajout élément id_carton aux résultat
    Renvoie objet de la forme [{'MelodyId':...,
    ('15', '46'): (id_art, id_carton)}]
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
    ind_features = ['nbSign', 'aireTot', 'aireImg']
    vents_uniq = [(88176, (32234, 28797)), ...]
    Le tri se fait sur le score et les différences entre les articles
    pondérées par les aires des emplcaments
    Il faut renvoyer un objet du type :
        [(score, (id1, id2)), (score, (id1, id2)), (score, (id1, id2))]
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
        """
        for i, vect_page_1 in enumerate(triplet_vect):
            for vect_page_2 in triplet_vect[i + 1:]:
                try:
                    diff_vect = vect_page_1 - vect_page_2
                except Exception as e:
                    str_exc = "Error with diff vector: {}.\n".format(e)
                    str_exc += "vect_page_1: {}.\n".format(vect_page_1)
                    str_exc += "vect_page_2: {}.".format(vect_page_2)
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
        """
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


def ExtractAndComputeProposals(dico_bdd,
                               liste_mdp_bdd,
                               path_data_input,
                               file_out,
                               verbose=2):
    """

    Parameters
    ----------
    dico_bdd : dictionary
        DESCRIPTION.
    liste_mdp_bdd : list of tuples
        DESCRIPTION.
    path_data_input : str
        Corresponds to a directory with 2 or 3 folders which contain the xml
        files with the articles, the rubric and the layout.
    file_out : str
        The absolute path of the file that will be created by this function
        with the results obtained.

    Returns
    -------
    str
        Indicates that everything went well.

    Steps of the function:
        - Creates the lists of dictionaries with the articles and the layout.
        - Determine the pages that can be made with these articles and that 
        layout.
        - Create an xml file with the results. The output file associates ids
        of articles with ids of modules.

    """
    # The list of features of the vectors article.
    # The compulsory features are: aireImg and aireTot.
    list_features = ['nbSign', 'nbBlock', 'abstract', 'syn']
    list_features += ['exergue', 'title', 'secTitle', 'supTitle']
    list_features += ['subTitle', 'nbPhoto', 'aireImg', 'aireTot']
    # Indexes of nbSign, nbPhoto, aireImg for the function FiltreArticles.
    list_to_index = ['nbSign', 'aireTot', 'aireImg']
    ind_features = [list_features.index(x) for x in list_to_index]
    print("The path_data_input: {}".format(path_data_input))
    # On met les données extraites dans des listes de dico
    list_arts_INPUT, mdp_INPUT, rub_INPUT = ExtrationDataInput(path_data_input)
    # Affichage MDP input
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
    # Attribution sommaire (pour l'instant) d'une rubrique aux articles
    list_rub = rub_INPUT
    dico_arts_rub_INPUT = {rub: [] for rub in list_rub}
    for dict_art in list_arts_INPUT:
        rub_picked = list_rub[0]
        # Si la rub est déjà présente, on ajoute l'art à la liste
        # There is only one rubric in fact
        dico_arts_rub_INPUT[rub_picked].append(dict_art)
    # Vérification du dico dico_arts_rub_INPUT
    if verbose > 2:
        print("{:-^75}".format("dico_arts_rub_INPUT"))
        for rub, liste_art in dico_arts_rub_INPUT.items():
            print(rub)
            for art in liste_art:
                list_prt = [elt for elt in art.items() if elt[0] is not 'blocs']
                print("Les blocs : {}".format(art['blocs']))
                print("Autre caract : \n{}\n".format(list_prt))
        print("{:-^75}".format("END dico_arts_rub_INPUT"))
    # Il faut transformer les liste_art en matrice X_INPUT et dico_id_artv
    dico_arts_rub_X = {}
    for rub, liste_art in dico_arts_rub_INPUT.items():
        big_x, dico_id_artv = GetXInput(liste_art, list_features)
        dico_arts_rub_X[rub] = (big_x, dico_id_artv)
    # L'obtention du dico_arts_rub_X est une étape important
    # Visualisation des résultats
    print("{:-^75}".format("dico_arts_rub_X"))
    for rub, (mat_x, dico_art) in dico_arts_rub_X.items():
        print("La rubrique : {}".format(rub))
        nb_ft = len(list_features)
        print("The features: " + ("{} " * nb_ft).format(*list_features))
        str_print = "The matrix X associated to all articles INPUT:\n{}\n"
        print(str_print.format(mat_x))
        print("Le dico de l'article : ")
        for clef, val in dico_art.items():
            print("{} {}".format(clef, val))
    print("{:-^75}".format("END dico_arts_rub_X"))
    # Pour chaque MDP, on regarde si on trouve une correspondance dans la BDD
    # Si pas de correspondance, on passe.
    # Sinon, on applique la méthode naïve pour filtrer les articles insérables
    # aux différents emplacements du MDP. Puis on applique la méthode MLV
    # pour avoir les meilleurs propositions.
    dict_global_results = {}
    for id_mdp, dict_mdp_input in mdp_INPUT.items():
        dict_global_results[id_mdp] = {}
        # Transformation du MDP en matrice
        list_keys = ['x', 'y', 'width', 'height']
        list_cartons = [[int(dict_carton[x]) for x in list_keys]
                        for dict_carton in dict_mdp_input.values()]
        list_feat = ['nbCol', 'nbImg', 'aireTotImgs', 'aireTot']
        X_nbImg = [[int(dic_carton[x]) for x in list_feat]
                   for dic_carton in dict_mdp_input.values()]
        mdp_loop = np.array(list_cartons)
        # Recherche d'une correspondance dans la BDD
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
                mdp_ref = res_found[0][1]
                liste_ids_found = res_found[0][2]
                correspondance = True
            except Exception as e:
                mdp_ref = mdp_loop
                correspondance = False
                str_prt = "An exception occurs while searching"
                print(str_prt + " for correspdces: ", e)
        # On parcourt les articles INPUT classés par rubrique
        for rub, (X_input, dico_id_artv) in dico_arts_rub_X.items():
            # Utilisation de la méthode naïve pour filtrer obtenir les
            # possibilités de placements
            args_naive = [mdp_ref, X_nbImg, X_input, dico_id_artv]
            args_naive += [ind_features, dict_mdp_input, list_arts_INPUT]
            try:
                dico_pos_naive, vents_uniq = methods.MethodeNaive(*args_naive)
            except Exception as e:
                print("Error Method Naive: {}".format(e))
                continue
            # Verification of vents_uniq
            if len(vents_uniq) == 0:
                str_prt = "When we delete duplicates of the naive method, "
                str_prt += "there is nothing left."
                print(str_prt)
                continue
            # Affichage Results naive
            print("{:-^75}".format("NAIVE"))
            for emp, poss in dico_pos_naive.items():
                nice_emp = ["{:.0f}".format(x) for x in emp]
                nice_poss = ["{}".format(id_art) for id_art in poss]
                print("{:<35} {}".format(str(nice_emp), nice_poss))
            print("{:-^75}".format("END NAIVE"))
            # Case where we have data about the layout input
            # We use the Machine Learning Method
            if correspondance == True:
                print("{:-^75}".format("CORRESPONDANCE FOUND"))
                args_mlv = [dico_bdd, liste_ids_found, mdp_ref, X_input]
                args_mlv += [dico_id_artv, list_features]
                try:
                    dico_possi_mlv = methods.MethodeMLV(*args_mlv)
                except Exception as e:
                    print("Error with MethodeMLV: {}".format(e))
                    continue
                print("{:-^75}".format("DICO POSSI MLV"))
                for key, value in dico_possi_mlv.items():
                    nice_val = [(round(sc, 2), ida) for sc, ida in value]
                    print("{:<30} {:<35}".format(str(key), str(nice_val)))
                print("{:-^75}".format("END DICO POSSI MLV"))
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
                    dict_global_results[id_mdp][rub] = first_results
                    continue
                # Affichage des résultats
                print("{:-^75}".format("MLV FILTERED"))
                for emp, poss in dict_mlv_filtered.items():
                    nice_emp = ["{:.0f}".format(x) for x in emp]
                    nice_poss = ["{:.2f} {}".format(sc, i) for sc, i in poss]
                    print("{:<35} {}".format(str(nice_emp), nice_poss))
                # Génération des objets avec toutes les pages possibles,
                # s'il y a des articles pour chaque EMP.
                first_results = ProposalsWithScore(dict_mlv_filtered)
                dict_global_results[id_mdp][rub] = first_results
                print("The 3 results with the best score")
                for score, ids in first_results:
                    str_print = ""
                    for id_art in ids: str_print += "--{}"
                    print(("{:-<15.2f}" + str_print).format(score, *ids))
                print("{:-^75}".format("END MDP"))
            # Dans le cas où le MDP INPUT n'est pas dans la BDD
            else:
                print("{:-^75}".format("NO CORRESPONDANCE FOUND"))
                # On parcourt le(s) MDP INPUT, on applique la méthode naïve
                # sur les articles INPUT et ce MDP
                print("{:*^75}".format("VENTS UNIQUE"))
                for elt in vents_uniq: print(elt)
                print("{:-^75}".format("END VENTS UNIQUE"))
                args_sel = [vents_uniq, dico_id_artv, ind_features]
                first_results = SelectionProposalsNaive(*args_sel)
                print("{:*^75}".format("FIRST RES"))
                for elt in first_results: print(elt)
                print("{:*^75}".format("END FIRST RES"))
                dict_global_results[id_mdp][rub] = first_results
    list_created_pg = CreateListeProposalsPage(dict_global_results, mdp_INPUT)
    print("The list of page created: {}".format(list_created_pg))
    CreateXmlPageProposals(list_created_pg, file_out)
    return "Xml output created"
