#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
from bs4 import BeautifulSoup
from sklearn.ensemble import RandomForestClassifier
from operator import itemgetter
import xml.etree.cElementTree as ET
import numpy as np
import useful_methods
import pickle, zipfile, math, time, re, os
from logging.handlers import RotatingFileHandler
import logging


# Classe pour écrire des exceptions personnalisées
class MyException(Exception):
    pass


# LOGGER. Initialisation du Logger
path_log = '/Users/baptistehessel/Documents/DAJ/MEP/montageIA/log/montage_ia.log'
str_fmt = '%(asctime)s :: %(filename)s :: %(levelname)s :: %(message)s'
logger = logging.getLogger('my_logger')
formatter = logging.Formatter(str_fmt)
handler = RotatingFileHandler(path_log, maxBytes=30000, backupCount=3)
logger.setLevel(logging.DEBUG)
handler.setFormatter(formatter)
logger.addHandler(handler)

# Les couleurs qui serviront à afficher les articles dans les MDP avec la
# fonction RepPageSlides
colors = ['green', 'darkred', 'blueviolet', 'dodgerblue', 'teal']
colors += ['darkorange', 'mediumslateblue', 'crimson', 'deeppink']
colors += ['brown', 'salmon', 'peru', 'chartreuse', 'deepskyblue']
dico_colors_edge = {'green': 'darkgreen',
                    'darkred': 'indianred',
                    'midnightblue': 'mediumblue',
                    'dodgerblue': 'navy',
                    'teal': 'lightseagreen',
                    'darkorange': 'burlywood',
                    'mediumslateblue': 'darkslateblue',
                    'crimson': 'palevioletred',
                    'deeppink': 'hotpink',
                    'blueviolet': 'navy',
                    'brown': 'navy',
                    'salmon': 'navy',
                    'peru': 'navy',
                    'chartreuse': 'navy',
                    'deepskyblue': 'navy'}


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
        dico_vector_input = {'aireImg': zzz,
                             'melodyId': zzz,
                             'nbPhoto': zzz,
                             'nbBlock': zzz,
                             'nbSign': zzz,
                             'supTitle': zzz,
                             'secTitle': zzz,
                             'subTitle': zzz,
                             'title': zzz,
                             'abstract': zzz,
                             'exergue': zzz,
                             'syn': zzz}
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
        widths = img_soup.find_all('crop_w')
        heights = img_soup.find_all('crop_h')
        for width_bal, height_bal in zip(widths, heights):
            width, height = width_bal.text, height_bal.text
            try:
                aire_img += int(width) * int(height) / 100
            except:
                print("width", width)
                print("height", height)
    sup_title = CleanBal(art_soup.find('suptitle').text)
    sec_title = CleanBal(art_soup.find('secondarytitle').text)
    sub_title = CleanBal(art_soup.find('subtitle').text)
    title = CleanBal(art_soup.find('title').text)
    abstract = CleanBal(art_soup.find('abstract').text)
    syn = CleanBal(art_soup.find('title').text)
    exergue = CleanBal(art_soup.find('exergue').text)
    sup_title_indicatrice = 1 if len(sup_title) > 0 else 0
    sec_title_indicatrice = 1 if len(sec_title) > 0 else 0
    sub_title_indicatrice = 1 if len(sub_title) > 0 else 0
    title_indicatrice = 1 if len(title) > 0 else 0
    abstract_indicatrice = 1 if len(abstract) > 0 else 0
    syn_indicatrice = 1 if len(syn) > 0 else 0
    exergue_indicatrice = 1 if len(exergue) > 0 else 0
    dico_vector_input['aireImg'] = aire_img
    dico_vector_input['melodyId'] = int(art_soup.find('id').text)
    dico_vector_input['nbPhoto'] = int(art_soup.find('nbphoto').text)
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
    return dico_vector_input


def ExtractMDP(file_path):
    """
    Anciennement : CreateListMDPInput(file_path).
    Extrait les cartons d'un MDP contenus dans un fichier.
    Renvoie un dico de la forme :
        dico_mdp = {id_mdp: {id_carton_1: {'nbCol': xxx,
                                           'nbImg': xxx,
                                           'height': xxx,
                                           'width': xxx,
                                           'x': xxx,
                                           'y': xxx},
                             id_carton_2: {...}, ...}}
    """
    with open(file_path, "r", encoding='utf-8') as file:
        try:
            content = file.readlines()
        except Exception as e:
            print(str(file))
        content = "".join(content)
        soup = BeautifulSoup(content, "lxml")
    page_template_soup = soup.find('pagetemplate')
    id_mdp = int(page_template_soup.find('id').text)
    cartons_soup = soup.find('cartons')
    cartons = []
    dico_mdp = {id_mdp: {}}
    for carton_soup in soup.find_all('carton'):
        nb_col = carton_soup.find('nbcol')
        nb_images = carton_soup.find('nbimages')
        mm_height = carton_soup.find('mmheight')
        mm_width = carton_soup.find('mmwidth')
        x_precision = carton_soup.find('x_precision')
        y_precision = carton_soup.find('y_precision')
        id_carton = carton_soup.find('id').text
        dico_mdp[id_mdp][id_carton] = {}
        dico_mdp[id_mdp][id_carton]['nbCol'] = int(nb_col.text)
        dico_mdp[id_mdp][id_carton]['nbImg'] = int(nb_images.text)
        dico_mdp[id_mdp][id_carton]['height'] = int(mm_height.text)
        dico_mdp[id_mdp][id_carton]['width'] = int(mm_width.text)
        dico_mdp[id_mdp][id_carton]['x'] = int(float(x_precision.text))
        dico_mdp[id_mdp][id_carton]['y'] = int(float(y_precision.text))
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
        id_art, vect_art = useful_methods.GetVectorArticleInput(*args)
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
    """
    mdp_trouve = []
    for nb_pages, mdp, liste_ids in liste_tuple_mdp:
        if np.allclose(mdp, mdp_input, atol=30):
            mdp_trouve.append((nb_pages, mdp, liste_ids))
    if len(mdp_trouve) == 0:
        stc_exc = "No correspondance found for that MDP:\n{}\n in the BDD."
        raise MyException(stc_exc.format(mdp_input))
    mdp_trouve.sort(key=itemgetter(0), reverse=True)
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
                Article.set("MelodyId", str(value))
    tree = ET.ElementTree(PagesLayout)
    tree.write(file_out, encoding="UTF-8")
    return True


def ProposalsWithScore(dict_mlv_filtered):
    """
    Création d'un dico avec id_art: score
    on enlève le score du dico_filtered
    on utilise la fonction pour faire le produit de listes
    on calcule le score pour chaque ventilation
    on renvoie les 3 ventilations avec les meilleurs scores
    """
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
    every_possib = useful_methods.ProductList(*l_args)
    # Calcul du score, on le rajoute à chaque possib
    liste_possib_score = []
    for possib in every_possib:
        score = 0
        for i, art_possib in enumerate(possib):
            list_score_numemp = dict_id_score[art_possib]
            try:
                score_emp_art = list(filter(lambda x: x[1] == i,
                                            list_score_numemp))[0][0]
                score += score_emp_art
            except Exception as e:
                args_format = [e, art_possib, dict_id_score[art_possib]]
                print("{} art_possib {} res du dico {}".format(*args_format))
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
                    args_zip = [liste_art_one_poss, dict_mdp.values()]
                    for id_art, dict_carton in zip(*args_zip):
                        tuple_emp = (dict_carton['x'], dict_carton['y'])
                        dict_one_res[tuple_emp] = id_art
                    liste_page_created.append(dict_one_res)
    return liste_page_created


def ExtractAndComputeProposals(dico_bdd, liste_mdp_bdd, file_in, file_out):
    """
    'file_in' correspond à une directory avec 3 dossiers correspondants aux
    articles, mdp et rubriques.
    Fonction globale qui sera importée dans le webservice montage_ia
    - Dézippe archive xml
    - Crée des listes de dictionnaires avec les articles et les MDP
    - Détermine les pages possibles avec ces articles et ces MDP en utilisant
    les différentes méthodes
        - filtre des possibilités avec la méthode naïve
        - placement des articles avec la méthode MLV
    - Crée un fichier xml avec les différentes possibilités de placements
    d'articles. Dans les fichiers xml, on associe des id de MDP et des id
    d'articles et leur position (x, y).
    """
    # Isolation répertoire du fichier file_in
    try:
        dir_file_in = os.path.dirname(file_in)
    except Exception as e:
        print("An exception occurs with os.path.dirname()", file_in)
    # Isolation nom du fichier file_in
    try:
        basename_file_in = os.path.basename(file_in)
    except:
        print('An exception occurs with os.path.basename()', file_in)
    # On dézippe l'archive et on crée un dossier input avec les xml dézippés
    print("file_in: {}".format(file_in))
    with zipfile.ZipFile(file_in, "r") as z:
        z.extractall(dir_file_in + "/input")
    path_data_input = dir_file_in + '/input/' + basename_file_in[:-4]
    print("The path_data_input: {}".format(path_data_input))
    # On met les données extraites dans des listes de dico
    list_articles_INPUT, mdp_INPUT, rub_INPUT = ExtrationDataInput(path_data_input)
    # Affichage MDP input
    print("{:-^75}".format("MDP-INPUT"))
    for id_mdp, dict_mdp_input in mdp_INPUT.items():
        print("id mdp: {:<25}".format(id_mdp))
        for key, val in dict_mdp_input.items():
            print("{} {}".format(key, val))
    print("{:-^75}".format("END-INPUT"))
    # Affichage Articles input
    print("{:-^75}".format("ART-INPUT"))
    for art_input in list_articles_INPUT:
        print("art input: {}".format(art_input))
    print("{:-^75}".format("END-INPUT"))
    # Attribution sommaire (pour l'instant) d'une rubrique aux articles
    # ICI IL FAUT TROUVER UN MEILLEUR MOYEN
    #list_rub = ['SPORTS', 'INFOS', 'LOC']
    # Il s'agit du meilleur moyen
    list_rub = rub_INPUT
    dico_arts_rub_INPUT = {rub: [] for rub in list_rub}
    for dict_art in list_articles_INPUT:
        try:
            rub_picked = np.random.choice(list_rub)
        except Exception as e:
            exc = "An exception occurs with np.random.choice()"
            print(exc, list_rub, type(list_rub))
            rub_picked = list_rub[0]
        # Si la rub est déjà présente, on ajoute l'art à la liste
        dico_arts_rub_INPUT[rub_picked].append(dict_art)
    # La liste des features qui seront associées aux vecteurs articles
    # ON ENLEVE 'nbCol'
    list_of_features = ['nbSign', 'nbBlock', 'abstract', 'syn']
    list_of_features += ['exergue', 'title', 'secTitle', 'supTitle']
    list_of_features += ['subTitle', 'nbPhoto', 'aireImg']
    # Obtention des indices de nbSign, nbPhoto, aireImg pour FiltreArticles
    ind_features = [list_of_features.index('nbSign'),
                    list_of_features.index('nbPhoto'),
                    list_of_features.index('aireImg')]
    # Vérification du dico dico_arts_rub_INPUT
    print("{:-^75}".format("dico_arts_rub_INPUT"))
    for rub, liste_art in dico_arts_rub_INPUT.items():
        print(rub)
        for art in liste_art:
            print(art)
    print("{:-^75}".format("END dico_arts_rub_INPUT"))     
    # Il faut transformer les liste_art en matrice X_INPUT et dico_id_artv
    dico_arts_rub_X = {}
    for rub, liste_art in dico_arts_rub_INPUT.items():
        big_x, dico_id_artv = GetXInput(liste_art, list_of_features)
        dico_arts_rub_X[rub] = (big_x, dico_id_artv)
    # L'obtention du dico_arts_rub_X est une étape important
    # Visualisation des résultats
    print("{:-^75}".format("dico_arts_rub_X"))
    for rub, (mat_x, dico_art) in dico_arts_rub_X.items():
        print("La rubrique : {}".format(rub))
        print("La matrice X associée à tous les articles INPUT : \n{}".format(mat_x))
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
        list_cartons = [[int(dict_carton['x']),
                         int(dict_carton['y']),
                         int(dict_carton['width']),
                         int(dict_carton['height'])]
                        for dict_carton in dict_mdp_input.values()]
        X_nbImg = [[int(dict_carton['nbCol']), int(dict_carton['nbImg'])]
                   for dict_carton in dict_mdp_input.values()]
        mdp_loop = np.array(list_cartons)
        # Recherche d'une correspondance dans la BDD
        try:
            nb_art = mdp_loop.shape[0]
            liste_tuple_mdp = SelectionModelesPages(liste_mdp_bdd, nb_art, 15)
            res_found = TrouveMDPSim(liste_tuple_mdp, mdp_loop)
            mdp_ref = res_found[0][1]
            liste_ids_found = res_found[0][2]
        except Exception as e:
            print("An exception occurs while searching for correspondances: ", e)
        # On parcourt les articles INPUT classés par rubrique
        for rub, (X_input, dico_id_artv) in dico_arts_rub_X.items():
            # Utilisation de la méthode naïve pour filtrer obtenir les
            # possibilités de placements
            try:
                args_naive = [mdp_ref, X_nbImg, X_input, dico_id_artv, colors, ind_features]
                args_mlv = [dico_bdd, liste_ids_found, mdp_ref, X_input]
                dico_possi_naive = useful_methods.MethodeNaive(*args_naive)
                # Affichage Results naive
                print("{:-^75}".format("NAIVE"))
                for emp, poss in dico_possi_naive.items():
                    nice_emp = ["{:.0f}".format(x) for x in emp]
                    try:
                        nice_poss = ["{}".format(id_art) for id_art in poss]
                        print("{:<35} {}".format(str(nice_emp), nice_poss))
                    except:
                        print("Erreur affichage nice_poss", poss)
                print("{:-^75}".format("END NAIVE"))
                dico_possi_mlv = useful_methods.MethodeMLV(*args_mlv, dico_id_artv)
                args_filt = [dico_possi_naive, dico_possi_mlv]
                dict_mlv_filtered = useful_methods.ResultsFiltered(*args_filt)
                # Affichage des résultats
                print("{:-^75}".format("MLV"))
                for emp, poss in dict_mlv_filtered.items():
                    nice_emp = ["{:.0f}".format(x) for x in emp]
                    nice_poss = ["{:.2f} {}".format(score, id_art)
                                 for score, id_art in poss]
                    print("{:<35} {}".format(str(nice_emp), nice_poss))
                # Génération des objets avec toutes les pages possibles, s'il 
                # y a des articles pour chaque EMP
                first_results = ProposalsWithScore(dict_mlv_filtered)
                dict_global_results[id_mdp][rub] = first_results
                print("The 3 results with the best score")
                for score, list_ids in first_results:
                    str_print = ""
                    for id_art in list_ids:
                        str_print += "--{}"
                    print(("{:-<15.2f}" + str_print).format(score, *list_ids))
                print("{:-^75}".format("END MDP"))
            except Exception as e:
                logger.info(e, exc_info=True)
                exc = "An exception occurs while applying the naive method: "
                print(exc, e)
    list_created_pg = CreateListeProposalsPage(dict_global_results, mdp_INPUT)
    print("The list of page created: {}".format(list_created_pg))
    CreateXmlPageProposals(list_created_pg, file_out)
    return "Xml output created"
