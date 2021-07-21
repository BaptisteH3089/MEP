#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Baptiste Hessel

Contains some important functions relative to the process of creation of new
pages and the filtering of the articles. For example, it contains the
functions:
    - MethodeNaive()
    - MethodeMLV()
"""
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from operator import itemgetter


class MyException(Exception):
    pass


def GetYClasses(big_y):
    """
    Renvoie un liste de tuples [(vect_emp, ind_classe), ...]
    """
    n = big_y.shape[0]
    return [(big_y[i], i) for i in range(n)]


def TransformArrayIntoClasses(big_y, list_classes):
    """
    big_y : matrice de la forme nb_arts * 4. Les colonnes sont (x, y, w, h).
    list_classes : liste de la forme [(vect_emp, ind_classe), ...]
    Transforme la matrice big_y en vecteur avec des classes.
    Renvoie un numpy array de longueur nb_arts avec des classes (0, 1, 2, ...)
    au lieu d'un vect_emp.
    little_y est de dimension 1
    """
    n = big_y.shape[0]
    little_y = np.zeros(n)
    # Pour chaque élément de la matrice big_y, on regarde tous les éléments
    # vect_emp de la liste list_classes.
    for i in range(n):
        # Si la diff max de tous les éléments est < 20, on attribue la classe
        # au vect_emp
        if max(abs(big_y[i] - list_classes[i][0])) < 20:
            little_y[i] = list_classes[i][1]
        else:
            for j in range(n):
                if max(abs(big_y[i] - list_classes[j][0])) < 20:
                    little_y[i] = list_classes[j][1]
                    break
    return little_y


def GetVectorArticleInput(dico_vector_input, features):
    """
    Transforme le dico_vector_input en un vecteur.
    Renvoie un tuple (id_art, vecteur_art).
    vect_art est un numpy array de dimension 2.
    Transforme les variables secTitle, title, etc... en indicatrices.
    """
    features_left = set(features) - set(dico_vector_input.keys())
    if len(features_left) > 0:
        sentence = "Some features aren't in the dict:\n"
        raise MyException(sentence + "{}".format(features_left))
    vector_art = []
    other_features = ['abstract', 'syn', 'exergue', 'title', 'secTitle']
    other_features += ['subTitle', 'supTitle']
    for feature in features:
        if feature == 'nbSign':
            if dico_vector_input['nbSign'] == 0:
                print("NbSign == 0 l.176 - GetVectorArticleInput")
                vector_art.append(dico_vector_input[feature])
            else:
                vector_art.append(dico_vector_input[feature])
        # Conversion des variables en indicatrices
        # Normalement plus la peine, comme déjà fait auparavant
        elif feature in other_features:
            if dico_vector_input[feature] > 0:
                vector_art.append(1)
            else:
                vector_art.append(0)
        else:
            vector_art.append(dico_vector_input[feature])
    return (dico_vector_input['melodyId'], np.array([vector_art]))


def GetTrainableData(dico_bdd, list_id_pages, y_ref, features):
    """
    features est une liste avec toutes les features qu'on veut utiliser pour
    former la grande matrice X_input
    les features à garder :
        - nbSign
        - nbBlock
        - abstract
        - syn
        - exergue
        - title
        - secTitle
        - supTitle
        - subTitle
        - nbPhoto
        - aireImg
        - aireTot ?
    """
    # Transformation de big_y into simpler classes
    classes = GetYClasses(y_ref)
    for cpt, id_page in enumerate(list_id_pages):
        # Formation de la big matrice X pour toute la page
        x_all_articles, y_all_articles = np.zeros(0), np.zeros(0)
        # Récupération du dico associé à chaque art dans la BDD
        dico_arts_page = dico_bdd[id_page]['articles']
        # Boucle pour créer les matrices x, y associées aux pages
        for i, (id_art, dico_art) in enumerate(dico_arts_page.items()):
            # Transformation du dico en vecteur
            melo_id, vect_art = GetVectorArticleInput(dico_art, features)
            # Formation numpy array emp y
            list_y = [dico_art[x] for x in ['x', 'y', 'width', 'height']]
            y_art = np.array(list_y, ndmin=2)
            if i == 0:
                x_all_articles = np.array(vect_art, ndmin=2)
                y_all_articles = y_art
            else:
                # On concatène le nouveau vect avec x_all_articles
                try:
                    x_all_articles = np.concatenate((x_all_articles, vect_art))
                except Exception as e:
                    dim_x = x_all_articles.shape
                    str_exc = "Error with np.concatenate"
                    str_exc += "((x_all_articles, vect_art))\n"
                    str_exc += "x_all_articles.shape : {}\n".format(dim_x)
                    str_exc += "vect_art.shape : {}\n".format(vect_art.shape)
                    raise MyException(str_exc + str(e))
                try:
                    y_all_articles = np.concatenate((y_all_articles, y_art))
                except Exception as e:
                    dim_y = y_all_articles.shape
                    str_exc = "Error with np.concatenate"
                    str_exc += "((y_all_articles, y_art))\n"
                    str_exc += "y_all_articles.shape : {}\n".format(dim_y)
                    str_exc += "y_art.shape : {}\n".format(y_art.shape)
                    raise MyException(str_exc + str(e))
        # On transforme la matrice y_all_articles en matrice avec des classes
        # au lieu de vecteurs emp.
        try:
            y = TransformArrayIntoClasses(y_all_articles, classes)
        except Exception as e:
            str_exc = "Error with TransformArrayIntoClasses: {}\n".format(e)
            str_exc = "y_all_arts.shape: {}\n".format(y_all_articles.shape)
            str_exc = "classes: {}".format(classes)
            raise MyException(str_exc)
        # Ajout des matrices x, y associées aux pages aux big_matrix X, Y
        if cpt == 0:
            very_big_x = x_all_articles
            very_big_y = y
        else:
            try:
                very_big_x = np.concatenate((very_big_x, x_all_articles))
            except Exception as e:
                dim_very_big_x = very_big_x.shape
                dim_x_all_arts = x_all_articles.shape
                str_exc = "Error with np.concatenate"
                str_exc += "((very_big_x, x_all_articles))\n"
                str_exc += "very_big_x.shape : {}\n".format(dim_x)
                str_exc += "x_all_arts.shape : {}\n".format(dim_x_all_arts)
                raise MyException(str_exc + str(e))
            try:
                very_big_y = np.concatenate((very_big_y, y))
            except Exception as e:
                dim_very_big_y = very_big_y.shape
                str_exc = "Error with np.concatenate((very_big_y, y))\n"
                str_exc += "very_big_y.shape : {}\n".format(dim_very_big_y)
                str_exc += "y: {}\n".format(y)
                raise MyException(str_exc + str(e))
    return very_big_x, very_big_y


def GetPossibilitiesMLV(probas_preds, X_rand, dico_id_vect_art):
    """
    Renvoie un dictionnaire avec pour chaque emp les articles qui ont obtenus
    une proba prédite supérieure à 30%.
    """
    dico_possib_emplacements = {i: [] for i in range(probas_preds.shape[1])}
    for i in range(len(probas_preds)):
        preds_art = probas_preds[i]
        vect_art = X_rand[i]
        f = lambda x: np.allclose(x[1], vect_art, atol=2)
        res_id_artx = list(filter(f, dico_id_vect_art.items()))
        if len(res_id_artx) == 0:
            val_dico = dico_id_vect_art.values()
            str_exc = "No vect_art found in the dico_id_vect_art which "
            str_exc += "corresponds to that vect input\n"
            str_exc += "vect_art: {}\n".format(vect_art)
            str_exc += "dico_id_vect_art.values(): {}".format(val_dico)
            raise MyException(str_exc)
        id_x_rand_i = res_id_artx[0][0]
        # On conserve les classes avec proba > 30%
        liste_tuple = [(i, pred) for i, pred in enumerate(preds_art)]
        poss_classes = list(filter(lambda x: x[1] >= 0.3, liste_tuple))
        if len(poss_classes) == 0:
            nice_liste = ["{:.2f}".format(score) for _, score in liste_tuple]
            str_exc = "No possibility for that article. Predicted probas: {}."
            raise MyException(str_exc.format(nice_liste))
        # Ajout des classes à un dico i
        for emp, score_pred in poss_classes:
            dico_possib_emplacements[emp].append((score_pred, id_x_rand_i))
    return dico_possib_emplacements


def MethodeMLV(dico_bdd,
               liste_id_pages,
               mdp_ref,
               X_input,
               dico_id_vect_art,
               list_features):
    """
    Retourne les possibilités de placements d'articles dans un MDP en se
    basant sur les prédictions d'un RFC entraîné sur les pages construites
    avec le même MDP.
    """
    # Création des matrices X avec les articles et Y avec les classes des EMP
    X, Y = GetTrainableData(dico_bdd, liste_id_pages, mdp_ref, list_features)
    # Vérification des tailles des matrices X, Y
    if X.shape[0] != Y.shape[0]:
        str_exc = "Les matrices X et Y n'ont pas le même nombre de lignes.\n"
        str_exc += "X.shape: {}\n".format(X.shape)
        str_exc += "Y.shape: {}".format(Y.shape)
        raise MyException(str_exc)
    # Les matrices X_input et X doivent avoir le même nombre de colonnes
    if X.shape[1] != X_input.shape[1]:
        str_exc = "Les matrices X et X_input n'ont pas le mm nb de cols.\n"
        str_exc += "X.shape: {}\n".format(X.shape)
        str_exc += "X_input.shape: {}\n".format(X_input.shape)
        str_exc += "X[0]: {}\n".format(X[0])
        str_exc += "X_input[0]: {}".format(X_input[0])
        raise MyException(str_exc)
    param_model = {'max_depth': 5,
                   'min_samples_split': 4,
                   'min_samples_leaf': 4,
                   'max_features': 0.5,
                   'max_leaf_nodes': 10}
    clf = RandomForestClassifier(**param_model)
    clf.fit(X, Y)
    # Matrix with the predicted probas for each article input
    probas_preds = clf.predict_proba(X_input)
    # Obtention de possibilités de placement des articles
    l_args = [probas_preds, X_input, dico_id_vect_art]
    dico_poss_plcmts = GetPossibilitiesMLV(*l_args)
    # Remplacer les emplacements 0, ..., 3 par les vrais tuples EMP
    classes = GetYClasses(mdp_ref)
    dico_plcmts_tuple = {}
    for emp, possib_art in dico_poss_plcmts.items():
        array_emp = list(filter(lambda x: x[1] == emp, classes))[0][0]
        dico_plcmts_tuple[tuple(array_emp)] = possib_art
    size_each_location = [len(x) for x in dico_poss_plcmts.values()]
    return dico_plcmts_tuple


def GetDicoCarton(mdp_ref, X_col_img):
    """
    Création d'un dico avec chaque carton et les caractéristiques dedans.
    """
    keys_model = ['x', 'y', 'width', 'height', 'nbCol', 'nbImg', 'aireImg']
    keys_model += ['aireTot']
    dico_cartons = {}
    for i, (carton, nb_col_img) in enumerate(zip(mdp_ref, X_col_img)):
        dico_cartons[i] = {}
        big_carton = list(carton) + list(nb_col_img)
        for j, (caract, key) in enumerate(zip(big_carton, keys_model)):
            dico_cartons[i][key] = caract
    return dico_cartons


def FiltreArticles(dico_id_artv, aire_tot_cart, aire_img_cart, ind_features):
    """
    ind_features = [ind_nb_sign, ind_aire_tot, ind_aire_img]
    - on fait le filtre sur le nb de caract d'un art - aire des images
    convertie en nb de caract
    - si le MDP comprend des images on doit soustraire l'aire des images
    de l'article à celle de l'aire totale
    ATTENTION il faut penser à changer les indices de nb_img et area si
    changement dans les features
    UPDATE 28/05 :
        - on enlève le filtre avec nb_img, on fait juste avec aire_img
        - suppression condition aire img < 2/3 * area_mdp
    """
    ind_nb_sign = ind_features[0]
    # ind_aire_tot_art
    tot = ind_features[1]
    # ind_area_img_art
    img = ind_features[2]
    print("{:<25} {:^25} {:>25}".format('TOT', 'IMG', 'TXT'))
    for ida, artv in dico_id_artv.items():
        txt = artv[0][tot] - artv[0][img]
        print("{:<25} {:^25} {:>25}".format(artv[0][tot], artv[0][img], txt))
    # Filtre sur l'aire des images
    print("L'aire des images du carton : {}".format(aire_img_cart))
    fcn = lambda x: (0.5 * aire_img_cart <= x[1][0][img] <= 1.5 * aire_img_cart)
    arts_left_img = [ida for ida, _ in filter(fcn, list(dico_id_artv.items()))]
    print("les ids des articles qui correspondent : {}".format(arts_left_img))
    dico_id_artv_img = {ida: dicoa for ida, dicoa in dico_id_artv.items()
                        if ida in arts_left_img}
    if len(arts_left_img) == 0:
        str_exc = 'No article for that carton with the filter on the area '
        str_exc += 'of the images.'
        raise MyException(str_exc)
    # Filtre sur l'aire du texte
    diff = aire_tot_cart - aire_img_cart
    print("L'aire du txt du carton : {}".format(diff))
    fcn = lambda x: (0.6 * diff <  x[1][0][tot] - x[1][0][img] < 1.4 * diff)
    list_items_dico_idartv = list(dico_id_artv_img.items())
    arts_left_txt = [ida for ida, _ in filter(fcn, list_items_dico_idartv)]
    print("les ids des articles qui correspondent : {}".format(arts_left_txt))
    dico_id_artv_txt = {ida: dicoa for ida, dicoa in dico_id_artv_img.items()
                        if ida in arts_left_txt}
    if len(arts_left_txt) == 0:
        str_exc = 'No article for that carton with the filter on the area '
        str_exc += 'of the text.'
        raise MyException(str_exc)
    # Filtre sur l'aire totale
    print("L'aire du carton : {}".format(aire_tot_cart))
    fcn = lambda x: (0.7 * aire_tot_cart < x[1][0][tot] < 1.3 * aire_tot_cart)
    arts_left = [ida for ida, _ in filter(fcn, list(dico_id_artv_txt.items()))]
    print("les ids des articles qui correspondent : {}".format(arts_left))
    dico_id_artv_left = {ida: dicoa for ida, dicoa in dico_id_artv_txt.items()
                         if ida in arts_left_txt}
    if len(arts_left) == 0:
        str_exc = 'No art. for that carton with the filter on the total area'
        raise MyException(str_exc)
    return arts_left


# Génére toutes les possibilités de placements d'articles
def ProductList(l1, l2, l3=None, l4=None, l5=None, l6=None, l7=None):
    if l3 is not None:
        if l4 is not None:
            if l5 is not None:
                if l6 is not None:
                    if l7 is not None:
                        return [[a, b, c, d, e, f, g] for a in l1
                                for b in l2 for c in l3
                                for d in l4 for e in l5
                                for f in l6 for g in l7]
                    else:
                        return [[a, b, c, d, e, f]
                                for a in l1 for b in l2 for c in l3
                                for d in l4 for e in l5 for f in l6]
                else:
                    return [[a, b, c, d, e] for a in l1
                            for b in l2 for c in l3
                            for d in l4 for e in l5]
            else:
                return [[a, b, c, d]
                        for a in l1 for b in l2
                        for c in l3 for d in l4]
        else:
            return [[a, b, c] for a in l1 for b in l2 for c in l3]
    return [[x, y] for x in l1 for y in l2]


def ScoreArticle(dico_id_artv, dico_cartons, id_art, id_carton, ind_features):
    """
    Retourne un score basé sur la distance entre aireCarton et nbSignArticle.
    Il faut aussi prendre en compte le nbImg dans le Carton.
    """
    aire_img_moy = 6000
    nb_img = dico_cartons[id_carton]['nbImg']
    width = dico_cartons[id_carton]['width']
    height = dico_cartons[id_carton]['height']
    aire_carton = width * height
    # la correspondance aire - nbSign est d'environ nbSign ~ aire / 10
    aire_relative = aire_carton - (5000 * nb_img) / 10
    try:
         good_id_art = int(id_art)
    except Exception as e:
        print("Error with int(id_art).\n")
        print("type(id_art): {}. id_art: {}".format(type(id_art), id_art))
        good_id_art = id_art
    # A EVENTUELLEMENT CHANGER. Ajout argument ind_nb_sign ?
    ind_nb_sign = ind_features[0]
    nb_sign_article = dico_id_artv[good_id_art][0][ind_nb_sign]
    distance = np.sqrt((nb_sign_article - aire_relative) ** 2)
    return distance


def GetGoodVentMethNaive(every_possib_score):
    """
    Le problème dans l'objet précédent est qu'un article peut apparaître à 2
    endroits s'il peut être placé à ces endroits, donc on filtre.
    """
    nb_art_in_page = len(every_possib_score[0][1])
    f_uni = lambda x: len(set(x[1])) == nb_art_in_page
    ventilations_unique = list(filter(f_uni, every_possib_score))
    dico_vent_score = {}
    for score, liste_id_art in ventilations_unique:
        if tuple(liste_id_art) not in dico_vent_score.keys():
            dico_vent_score[tuple(liste_id_art)] = score
        else:
            best_score = min(score, dico_vent_score[tuple(liste_id_art)])
            dico_vent_score[tuple(liste_id_art)] = best_score
    # Suppression des doublons des ventilations
    vents = [(score, vent) for vent, score in dico_vent_score.items()]
    vents.sort(key=itemgetter(0))
    return vents


def SimpleLabels(liste_labels):
    """
    Retourne une liste de labels simplifiés
    """
    simpler_list = []
    for elt in liste_labels:
        if elt in ['TEXTE AUTEUR', 'TEXTE NOTE']:
            simpler_list.append('TEXTE')
        elif elt in ['SURTITRE TITRE', 'SURTITRE (COMMUNE) TITRE']:
            simpler_list.append('SURTITRE')
        elif 'PHOTO' in elt:
            simpler_list.append('PHOTO')
        else:
            simpler_list.append(elt)
    return simpler_list


def FiltreBlocs(dict_mdp_input, list_articles_input, dico_cartons, emp):
    """
    Retourne les articles dont les labels des blocs correspondent à ceux des
    cartons
    dict_mdp_input = {'3764':
        {'x': 15, 'y': 323, 'width': 280, 'height': 126,
        'nbCol': 6, 'nbImg': 1, 'aireTotImgs': 8494,
        'listeAireImgs': [8494], 'aireTot': 44607,
        'blocs': [{'labels':, ...}, {}, ...]}
    list_articles_input = [{'blocs': [], melodyId:..., ...}]
    UPDATE 09/06: We do an inclusion instead of an equality the last lines
    """
    liste_arts_poss = []
    # Déjà, il faut que je récupère le dico du carton
    x_cart, y_cart = dico_cartons[emp]['x'], dico_cartons[emp]['y']
    f = lambda x: (np.allclose(x[1]['x'], x_cart, atol=0.5)) & \
        (np.allclose(x[1]['y'], y_cart, atol=1))
    carton_found = dict(filter(f, dict_mdp_input.items()))
    if len(carton_found) != 1:
        str_exc = 'Error in FiltreBlocs \n'
        str_exc += 'carton found: {}'.format(carton_found)
        str_exc += '(x_cart, y_cart): ({}, {}) \n'.format(x_cart, y_cart)
        str_exc += 'dict_mdp_input: {}'.format(dict_mdp_input)
        raise MyException(str_exc)
    id_carton = list(carton_found.keys())[0]
    # print("The carton found with x and y : \n{}".format(carton_found.values()))
    # On parcourt les articles, et on compare les éléments blocs
    for dicoa in list_articles_input:
        # On transforme les éléments blocs en une liste de labels plus simple
        blocs_label_art = [dicob['label'] for dicob in dicoa['blocs']]
        blocs_label_carton = [dicob['label']
                              for dicob in carton_found[id_carton]['blocs']]
        # print("Les labels des articles : {}".format(blocs_label_art))
        # print("Les labels des cartons : {}".format(blocs_label_carton))
        labels_art = SimpleLabels(blocs_label_art)
        labels_carton = SimpleLabels(blocs_label_carton)
        print("Les labels simples des articles : {}".format(labels_art))
        print("Les labels simples des cartons : {}\n".format(labels_carton))
        # If all blocks of the article are also present in the carton, we can
        # place this article in that carton.
        if set(labels_art) <= set(labels_carton):
            liste_arts_poss.append(dicoa['melodyId'])
    return liste_arts_poss


def MethodeNaive(mdp_ref, X_nbImg, X_input, dico_id_artv, ind_features,
                 dict_mdp_input, list_articles_input):
    """
    Renvoie des ventilations en se basant sur le nb d'images dans les cartons
    et l'aire du carton par rapport au nombre de signes des articles et à
    l'aire totale des images.
    Pour faire simple, donne toutes les possibilités de placements tq les
    articles correspondent à peu près aux cartons
    ind_features = [ind_nb_sign, ind_aire_tot, ind_area_img]
    """
    dico_cartons = GetDicoCarton(mdp_ref, X_nbImg)
    # Création du dico dico_possib_placements
    dico_possib_placements = {id_emp: [] for id_emp in dico_cartons.keys()}
    for emp, liste_art_poss in dico_possib_placements.items():
        # CORRESPONDANCE DES BLOCS
        args_filtre = [dict_mdp_input, list_articles_input, dico_cartons, emp]
        liste_art_poss += FiltreBlocs(*args_filtre)
        str_print = "The list of possible arts after the bloc filter"
        print(str_print + " : {}".format(liste_art_poss))
        # Pour chaque article, on regarde s'il est insérable à cet emplacement
        nb_img = dico_cartons[emp]['nbImg']
        aire_img_carton = dico_cartons[emp]['aireImg']
        # HUM, VOIR SI PAS MIEUX DE FAIRE AVEC ELTS DE OPTIONS. SI.
        #aire_tot_carton = dico_cartons[emp]['width'] * dico_cartons[emp]['height']
        #print("AIRE TOTALE AVEC w*h {}".format(aire_tot_carton))
        aire_tot_carton = dico_cartons[emp]['aireTot']
        #print("AIRE TOTALE AVEC options {}".format(aire_tot_carton))
        ## GROS CHANGEMENTS ICI
        ## CORRESPONDANCE ENTRE AIRE IMG, AIRE TXT, AIRE TOTALE
        args_filt = [dico_id_artv, aire_tot_carton, aire_img_carton]
        args_filt += [ind_features]
        try:
            print("{:-^75}".format("FILTRE ART"))
            print("L'aire des imgs de ce MDP : {}".format(aire_img_carton))
            bornes = (.5 * aire_img_carton, 1.5 * aire_img_carton)
            print("bornes : {} et {}".format(*bornes))
            liste_art_poss += FiltreArticles(*args_filt)
        except Exception as e:
            first_item = list(dico_id_artv.items())[0]
            str_exc = "Error with FiltreArticles : {}\n".format(e)
            str_exc += "ind_features: {}\n".format(ind_features)
            str_exc += "aire_tot_carton: {}.\n".format(aire_tot_carton)
            str_exc += "aire_img_carton: {}.\n".format(aire_img_carton)
            str_exc += "extrait de dico_id_artv: {}".format(first_item)
            raise MyException(str_exc)
    liste_len = [len(list_id) for list_id in dico_possib_placements.values()]
    if 0 in liste_len:
        str_exc = "No possibility for at least one location: {}"
        raise MyException(str_exc.format(liste_len))
    l_args = [emp_poss for emp_poss in dico_possib_placements.values()]
    every_possib = ProductList(*l_args)
    # print("every_possib : \n {}".format(every_possib))
    # Calcul des scores de chaque ventil. every_possib = [[id1, id2], ...].
    # Retourne [([id1, id2, id3], score_vent), ...].
    every_possib_score = []
    for vent in every_possib:
        score_vent = 0
        for id_art, id_emp in zip(vent, dico_possib_placements.keys()):
            args = [dico_id_artv, dico_cartons, id_art, id_emp, ind_features]
            score_vent += ScoreArticle(*args)
        try:
            score_tot = int(score_vent)
        except Exception as e:
            str_exc = "Error with int(score_vent): {}\n".format(e)
            str_exc += "score_vent: {}".format(score_vent)
            str_exc += "type(score_vent): {}".format(type(score_vent))
            raise MyException(str_exc)
        every_possib_score.append((score_tot, vent))
    # ventilations triées
    ventilations_unique = GetGoodVentMethNaive(every_possib_score)
    # ventilations_unique = [(88176, (32234, 28797)), ...]
    # Il faut remplacer les ids des emplacements par les 4-uplets
    new_dico_possib_placements = {}
    for id_emp, liste_id_articles in dico_possib_placements.items():
        list_key = ['x', 'y', 'width', 'height']
        tuple_emp = tuple((dico_cartons[id_emp][key] for key in list_key))
        new_dico_possib_placements[tuple_emp] = liste_id_articles
    return new_dico_possib_placements, ventilations_unique


def ResultsFiltered(dico_possib_naive, dico_possib_mlv):
    # The two dictionaries must have the same keys
    if set(dico_possib_naive.keys()) != set(dico_possib_mlv.keys()):
        str_exc = "The two dicts input of ResultsFiltered don't have the "
        str_exc += "same keys. \n"
        str_exc += "keys d_naive: {}\n".format(list(dico_possib_naive.keys()))
        str_exc += "keys d_mlv: {}\n".format(list(dico_possib_mlv.keys()))
        raise MyException(str_exc)
    dict_mlv_filtered = {}
    for key_naive, key_mlv in zip(dico_possib_naive.keys(),
                                  dico_possib_mlv.keys()):
        f = lambda x: x[1] in dico_possib_naive[key_mlv]
        val_filtered = list(filter(f, dico_possib_mlv[key_mlv]))
        if len(val_filtered) == 0:
            str_exc = "After filtration with the predictions of the "
            str_exc += "mlv method, there is no possibility left"
            raise MyException(str_exc)
        dict_mlv_filtered[key_mlv] = val_filtered
    return dict_mlv_filtered
