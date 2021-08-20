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
    Associates a label to a vector location (x, y, width, height).

    Parameters
    ----------
    big_y: numpy array
        A matrix associated to a layout.
        dim(big_y) = (nb_articles_in_page, 4)

    Returns
    -------
    list of tuples
        [(numpy.array([x0, y0, w0, h0]), 0), ...].

    """

    n = big_y.shape[0]
    return [(big_y[i], i) for i in range(n)]


def TransformArrayIntoClasses(big_y, list_classes, tol=10):
    """
    Uses the correspondance between the vect_modules and the labels of these
    vectors to convert a layout into an array with the labels of the modules.

    Parameters
    ----------
    big_y: numpy array
        A matrix associated to a layout.
        dim(big_y) = (nb_arts, 4)

    list_classes: list of tuples
        list_classes = [(numpy.array([x0, y0, w0, h0]), 0), ...].

    tol: int (default=10)
        The max difference allows between the elements of the vect_module of
        big_y and the elements in list_classes.

    Returns
    -------
    little_y: numpy array
        little_y = np.array([1, 0, 2, 3]) for a big_y with 4 articles.
    """
    n = big_y.shape[0]
    little_y = np.zeros(n)
    # For each vect_module of the matrix big_y, we search for a match into
    # the "list_classes"
    for i in range(n):
        # If the max diff is < tol, there is a match
        if max(abs(big_y[i] - list_classes[i][0])) < tol:
            little_y[i] = list_classes[i][1]
        else:
            for j in range(n):
                if max(abs(big_y[i] - list_classes[j][0])) < tol:
                    little_y[i] = list_classes[j][1]
                    break
    return little_y


def GetVectorArticleInput(dico_vector_input, features):
    """
    Returns a tuple with the id of the article and the vector article with
    the features in "features".

    Parameters
    ----------
    dico_vector_input: dict
        dico_vector_input = {'melodyId': ..., 'title': ..., ...}.

    features: list of strings
        DESCRIPTION.

    Raises
    ------
    MyException
        If one feature or more is not in the dict_vector_input.

    Returns
    -------
    tuple
        (melodyId_article, numpy.array(vector_article)).

    """
    features_left = set(features) - set(dico_vector_input.keys())
    if len(features_left) > 0:
        sentence = "Some features aren't in the dict:\n"
        raise MyException(sentence + "{}".format(features_left))

    vector_art = []
    other_features = ['abstract', 'syn', 'exergue', 'title', 'secTitle']
    other_features += ['subTitle', 'supTitle']
    for feature in features:
        if feature in other_features:
            # We only add indicators functions
            if dico_vector_input[feature] > 0:
                vector_art.append(1)
            else:
                vector_art.append(0)
        else:
            vector_art.append(dico_vector_input[feature])
    return (dico_vector_input['melodyId'], np.array([vector_art]))


def GetTrainableData(dico_bdd, list_id_pages, y_ref, features):
    """

    Parameters
    ----------
    dico_bdd: dict
        The usual dict with all the info about the pages.

    list_id_pages: list
        list_id_pages = [id_page0, id_page1, ...].

    y_ref: numpy array
        The matrix associated to a layout.

    features: list of features
        List with the features used to create the matrix X.

    Raises
    ------
    MyException
        If there is an error during the concatenation or the transformation
        into classes.

    Returns
    -------
    very_big_x: numpy array
        The matrix with all the vectors articles concatenated by lines.

    very_big_y: numpy array
        The matrix with all the labels.

    """
    # Transformation de big_y into simpler classes
    classes = GetYClasses(y_ref)
    for cpt, id_page in enumerate(list_id_pages):
        # Formation de la big matrice X pour toute la page
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
                str_exc += "very_big_x.shape : {}\n".format(dim_very_big_x)
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


def GetPossibilitiesMLV(probas_preds, X_input, dico_id_vect_art):
    """
    Returns a dict that indicates for each location the articles that obtained
    a predicted proba > 30%.

    Parameters
    ----------
    probas_preds: numpy array
        The predicted probas for each article and each module.

    X_input: numpy array
        The matrix with all the articles input.

    dico_id_vect_art: dict
        DESCRIPTION.

    Raises
    ------
    MyException
        If we don't find any module with a proba > 30%.

    Returns
    -------
    dico_possib_emplacements: dict
        dico_possib_emplacements = {emp: [(score_pred, id_article), ...],
                                    ...}.

    """
    dico_possib_emplacements = {i: [] for i in range(probas_preds.shape[1])}
    for i in range(len(probas_preds)):
        preds_art = probas_preds[i]
        vect_art = X_input[i]
        f = lambda x: np.allclose(x[1], vect_art, atol=2)
        res_id_artx = list(filter(f, dico_id_vect_art.items()))

        if len(res_id_artx) == 0:
            val_dico = dico_id_vect_art.values()
            str_exc = "No vect_art found in the dico_id_vect_art which "
            str_exc += "corresponds to that vect input\n"
            str_exc += "vect_art: {}\n".format(vect_art)
            str_exc += "dico_id_vect_art.values(): {}".format(val_dico)
            raise MyException(str_exc)

        id_x_input_i = res_id_artx[0][0]
        # We keep the classes with proba > 30%
        liste_tuple = [(i, pred) for i, pred in enumerate(preds_art)]
        poss_classes = list(filter(lambda x: x[1] >= 0.3, liste_tuple))

        if len(poss_classes) == 0:
            nice_liste = ["{:.2f}".format(score) for _, score in liste_tuple]
            str_exc = "No possibility for that article. Predicted probas: {}."
            raise MyException(str_exc.format(nice_liste))

        # Add classes to a dict
        for emp, score_pred in poss_classes:
            dico_possib_emplacements[emp].append((score_pred, id_x_input_i))

    return dico_possib_emplacements


def MethodeMLV(dico_bdd,
               liste_id_pages,
               mdp_ref,
               X_input,
               dico_id_vect_art,
               list_features):
    """
    Returns the list of articles that can be inserted in each module in a dict.
    We use a RFC trained on the pages build with the same layout to give a
    score that corresponds to the fit of the article in the given module.

    Parameters
    ----------
    dico_bdd: dict
        The usual dict with all the info about the pages.

    liste_id_pages: list of float
        The list with all the ids of pages that we use to create the matrixes
        X and Y.

    mdp_ref: numpy array
        The matrix associated to the layout input.
        dim(mdp_ref) = (nb_modules_layout, 4)

    X_input: numpy array
        The matrix with al the vectors articles input.
        dim(X_input) = (nb_articles_input, len(list_features))

    dico_id_vect_art: dict
        dico_id_vect_art = {id_article: vector_article, ...}.

    list_features: list of strings
        The list with the features used to create the vectors articles.

    Raises
    ------
    MyException
        If the matrix used to train the model hasn't to right number of
        columns.

    Returns
    -------
    dico_plcmts_tuple: dict
        dico_plcmts_tuple = {tuple_module_xywh: [id_art, ...], ...}.

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

    return dico_plcmts_tuple


def GetDicoCarton(mdp_ref, X_col_img):
    """
    Creates a dict with the characteristics of each module.

    Parameters
    ----------
    mdp_ref: numpy array
        The matrix associated to the layout input.

    X_col_img: list of lists
        X_col_img = [[d[x]
                      for x in ['nbCol', 'nbImg', 'aireTotImgs', 'aireTot']
                      for for dict_carton in dict_mdp_input.values()]

    Returns
    -------
    dico_cartons: dict
        dico_cartons = {0: {'x': ..., 'y':..., 'width': ...,
                            'height': ..., 'nbImg': ..., 'aireImg': ...},
                        1: {...},
                        ...}.
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


def FiltreArticles(dico_id_artv,
                   aire_tot_cart,
                   aire_img_cart,
                   ind_features,
                   verbose,
                   tol_area_images=0.5,
                   tol_area_text=0.4,
                   tol_total_area=0.3):
    """
    We filter the articles that respect the constraints imposed by the module.
        - filtering area of the images
        - filtering area text
        - filtering total area

    Parameters
    ----------
    dico_id_artv: dict
        dico_id_artv = {id_article: vect_article, ...}.

    aire_tot_cart: float
        The total area of the module.

    aire_img_cart: float
        the total area of the images in the module.

    ind_features: list
        ind_features = [ind_nb_sign, ind_aire_tot, ind_aire_img].

    verbose: int >= 0
        If verbose > 0, print intermediate results.

    tol_area_images: float, optional
        The tolerance between the area of the images of the article and the
        module. (default=0.5)

    tol_area_text: float, optional
        The tolerance between the area of the text of the article and the
        module. (default=0.4)

    tol_total_area: float, optional
        The tolerance between the area of the article and of the module.
        (default=0.3)

    Raises
    ------
    MyException
        If there is no possibility of articles for the module.

    Returns
    -------
    arts_left: list
        The ids of articles that can be inserted in that module.
    """

    # ind_aire_tot_art
    tot = ind_features[1]
    # ind_area_img_art
    img = ind_features[2]

    if verbose > 0:
        print("{:<25} {:^25} {:>25}".format('TOT', 'IMG', 'TXT'))
        for ida, artv in dico_id_artv.items():
            txt = artv[0][tot] - artv[0][img]
            print(f"{artv[0][tot]:<25} {artv[0][img]:^25} {txt:>25}")

    # Filtering on the area of the images.
    if verbose > 0:
        print("L'aire des images du carton : {}".format(aire_img_cart))

    fcn = lambda x: ((1 - tol_area_images)*aire_img_cart <= x[1][0][img] \
                     <= (1 + tol_area_images)*aire_img_cart)

    arts_left_img = [ida for ida, _ in filter(fcn, list(dico_id_artv.items()))]

    if verbose > 0:
        print(f"les ids des articles qui correspondent : {arts_left_img}")

    dico_id_artv_img = {ida: dicoa for ida, dicoa in dico_id_artv.items()
                        if ida in arts_left_img}

    if len(arts_left_img) == 0:
        str_exc = 'No article for that carton with the filter on the area '
        str_exc += 'of the images.'
        raise MyException(str_exc)

    # Filtering on the area of the text.
    diff = aire_tot_cart - aire_img_cart
    if verbose > 0:
        print("L'aire du txt du carton : {}".format(diff))

    fcn = lambda x: ((1 - tol_area_text)*diff <=  x[1][0][tot] - x[1][0][img] \
                     <= (1 + tol_area_text)*diff)

    list_items_dico_idartv = list(dico_id_artv_img.items())
    arts_left_txt = [ida for ida, _ in filter(fcn, list_items_dico_idartv)]

    if verbose > 0:
        print(f"les ids des articles qui correspondent : {arts_left_txt}")

    dico_id_artv_txt = {ida: dicoa for ida, dicoa in dico_id_artv_img.items()
                        if ida in arts_left_txt}

    if len(arts_left_txt) == 0:
        str_exc = 'No article for that carton with the filter on the area '
        str_exc += 'of the text.'
        raise MyException(str_exc)

    # Filtering on the total area.
    if verbose > 0:
        print("L'aire du carton : {}".format(aire_tot_cart))

    fcn = lambda x: ((1 - tol_total_area)*aire_tot_cart <= x[1][0][tot] \
                     <= (1 + tol_total_area)*aire_tot_cart)
    arts_left = [ida for ida, _ in filter(fcn, list(dico_id_artv_txt.items()))]

    if verbose > 0:
        print("les ids des articles qui correspondent : {}".format(arts_left))

    if len(arts_left) == 0:
        str_exc = 'No art. for that carton with the filter on the total area'
        raise MyException(str_exc)

    return arts_left


def ProductList(l1, l2, l3=None, l4=None, l5=None, l6=None, l7=None):
    """

    Parameters
    ----------
    l1: list
        Any types of list.

    l2: list
        Any types of list.

    l3: list, optional
        Any types of list.. The default is None.

    l4: list, optional
        Any types of list.. The default is None.

    l5: list, optional
        Any types of list.. The default is None.

    l6: list, optional
        Any types of list.. The default is None.

    l7: list, optional
        Any types of list.. The default is None.

    Returns
    -------
    list
        The product of all the lists passes in arguments.
        Example: l1 = [1, 2, 3], l2 = [4, 5, 6]
        -> returns: [[1, 4], [1, 5], [1, 6], ..., [3, 5], [3, 6]]

    """
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

    Parameters
    ----------
    dico_id_artv: dict
        dico_id_artv = {id_article: vector_article, ...}.

    dico_cartons: dict
        dico_cartons = {id_carton: {'nbImg': ..., ...}, ...}.

    id_art: float
        The id of the article.

    id_carton: float
        The id of the module.

    ind_features: list
        ind_features = [ind_nb_sign, ...].

    Returns
    -------
    distance: float
        The euclidean distance between the number of signs of the article and
        the relative area(because with remove area of the images) of the
        module.

    """
    nb_img = dico_cartons[id_carton]['nbImg']
    width = dico_cartons[id_carton]['width']
    height = dico_cartons[id_carton]['height']
    aire_carton = width * height
    # The link area - nbSign is roughly: nbSign ~ area/10
    aire_relative = aire_carton - (5000*nb_img)/10
    try:
         good_id_art = int(id_art)
    except Exception as e:
        print(f"Error with int(id_art): {e}.\n")
        print("type(id_art): {}. id_art: {}".format(type(id_art), id_art))
        good_id_art = id_art
    # A EVENTUELLEMENT CHANGER. Ajout argument ind_nb_sign ?
    ind_nb_sign = ind_features[0]
    nb_sign_article = dico_id_artv[good_id_art][0][ind_nb_sign]
    distance = np.sqrt((nb_sign_article - aire_relative) ** 2)
    return distance


def GetGoodVentMethNaive(every_possib_score):
    """
    Removes the pages with an article that appears in several modules.

    Parameters
    ----------
    every_possib_score: list of tuples
        every_possib_score = [(score, [id_art1, id_art2, id_art3, ...]), ...].

    Returns
    -------
    vents: list of tuples
        vents = [(score, [id1, id2, id3]), ...] for a pages with 3 articles.

    """
    nb_art_in_page = len(every_possib_score[0][1])
    f_uni = lambda x: len(set(x[1])) == nb_art_in_page
    ventilations_unique = list(filter(f_uni, every_possib_score))

    # Construction of a dict: {list_articles_page: score_page, ...}.
    dico_vent_score = {}
    for score, liste_id_art in ventilations_unique:
        if tuple(liste_id_art) not in dico_vent_score.keys():
            dico_vent_score[tuple(liste_id_art)] = score
        else:
            best_score = min(score, dico_vent_score[tuple(liste_id_art)])
            dico_vent_score[tuple(liste_id_art)] = best_score

    # We delete duplicates of the pages results
    vents = [(score, vent) for vent, score in dico_vent_score.items()]
    vents.sort(key=itemgetter(0))
    return vents


def SimpleLabels(liste_labels):
    """
    Reduce the number of labels by aggregating some labels.

    Parameters
    ----------
    liste_labels: list of strings
        liste_labels = ['TEXTE AUTEUR', 'INTERTITRE', ...].

    Returns
    -------
    simpler_list: list of strings
        simpler_list = ['TEXTE', 'PHOTO', ...].

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


def FiltreBlocs(dict_mdp_input, list_articles_input, dico_cartons,
                emp, verbose):
    """
    Returns the articles whose blocks labels correspond to the ones of the
    modules.

    Parameters
    ----------
    dict_mdp_input: dict
        dict_mdp_input = {'3764': {'x': 15, 'y': 323, 'width': 280,
                                   'height': 126, 'nbCol': 6, 'nbImg': 1,
                                   'aireTotImgs': 8494,
                                   'listeAireImgs': [8494], 'aireTot': 44607,
                                   'blocs': [{'labels':, ...}, {}, ...]}.

    list_articles_input: list of dicts
        list_articles_input = [dict_article_input1, dict_article_input2, ...].

    dico_cartons: dict
        dico_cartons = {emp: dict_module, ...}.

    emp: int
        The integer associated to the module (back to the function
        GetDicoCarton).

    verbose: int
        whether to print something or not.

    Raises
    ------
    MyException
        If we don't find back the module in dict_mdp_input.

    Returns
    -------
    liste_arts_poss: list
        The list with the ids of the articles that respect the constraints of
        the blocks.

    """
    liste_arts_poss = []
    # Déjà, il faut que je récupère le dico du carton
    x_cart, y_cart = dico_cartons[emp]['x'], dico_cartons[emp]['y']
    f = lambda x: (np.allclose(x[1]['x'], x_cart, atol=10)) & \
        (np.allclose(x[1]['y'], y_cart, atol=10))
    carton_found = dict(filter(f, dict_mdp_input.items()))

    if len(carton_found) != 1:
        str_exc = 'Error in FiltreBlocs. len(carton_found) != 1 \n'
        str_exc += 'carton found: {}\n'.format(carton_found)
        str_exc += '(x_cart, y_cart): ({}, {}) \n'.format(x_cart, y_cart)
        str_exc += 'dict_mdp_input: {}\n'
        for id_mdp, dict_mdp in dict_mdp_input.items():
            str_exc += f"{id_mdp}: x={dict_mdp['x']}, y={dict_mdp['y']} \n"
        raise MyException(str_exc)

    id_carton = list(carton_found.keys())[0]

    # On parcourt les articles, et on compare les éléments blocs
    for dicoa in list_articles_input:
        # On transforme les éléments blocs en une liste de labels plus simple
        blocs_label_art = [dicob['label'] for dicob in dicoa['blocs']]
        blocs_label_carton = [dicob['label']
                              for dicob in carton_found[id_carton]['blocs']]
        labels_art = SimpleLabels(blocs_label_art)
        labels_carton = SimpleLabels(blocs_label_carton)

        if verbose > 0:
            print(f"Les labels simples des articles : {labels_art}")
            print(f"Les labels simples des cartons : {labels_carton}")

        # If all blocks of the article are also present in the carton, we can
        # place this article in that carton.
        if set(labels_art) <= set(labels_carton):
            liste_arts_poss.append(dicoa['melodyId'])

    return liste_arts_poss


def MethodeNaive(mdp_ref,
                 X_nbImg,
                 X_input,
                 dico_id_artv,
                 ind_features,
                 dict_mdp_input,
                 list_articles_input,
                 tol_area_images_mod,
                 tol_area_text_mod,
                 tol_total_area_mod,
                 verbose):
    """

    Returns the pages such that the articles has the best fit with the modules.

    Parameters
    ----------
    mdp_ref: numpy array
        The matrix associated to the layout input.

    X_nbImg: list of lists
        Contains the info about the modules of the layout input.
        X_nbImg = [[int(dic_carton[x]) for x in list_feat]
                   for dic_carton in dict_mdp_input.values()]

    X_input: numpy array
        The matrix with all the articles input.
        dim(X_input) = (nb_articles, len(list_features))

    dico_id_artv: dict
        dico_id_artv = {id_article: vector_article, ...}.

    ind_features: list
        ind_features = [ind_nbSign, ind_aireTot, ind_aireImg]

    dict_mdp_input: dict
        dict_mdp_input = {id_module: dict_module, ...}.

    list_articles_input: list of dicts
        List with the dictionaries of the articles input.

    tol_area_images_mod: float
        The tolerance between the area of the images of the article and the
        module. (advice: 0.5)

    tol_area_text_mod: float
        The tolerance between the area of the text of the article and the
        module. (advice: 0.4)

    tol_total_area_mod: float
        The tolerance between the area of the article and of the module.
        (advice: 0.3)

    verbose: int >=0
        If > 0, print intermediate results.

    Raises
    ------
    MyException
        If FiltreArticle doesn't work or if no possibility.

    Returns
    -------
    new_dico_possib_placements: dict
        Similar to dico_possib_placements but without the duplicates.
        new_dico_possib_placements = {(x, y, w, h): [id_article_1, ...], ...}

    ventilations_unique: list of tuples
        ventilations_unique = [(score, [id1, id2, id3]), ...] for a pages with
        3 articles..

    """
    dico_cartons = GetDicoCarton(mdp_ref, X_nbImg)
    # Création du dico dico_possib_placements
    dico_possib_placements = {id_emp: [] for id_emp in dico_cartons.keys()}

    for emp, liste_art_poss in dico_possib_placements.items():
        # CORRESPONDANCE DES BLOCS
        args_filtre = [dict_mdp_input, list_articles_input, dico_cartons, emp]
        liste_art_poss += FiltreBlocs(*args_filtre, verbose)

        if verbose > 0:
            str_print = "The list of possible arts after the bloc filter"
            print(str_print + " : {}".format(liste_art_poss))

        # Pour chaque article, on regarde s'il est insérable à cet emplacement
        aire_img_carton = dico_cartons[emp]['aireImg']
        aire_tot_carton = dico_cartons[emp]['aireTot']

        ## CORRESPONDANCE ENTRE AIRE IMG, AIRE TXT, AIRE TOTALE
        args_filt = [dico_id_artv, aire_tot_carton, aire_img_carton]
        args_filt += [ind_features, verbose, tol_area_images_mod]
        args_filt += [tol_area_text_mod, tol_total_area_mod]
        try:
            if verbose > 0:
                print("{:-^75}".format("FILTRE ART"))
                print("L'aire des imgs de ce MDP : {}".format(aire_img_carton))
                bornes = (.5 * aire_img_carton, 1.5 * aire_img_carton)
                print("bornes : {} et {}".format(*bornes))
            liste_art_poss += FiltreArticles(*args_filt)
        except Exception as e:
            first_item = list(dico_id_artv.items())[0]
            str_exc = "Error with FiltreArticles : {}\n".format(e)
            str_exc += "ind_features: {}\n".format(ind_features)
            str_exc += "verbose: {}\n".format(verbose)
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
    """
    Filtering of the results of the method MLV with the results of the method
    NAIVE.

    Parameters
    ----------
    dico_possib_naive: dict
        dico_possib_naive = {id_module: list_articles_naive, ...}.

    dico_possib_mlv: dict
        dico_possib_mlv = {id_module: list_articles_mlv, ...}.

    Raises
    ------
    MyException
        If there is no possibility.

    Returns
    -------
    dict_mlv_filtered: dict
        dict_mlv_filtered = {id_module: list_articles_filtered, ...}.

    """
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
