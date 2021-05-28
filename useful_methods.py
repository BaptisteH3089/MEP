#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from operator import itemgetter


class MyException(Exception):
    pass


def GetYClasses(big_y):
    n = big_y.shape[0]
    return [(big_y[i], i) for i in range(n)]


def TransformArrayIntoClasses(big_y, list_classes):
    n = big_y.shape[0]
    little_y = np.zeros(n)
    for i in range(n):
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
    vect_art est un numpy array de dimension 2
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
    """
    # Transformation de big_y into simpler classes
    classes = GetYClasses(y_ref)
    for cpt, id_page in enumerate(list_id_pages):
        # Formation de la big matrice X pour toute la page
        # IL FAUDRAIT FORMER Y EN MM TPS POUR EVITER TOUTE ERREUR DE
        # CORRESPONDANCE
        x_all_articles, y_all_articles = np.zeros(0), np.zeros(0)
        dico_arts_page = dico_bdd[id_page]['dico_page']['articles']
        for i, (id_art, dico_art) in enumerate(dico_arts_page.items()):
            melo_id, vect_art = GetVectorArticleInput(dico_art, features)
            y_art = np.array([[dico_art['x'], dico_art['y'],
                              dico_art['width'], dico_art['height']]])
            if i == 0:
                x_all_articles = np.array(vect_art, ndmin=2)
                y_all_articles = y_art
            else:
                # On concatène le nouveau vect avec x_all_articles
                try:
                    x_all_articles = np.concatenate((x_all_articles, vect_art))
                except Exception as e:
                    print("Error with np.concatenate((x_all_articles, vect_art))")
                    print(e)
                    print("x_all_articles", x_all_articles)
                    print("vect_art", vect_art)
                y_all_articles = np.concatenate((y_all_articles, y_art))
        try:
            y = TransformArrayIntoClasses(y_all_articles, classes)
        except Exception as e:
            print("Error with TransformArrayIntoClasses", e)
            print("y_all_articles", y_all_articles)
            print("classes", classes)
            raise e
        if cpt == 0:
            very_big_x = x_all_articles
            very_big_y = y
        else:
            very_big_x = np.concatenate((very_big_x, x_all_articles))
            very_big_y = np.concatenate((very_big_y, y))
    return very_big_x, very_big_y


def GetPossibilitiesMLV(probas_preds, X_rand, dico_id_vect_art, verbose=False):
    """
    Renvoie un dictionnaire avec pour chaque emp les articles qui ont obtenus
    une proba prédite supérieure à 40%
    """
    dico_possib_emplacements = {i: [] for i in range(probas_preds.shape[1])}
    for i in range(len(probas_preds)):
        preds_art = probas_preds[i]

        if verbose is True:
            string_affichage = ""
            for pred in preds_art:
                string_affichage += "{:^8.2f}"
            print(("art. {}" + string_affichage).format(i, *preds_art))

        # Il faut récupérer l'id de l'article
        # vect_art = X_rand[i][[0, 1, -2, -1]]
        vect_art = X_rand[i]

        #print("vect_art {}".format(vect_art))
        #print("dico_id_vect_art.items(): {}".format(dico_id_vect_art.items()))

        f = lambda x: np.allclose(x[1], vect_art, atol=2)
        res_id_artx = list(filter(f, dico_id_vect_art.items()))

        if verbose is True:
            print("res_id_artx {}".format(res_id_artx))

        id_x_rand_i = res_id_artx[0][0]
        # On conserve les classes avec proba > 40%
        liste_tuple = [(i, pred) for i, pred in enumerate(preds_art)]
        
        if verbose is True:
            print("liste_tuple {}".format(liste_tuple))

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
               dico_id_vect_art):
    """
    Retourne les possibilités de placements d'articles dans un MDP en se
    basant sur les prédictions d'un RFC entraîné sur les pages construites
    avec le même MDP.
    """
    # Entraînenemt du modèle
    list_of_features = ['nbSign', 'nbBlock', 'abstract', 'syn']
    list_of_features += ['exergue', 'title', 'secTitle', 'supTitle']
    list_of_features += ['subTitle', 'nbPhoto', 'aireImg']
    X, Y = GetTrainableData(dico_bdd, liste_id_pages, mdp_ref, list_of_features)
    # print("X.shape: {}".format(X.shape))
    # print("Y.shape: {}".format(Y.shape))
    param_model = {'max_depth': 5,
                   'min_samples_split': 4,
                   'min_samples_leaf': 4,
                   'max_features': 3,
                   'max_leaf_nodes': 10}
    clf = RandomForestClassifier(**param_model)
    try:
        clf.fit(X, Y)
    except Exception as e:
        print("Error with fit", e)
        print(X.shape)
        print(Y.shape)
        print("X", X)
        print("Y", Y)
    try:
        probas_preds = clf.predict_proba(X_input)
    except Exception as e:
        print("Error with probas preds", e)
        print('X_input.shape', X_input.shape)
        print(X_input)

    # Obtention de possibilités de placement des articles
    l_args = [probas_preds, X_input, dico_id_vect_art]
    dico_poss_plcmts = GetPossibilitiesMLV(*l_args)

    # Remplacer les emplacements 0, ..., 3 par les vrais tuples EMP
    classes = GetYClasses(mdp_ref)
    dico_plcmts_tuple = {}
    for emp, possib_art in dico_poss_plcmts.items():
        array_emp = list(filter(lambda x: x[1] == emp, classes))[0][0]
        dico_plcmts_tuple[tuple(array_emp)] = possib_art

    size_each_location = [len(poss_art)
                          for poss_art in dico_poss_plcmts.values()]

    return dico_plcmts_tuple


def GetDicoCarton(mdp_ref, X_col_img):
    """
    Création d'un dico avec chaque carton et les caractéristiques dedans.
    """
    keys_model = ['x', 'y', 'width', 'height', 'nbCol', 'nbImg']
    dico_cartons = {}
    for i, (carton, nb_col_img) in enumerate(zip(mdp_ref, X_col_img)):
        dico_cartons[i] = {}
        big_carton = list(carton) + list(nb_col_img)
        for j, (caract, key) in enumerate(zip(big_carton, keys_model)):
            dico_cartons[i][key] = caract
    return dico_cartons


def FiltreArticles(dico_id_artv, nb_img, area, ind_features):
    """
    ind_features correspond à une liste [ind_nb_sign, ind_nb_img, ind_area]
    - l'aire des images d'un art doit occuper moins de 2/3 de l'espace du MDP
    - on fait le filtre sur le nb de caract d'un art - aire des images
    convertie en nb de caract
    - si le MDP comprend des images on doit soustraire l'aire des images
    de l'article à celle de l'aire totale
    ATTENTION il faut penser à changer les indices de nb_img et area si
    changement dans les features
    """
    ind_nb_sign = ind_features[0]
    ind_nb_img = ind_features[1]
    ind_area = ind_features[2]
    fcn = lambda x: (x[1][0][ind_nb_img] == nb_img) & \
                    (x[1][0][ind_area] < 0.66 * area) & \
                    ((area - x[1][0][ind_area]) / 15 < x[1][0][ind_nb_sign] \
                     < (area - x[1][0][ind_area]) / 5)

    # print("\nFiltrage des articles pour {} images et une "
    #       "aire de {} ({:.0f} {:.0f})".format(nb_img, area, area/15, area/5))
    res_found = [ida for ida, _ in filter(fcn, list(dico_id_artv.items()))]

    # for i, id_art in enumerate(res_found):
        # print("res found {} {:>20}".format(i, str(dico_id_artv[id_art])))

    return res_found


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


def ScoreArticle(dico_id_artv, dico_cartons, id_art, id_carton):
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
    except:
        good_id_art = id_art
    # A EVENTUELLEMENT CHANGER
    nb_sign_article = dico_id_artv[good_id_art][0][0]
    distance = np.sqrt((nb_sign_article - aire_relative) ** 2)
    return distance


def GetGoodVentMethNaive(every_possib_score):
    """
    Le problème dans l'objet précédent est qu'un article peut apparaître à 2
    endroits s'il peut être placé à ces endroits, donc on filtre
    """
    nb_art_in_page = len(every_possib_score[0][1])
    f_uni = lambda x: len(set(x[1])) == nb_art_in_page
    ventilations_unique = list(filter(f_uni, every_possib_score))
    dico_vent_score = {}
    for score, liste_id_art in ventilations_unique:
        if tuple(liste_id_art) not in dico_vent_score.keys():
            dico_vent_score[tuple(liste_id_art)] = score
        else:
            best_score = min(score, ico_vent_score[tuple(liste_id_art)])
            dico_vent_score[tuple(liste_id_art)] = best_score
    # Suppression des doublons des ventilations
    vents = [(score, vent) for vent, score in dico_vent_score.items()]
    vents.sort(key=itemgetter(0))
    return vents


def MethodeNaive(mdp_ref, X_nbImg, X_input, dico_id_artv, colors,
                 ind_features, show=False, verbose=False):
    """
    Renvoie des ventilations en se basant sur le nb d'images dans les cartons
    et l'aire du carton par rapport au nombre de signes des articles et à
    l'aire totale des images. 
    Pour faire simple, donne toutes les possibilités de placements tq les
    articles correspondent à peu près aux cartons
    """
    # Dico pour attribuer les couleurs aux articles
    nb_art = len(dico_id_artv)
    if nb_art > len(colors):
        colors_alea = np.random.choice(colors, size=nb_art, replace=True)
    else:
        colors_alea = np.random.choice(colors, size=nb_art, replace=False)
    dico_colors = {ar: c for ar, c in zip(dico_id_artv.keys(), colors_alea)}

    dico_cartons = GetDicoCarton(mdp_ref, X_nbImg)

    # Création du dico dico_possib_placements
    dico_possib_placements = {id_emp: [] for id_emp in dico_cartons.keys()}
    
    for emp, liste_art_poss in dico_possib_placements.items():
        # Pour chaque article, on regarde s'il est insérable à cet emplacement
        nb_img = dico_cartons[emp]['nbImg']
        area = dico_cartons[emp]['width'] * dico_cartons[emp]['height']
        try:
            liste_art_poss += FiltreArticles(dico_id_artv, nb_img, area, ind_features)
        except Exception as e:
            print("Error with FiltreArticles : ", e)
            print("ind_features", ind_features)
            print("nb_img", nb_img)
            print("area", area)
            print("dico_id_artv")
            for ida, art in dico_id_artv.items():
                print("{} {}".format(ida, art))

    # TRICHE
    triche = False
    if triche is True:
        for id_emp, liste_id in dico_possib_placements.items():
            if len(liste_id) == 0:
                print("\n\nCertain(s) emp(s) sont vides\n\n")
                rd_index = np.random.choice(list(dico_id_artv.keys()),
                                            size=2,
                                            replace=False)
                dico_possib_placements[id_emp] += list(rd_index)
                print("Ajout de rdm indexes", dico_possib_placements[id_emp])

    if verbose is True:
        print("\nThe dico with all possibilities\n")
        for id_emp, possib in dico_possib_placements.items():
            print("{:<15} {}".format(id_emp, possib))

    liste_len = [len(liste_id) for liste_id in dico_possib_placements.values()]

    if 0 in liste_len:
        str_exc = "No possibility for at least one location: {}"
        raise MyException(str_exc.format(liste_len))

    l_args = [emp_poss for emp_poss in dico_possib_placements.values()]
    every_possib = ProductList(*l_args)

    # Caclul des scores de chaque ventilations
    # every_possib = [ [id1, id2, id3], ...]
    # retourne [ ([id1, id2, id3], score_vent), ...]
    every_possib_score = []
    for vent in every_possib:
        score_vent = 0
        for id_art, id_emp in zip(vent, dico_possib_placements.keys()):
            args = [dico_id_artv, dico_cartons, id_art, id_emp]
            score_vent += ScoreArticle(*args)
        try:
            every_possib_score.append((int(score_vent), vent))
        except Exception as e:
            print("Error with every_possib_score.append()", e)
            print("score_vent", score_vent)
            print("vent", vent)

    # ventilations triées
    ventilations_unique = GetGoodVentMethNaive(every_possib_score)

    # Il faut remplacer les ids des emplacements par les 4-uplets
    new_dico_possib_placements = {}
    for id_emp, liste_id_articles in dico_possib_placements.items():
        tuple_emp = (dico_cartons[id_emp]['x'],
                     dico_cartons[id_emp]['y'],
                     dico_cartons[id_emp]['width'],
                     dico_cartons[id_emp]['height'])
        new_dico_possib_placements[tuple_emp] = liste_id_articles

    return new_dico_possib_placements


def ResultsFiltered(dico_possib_naive, dico_possib_mlv):
    dict_mlv_filtered = {}
    for key_naive, key_mlv in zip(dico_possib_naive.keys(),
                                  dico_possib_mlv.keys()):
        f = lambda x: x[1] in dico_possib_naive[key_mlv]
        val_filtered = list(filter(f, dico_possib_mlv[key_mlv]))
        if len(val_filtered) == 0:
            raise MyException("After filtration with the predictions of the "
                              "mlv method, there is no possibility left")
        dict_mlv_filtered[key_mlv] = val_filtered
    return dict_mlv_filtered