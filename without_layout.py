#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Baptiste Hessel

Contains the functions relative to the part where there is no layout input.

"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import xml.etree.cElementTree as ET
from pathlib import Path
from operator import itemgetter
import time
import module_montage
import propositions
import methods
import module_art_type
from xgboost import XGBRegressor


class MyException(Exception):
    pass


def ConstraintArea(all_ntuple, verbose, tol_total_area=0.08):
    """

    Parameters
    ----------
    all_ntuple: list of tuples
        List with all the possibilities of pages with nb_arts in it.
        all_ntuple = [([dicta[x] for x in list_feat], ...), ...].

    verbose: int >= 0
        If verbose > 0, print info.

    tol_total_area: float (default=0.08)
        The pourcentage of tolerance for the constraint total area.

    Returns
    -------
    possib_area: list of tuples
        possib_area = [(id1, id2, id3), ...].
        The length of the tuples corresponds to the lengths of the tuples of
        all_ntuple.

    """
    # Now we select the ones that respect the constraint of the total area.
    # Don't know about this value mean_area
    mean_area = 95000
    possib_area = []
    all_sums = []
    for ntuple in all_ntuple:
        sum_art = sum((art[0] for art in ntuple))
        all_sums.append(sum_art)
        if (1 - tol_total_area)*mean_area <= sum_art \
            <= (1 + tol_total_area) * mean_area:
            # If constraint ok, we add the id of the article
            possib_area.append(tuple([art[-1] for art in ntuple]))

    if verbose > 0:
        print(f"Constraint area: {len(possib_area)} articles left.")
        print(f"Mean area: {np.mean(all_sums):1f}.")

    return possib_area


def ConstraintImgs(all_ntuple, verbose, tol_nb_images=1):
    """
    Selects the pages with at least 1 images in it.

    Parameters
    ----------
    all_ntuple: list of tuples
        List with all the possibilities of pages with nb_arts in it.
        all_ntuple = [([dicta[x] for x in list_feat], ...), ...].

    verbose: int >= 0
        If verbose > 0, print info.

    tol_nb_images: int (default=1)
        The tolerance for the minimum number of images in the page.

    Returns
    -------
    possib_imgs: list of tuples
        possib_area = [(id1, id2, id3), ...].
        The length of the tuples corresponds to the lengths of the tuples of
        all_ntuple.
        The ntuples are the combinations of articles that respect the
        constraint "image".
    """
    possib_imgs = []
    all_nb_imgs = []
    for ntuple in all_ntuple:
        try:
            nb_imgs = sum((art[1] for art in ntuple))
        except Exception as e:
            if verbose > 0:
                str_prt = (f"Error with the sum of the nbImg: {e}.\n"
                           f"ntuple: {ntuple}.\nnb_imgs: {nb_imgs}")
                print(str_prt)

        all_nb_imgs.append(nb_imgs)

        if nb_imgs >= tol_nb_images:
            possib_imgs.append(tuple([art[-1] for art in ntuple]))

    if verbose > 0:
        print(f"Constraint images {len(possib_imgs)} articles left.")
        print(f"Mean nb of images: {np.mean(all_nb_imgs)}.")

    return possib_imgs


def ConstraintScore(all_ntuple, verbose, tol_score_min=0.20):
    """
    Selects the pages with at least an article with a score > 0.35

    Parameters
    ----------
    all_ntuple: list of tuples
        List with all the possibilities of pages with nb_arts in it.
        all_ntuple = [([dicta[x] for x in list_feat], ...), ...].

    verbose: int
        If verbose > 0, print info.

    Returns
    -------
    possib_score: list of tuples
        possib_score = [(idA, idB, ...), ...].
        The ntuples in possib_score correspond to the ones that satisfy the
        constraint "score".

    """
    possib_score, all_max_score = [], []
    for ntuple in all_ntuple:
        max_score = max((art[2] for art in ntuple))
        all_max_score.append(max_score)
        if max_score >= tol_score_min:
            possib_score.append(tuple([art[-1] for art in ntuple]))

    if verbose > 0:
        print("{} {}.".format("Constraint score", len(possib_score)))
        print("{} {}.".format("Mean score", np.mean(all_max_score)))

    return possib_score


##############################################################################
#                                                                            #
#           ALL THE FUNCTIONS NECESSARY FOR CreateXYFromScratch              #
#                                                                            #
##############################################################################


def SelectionPagesNarts(dict_pages, nb_arts):
    """
    Selects in the dict_pages the pages with "nb_arts" articles.

    Parameters
    ----------
    dict_pages: dict of dict
        The usual dict with all the infos on the pages.

    nb_arts: int
        The number of articles in the page.

    Returns
    -------
    dico_narts: dict of dict
        Of the same form as dict_pages but with only pages with "nb_arts"
        articles.
        dico_narts = {id_page: dico_articles}.

    """
    dico_narts = {}
    for key in dict_pages.keys():
        if len(dict_pages[key]['articles']) == nb_arts:
            dico_narts[key] = dict_pages[key]['articles']
    return dico_narts


def SelectionMDPNarts(list_mdp_data, nb_modules):
    """
    Selects the layouts with "nb_modules" modules.

    Parameters
    ----------
    list_mdp_data: list of tuples
        list_mdp_data = [(nb, array, list_ids), ...].

    nb_modules: int
        the desired number of modules.

    Returns
    -------
    list_mdp_narts: list of lists
        list_mdp_narts = [[array_layout, list_ids], ...].

    """
    list_mdp_narts = []
    for nb, mdp, list_ids in list_mdp_data:
        if mdp.shape[0] == nb_modules:
            list_mdp_narts.append([mdp, list_ids])
    return list_mdp_narts


def VerificationPageMDP(dico_narts, list_mdp_narts):
    """
    Checks if the ids of pages in the object "list_mdp_narts" are also in the
    list "list_mdp_narts" and consequently associated to a layout.

    Parameters
    ----------
    dico_narts: dict of dict
        dico_narts = {id_page: dict_articles_page_with_narts, ...}.

    list_mdp_narts: list of lists
        list_mdp_narts = [[array_mdp_narts, list_ids], ...].

    Returns
    -------
    str
        Indicates the number of pages in our data.

    """
    list_all_matches = []
    for key in dico_narts.keys():
        list_match = [1 for _, list_ids in list_mdp_narts if key in list_ids]
        list_all_matches.append(sum(list_match))
    return f"The total nb of pages in our data: {len(dico_narts)}"


def CreationVectXY(dico_narts,
                   dict_arts,
                   list_mdp_narts,
                   list_features,
                   nb_pages_min,
                   verbose):
    """
    Creates the matrices X and Y with the vect_pages and the labels of the
    layouts. We keep only layouts with at least "np_pages_min" pages using it.

    Parameters
    ----------
    dico_narts: dict
        The dict with only the pages with the wanted number of articles.
        dico_narts = {id_page: dict_articles, ...}

    dict_arts: dict of dict
        dict_arts = {id_article: dict_article, ...}.

    list_mdp_narts: list with only the mdp with the good number of modules.
        list_mdp_narts = [[array_layout, list_ids], ...].

    list_features: list of strings
        The list with the features used to create X.

    nb_pages_min: int
        The minimum number of pages per layout.

    verbose: int >= 0
        If verbose > 0, print info.

    Returns
    -------
    vect_XY: list of tuples
        vect_XY = [(vect_page, array_layout), ...].

    """
    if verbose > 0:
        print(f"CreationVectXY. Initial len of dico_narts: {len(dico_narts)}")

    vect_XY = []
    for key in dico_narts.keys():
        for i, (ida, dicoa) in enumerate(dico_narts[key].items()):
            vect_art = [dict_arts[ida][x] for x in list_features]
            vect_art = np.array(vect_art, ndmin=2)
            if i == 0:
                vect_page = vect_art
            else:
                vect_page = np.concatenate((vect_page, vect_art), axis=1)
        for mdp, list_ids in list_mdp_narts:
            if len(list_ids) >= nb_pages_min:
                if key in list_ids:
                    vect_XY.append((vect_page, mdp))
                    break
    if verbose > 0:
        str_prt = (f"With a min number of {nb_pages_min} pages per layout, we "
                   f"keep {len(vect_XY)} pages to train the model.")
        print(str_prt)

    return vect_XY


# Now, we create a dico with the mdp
def CreationDictLabels(vect_XY, verbose):
    """
    Returns {0: np.array([...]),
             1: np.array([...]),
             ...}
    The values correspond to the layouts.

    Parameters
    ----------
    vect_XY: list of tuples
        vect_XY = [(array_vect_page, array_layout), ...].

    verbose: int >= 0
        If verbose > 0, print info.

    Returns
    -------
    dict_labels: dict
        dict_labels = {label_array_layout: array_layout, ...}.
        label_array_layout is just an integer starting from 0 and increasing
        by one unit.

    """

    label = 0
    dict_labels = {}
    for _, mdp in vect_XY:
        find = False
        for mdp_val in dict_labels.values():
            if np.allclose(mdp, mdp_val):
                find = True
        if find is False:
            dict_labels[label] = mdp
            label += 1
    if verbose > 0:
        print("{} {}.".format("The nb of labels in that dict", label + 1))
    return dict_labels


def CreationXY(vect_XY, dict_labels):
    """
    Creation of the matrix X and Y used to train a model.

    Parameters
    ----------
    vect_XY: list of tuples
        vect_XY = [(array_vect_page, array_layout), ...].

    dict_labels: dict
        dict_labels = {0: array_layout_0, 1: array_layout_1, ...}.

    Returns
    -------
    numpy array
        big_X: The array with the vectors page concatenated.
    numpy array
        better_big_Y: The array with the labels of the layouts.

    """
    if len(vect_XY) < 5:
        return [], []
    for i, (vect_X, vect_Y) in enumerate(vect_XY):
        if i == 0:
            big_X = vect_X
            for label, vect_label in dict_labels.items():
                if np.allclose(vect_label, vect_Y):
                    big_Y = [label]
        else:
            big_X = np.concatenate((big_X, vect_X))
            for label, vect_label in dict_labels.items():
                if np.allclose(vect_label, vect_Y):
                    try:
                        big_Y.append(label)
                    except:
                        print("ERROR")
                        print(big_Y, label)
    # For now, there is an issue with big_Y since we can have something like:
    # set(big_Y) = {0, 1, 4, 5, 8}. I want to have set(big_Y)={0, 1, 2, 3, 4}.
    dict_corres_labels = {label: i for i, label in enumerate(set(big_Y))}
    better_big_Y = np.array([dict_corres_labels[label] for label in big_Y])
    return big_X, better_big_Y


def CreateXYFromScratch(dict_pages,
                        dict_arts,
                        list_mdp,
                        nb_arts,
                        nbpg_min,
                        list_feats,
                        verbose):
    """
    Returns the matrices X and Y such that X is composed of pages with
    "nb_arts" articles in it and the pages should use layouts that appear more
    than "nbpg_min" times in the data.

    Parameters
    ----------
    dict_pages: dict of dict
        The usual dict with all the data.

    dict_arts: dict
        The dict with the characteristics of the articles.
        dict_arts = {id_article: dict_article, ...}

    list_mdp: list of tuples
        list_mdp = [(nb, array, list_ids), ...].

    nb_arts: int
        The wanted number of articles in the pages used to train.

    nbpg_min: int
        The min number of pages with each of the layouts in Y.

    list_feats: list of strings
        The features used to build the vectors page.

    verbose: int >= 0
        If verbose > 0, print info.

    Returns
    -------
    X: numpy array
        Concatenation of vectors page. The number of columns = nb of features.

    Y: numpy array
        Vector column with the labels of the layouts.

    dict_labels: dict
        Dict that gives the correspondence between label_layout and
        array_layout.

    """
    dico_narts = SelectionPagesNarts(dict_pages, nb_arts)
    list_mdp_narts = SelectionMDPNarts(list_mdp, nb_arts)

    args_xy = [dico_narts, dict_arts, list_mdp_narts, list_feats, nbpg_min]
    vect_XY = CreationVectXY(*args_xy, verbose)
    dict_labels = CreationDictLabels(vect_XY, verbose)

    try:
        X, Y = CreationXY(vect_XY, dict_labels)
    except Exception as e:
        str_e = f"An error with without_layout:CreateXYFromScratch:CreationXY"
        str_e += f"\n{e}\n"
        str_e += f"vect_XY: \n{vect_XY}\n dict_labels: \n{dict_labels}"
        raise MyException(str_e)

    return X, Y, dict_labels


##############################################################################
#                                                                            #
#        END OF ALL THE FUNCTIONS NECESSARY FOR CreateXYFromScratch          #
#                                                                            #
##############################################################################


def PredsModel(X, Y, list_vect_page, trained_gbc, verbose):
    """

    Parameters
    ----------
    X: numpy array
        The matrix with the vectors page.

    Y: numpy array
        Vector column with the labels.

    list_vect_page: list of numpy array
        list_vect_page = [vect_page0, ...].

    trained_gbc: scikit learn object
        A trained Gradient Boosting Classifier.

    verbose: int >= 0
        If verbose > 0, print info.

    Returns
    -------
    preds_gbc: array
        The predicted probas for each vector page. The elements (i, j) of
        preds_gbc corresponds to the likelyhood of the layouts j for the
        vector page i.

    """

    for i, vect_page in enumerate(list_vect_page):
        if i == 0:
            matrix_input = vect_page
        else:
            matrix_input = np.concatenate((matrix_input, vect_page))

    if verbose > 0:
        str_prt = "The shape of the matrix input"
        print("{} {}".format(str_prt, str(matrix_input.shape)))

    preds_gbc = trained_gbc.predict_proba(matrix_input)
    preds = trained_gbc.predict(matrix_input)

    if verbose > 0:
        print("\n{:^80}".format("The variety of the predictions (classes, count)"))
        list_sh = [(x, list(preds).count(x)) for x in set(list(preds))]
        print("{:^80}\n".format(str(list_sh)))

    return preds_gbc


def CreationDDArt(rep_data):
    """
    Creates a dict of dictionaries with the articles input extracted from the
    archive.

    Parameters
    ----------
    rep_data: str
        The folder with the articles input

    Returns
    -------
    dict_arts_input: dict
        A dictionary of the form {ida: dicoa, ...}

    """
    dict_arts_input = {}
    rep_articles = rep_data + '/articles'

    for file_path in Path(rep_articles).glob('*.xml'):
        try:
            dict_art = propositions.ExtractDicoArtInput(file_path)
            dict_arts_input[dict_art['melodyId']] = dict_art
        except Exception as e:
            str_exc = (f"An error with an article input: {e}.\n"
                       f"The file_path: {file_path}")
            raise MyException(str_exc)

    return dict_arts_input


def GenerationAllNTupleFromFiles(dict_arts_input, nb_arts, verbose):
    """

    Parameters
    ----------
    dict_arts_input: dict
        A dictionary with all the articles input we have.

    nb_arts: int
        The number of articles in the pages we want to create.

    verbose: int >= 0
        If verbose > 0, print info.

    Raises
    ------
    MyException
        Raises Exception if wrong number of articles.

    Returns
    -------
    all_ntuple: list of tuples
        List with all the possibilities of pages with nb_arts in it.
        all_ntuple = [([dicta[x] for x in list_feat], ...), ...]

    """

    list_feat = ['aireTot', 'nbPhoto', 'score', 'melodyId']
    val_dict = dict_arts_input.values()
    select_arts = [[dicta[x] for x in list_feat] for dicta in val_dict]

    if verbose > 0:
        str_prt = 'The selection of arts [aireTot, nbPhoto, melodyId]'
        print(f"{str_prt:-^75}")
        for i, list_art in enumerate(select_arts):
            print("{:<15} {}".format(i + 1, list_art))
        print("{:-^75}".format("End of the selection of arts"))

    if nb_arts == 2:
        all_ntuple = [(x, y) for i, x in enumerate(select_arts)
                      for y in select_arts[i + 1:]]

    elif nb_arts == 3:
        all_ntuple = [(x, y, z) for i, x in enumerate(select_arts)
                      for j, y in enumerate(select_arts[i + 1:])
                      for z in select_arts[i + j + 2:]]

    elif nb_arts == 4:
        all_ntuple = [(a, b, c, d) for i, a in enumerate(select_arts)
                      for j, b in enumerate(select_arts[i + 1:])
                      for k, c in enumerate(select_arts[i + j + 2:])
                      for d in select_arts[i + j + k + 3:]]

    elif nb_arts == 5:
        all_ntuple = [(a, b, c, d, e) for i, a in enumerate(select_arts)
                      for j, b in enumerate(select_arts[i + 1:])
                      for k, c in enumerate(select_arts[i + j + 2:])
                      for l, d in enumerate(select_arts[i + j + k + 3:])
                      for e in select_arts[i + j + k + l + 4:]]

    else:
        str_exc = 'Wrong nb of arts: {}. Must be in [2, 5]'.format(nb_arts)
        raise MyException(str_exc)

    if verbose > 0:
        print(f"The total nb of {nb_arts}-tuple generated: {len(all_ntuple)}.")

    return all_ntuple


def GetSelectionOfArticlesFromFiles(all_ntuples,
                                    tol_total_area,
                                    tol_nb_images,
                                    tol_score_min,
                                    verbose):
    """

    Parameters
    ----------
    all_ntuples: list of tuples
        List with all the possibilities of pages with nb_arts in it.
        all_ntuples = [([dicta[x] for x in list_feat], ...), ...].

    tol_total_area: float
        The tolerance between the area of the module and of the article.
        between 0 and 1

    tol_nb_images: int
        The minimum number of images of the pages results.

    verbose: int >= 0
        If verbose > 0, print info.

    Returns
    -------
    final_obj: set
        A set with all the tuples containing th ids of articles that together
        respect the constraints.

    """
    # Use of the constraints
    possib_area = ConstraintArea(all_ntuples, verbose,
                                 tol_total_area=tol_total_area)
    possib_imgs = ConstraintImgs(all_ntuples, verbose,
                                 tol_nb_images=tol_nb_images)
    possib_scores = ConstraintScore(all_ntuples, verbose,
                                    tol_score_min=tol_score_min)
    # CONSTRAINT SCORE ??? -> I should add that
    # Intersection of each result obtained
    final_obj = set(possib_imgs) & set(possib_area) & set(possib_scores)

    if verbose > 0:
        str_prt = "The number of ntuples that respect the two constraints"
        print("{}: {}.".format(str_prt, len(final_obj)))

    return final_obj


def CreationListVectorsPageFromFiles(final_obj, artd_input, list_feat):
    """
    Uses the ids of pages in "final_obj" to create vectors article and to
    concatenate them to build vect page.

    Parameters
    ----------
    final_obj: set
        set of tuples with the ids of page.

    artd_input: dict
        The dict with the articles INPUT.

    list_feat: list of strings
        The features used to create .

    Returns
    -------
    list_vect_page: list of numpy array
        List with the numpy_array_vect_page.

    """
    list_vect_page = []
    for triplet in final_obj:
        for i, ida in enumerate(triplet):
            artv = np.array([artd_input[ida][x] for x in list_feat], ndmin=2)
            if i == 0:
                vect_page = artv
            else:
                vect_page = np.concatenate((vect_page, artv), axis=1)
        list_vect_page.append(vect_page)
    return list_vect_page


def SelectBestTriplet(all_triplets, list_features, verbose):
    """
    Used in the method with files.
    Computation of the variance of a proposition and then the final score of a
    triplet.

    Parameters
    ----------
    all_triplets: list of lists/tuple
        all_triplets = [[(score, vect_page), (s, v), (s, v)], ...].

    list_features: list of strings
        The list with the features used to create the vectors page.

    verbose: int >= 0
        If verbose > 0, print info.

    Returns
    -------
    best_triplet: tuple
        best_triplet = (score_prop, triplet).
        triplet = [(score, vect_page), (s, v), (s, v)]

    """
    vect_var = []
    for triplet in all_triplets:
        mean_vect = np.mean([vect_pg for _, vect_pg in triplet], axis=0)
        # Computation of the variance
        # Need to determine the area
        # Size of the vectors article: 15
        ind_aireTot = list_features.index('aireTot')
        size_vect_pg = mean_vect.shape[1]

        array_weights = triplet[0][1][0][range(ind_aireTot, size_vect_pg, 15)]
        weights_tmp = list(array_weights / sum(array_weights))

        weights = []
        for weight in weights_tmp:
            weights += [weight]*15
        if len(weights) != size_vect_pg:
            str_exc = (f"Something's wrong with the weights. \n"
                       f"weights\n {weights}.\n"
                       f"weights_tmp\n {weights_tmp}.\n"
                       f"array_weights\n {array_weights}.\n"
                       f"triplet: \n{triplet}.")
            raise MyException(str_exc)

        variance_triplet = 0
        for _, vect_page in triplet:
            diff_vect = np.array(weights) * (vect_page - mean_vect)
            variance_triplet += int(np.sqrt(np.dot(diff_vect, diff_vect.T)))
        vect_var.append(variance_triplet)

    # We standardize the vect_var
    std_var = np.std(vect_var)
    mean_var = np.mean(vect_var)
    stand_vect_var = (vect_var - mean_var) / std_var

    all_triplets_scores = []
    for triplet, var in zip(all_triplets, stand_vect_var):
        score_prop = sum((sc for sc, _ in triplet))
        all_triplets_scores.append((score_prop + var, triplet))

    # Ultimately, we take the triplet with the best score
    all_triplets_scores.sort(key=itemgetter(0), reverse=True)
    best_triplet = all_triplets_scores[0]

    if verbose > 0:
        str_triplet = "The best triplet has the score"
        print("{:<35} {:>35.2f}.".format(str_triplet, best_triplet[0]))
        print("{:-^80}".format("The score of each page"))
        for sc, _ in best_triplet[1]:
            print("{:40.4f}".format(sc))
        print("{:-^80}".format("The vectors page"))
        for _, page in best_triplet[1]:
            print("{:^40}".format(str(page)))

    return best_triplet


def GenerationOfAllTriplets(preds_gbc, list_vect_page):
    """
    Selects the max of each vector of predicted probas. It corresponds to the
    score of the page. It constructs an object with all the triplet of pages
    with the score of each page.

    Parameters
    ----------
    preds_gbc: numpy array
        The matrix with the predicted probas for each vector page and each
        layout.

    list_vect_page: list of numpy array
        The list with all the vectors page input.

    Returns
    -------
    all_triplets: list of lists
        all_triplets = [[(score_page1, vect_page1),
                         ...,
                         (score_page3, vect_page3)], ...].
        score_page is a float and vect_page a vector line of type numpy array.

    """
    pages_with_score = []
    for line, vect_pg in zip(preds_gbc, list_vect_page):
        pages_with_score.append((round(max(line), 4), vect_pg))
    # Generation of every triplet of pages
    all_triplets = [[p1, p2, p3] for i, p1 in enumerate(pages_with_score)
                    for j, p2 in enumerate(pages_with_score[i + 1:])
                    for p3 in pages_with_score[i + j +2:]]
    return all_triplets


def FillShortVectorPage(final_obj, list_features, dict_arts):
    """
    Computes a kind of standard vector page for each ntuple.

    Parameters
    ----------
    final_obj: set of tuples
        The set with all the possible combinations of articles to create pages.
        final_obj = {(id1, id2, id3), ...}.
        Since I treat the pages with a given number of articles one by one,
        the length of the tuples is the same.

    list_features: list of strings
        The list with the features used to train the model.

    dict_arts: dict
        The data dict with the characteristics of the articles.

    Returns
    -------
    list_short_vect_page: list of numpy array
        List with the sum vectors page associated to each possibility.

    """
    list_short_vect_page = []
    for ntuple in final_obj:
        # Line vector
        short_pagev = []
        # For each feature, we sum the values of all vects in the tuple.
        for feat in list_features:
            short_pagev.append(sum((dict_arts[ida][feat] for ida in ntuple)))
        short_pagev = np.array(short_pagev, ndmin=2)
        # We add that vector to the list
        list_short_vect_page.append(short_pagev)
    return list_short_vect_page


def ShowsBestTriplet(best_triplet):
    """
    Just shows nicely the best triplet.

    Parameters
    ----------
    best_triplet: tuple
        best_triplet = (score_prop,
                        [(sc1, label1, vect1), ..., (sc3, label3, vect3)]).

    Returns
    -------
    None.

    """
    str_prt = "The best triplet has the score"
    print("{:<30} {:>15.2f}.".format(str_prt, best_triplet[0]))
    print("{:-^80}".format("The score of each page"))
    for sc, _, _ in best_triplet[1]: print("{:40.4f}".format(sc))
    print("{:-^80}".format("The vectors page"))
    for _, _, page in best_triplet[1]:
        x0, x1, x14, x15 = page[0][0], page[0][1], page[0][-2], page[0][-1]
    print(f"{x0:>15} {x1:6} ... {x14:6} {x15:6}")


def SelectBestTripletAllNumbers(all_triplets, global_mean_vect,
                                global_vect_std, verbose):
    """
    Computation of the variance of a proposition and then the final score of a
    triplet.
    Addition of the global score of the proposition to the triplet.

    Parameters
    ----------
    all_triplets: list of lists of tuples
        all_triplets = [[(score_page1, label_pred1, short_vect_pg1),
                         ...,
                         (score_page3, label_pred3, short_vect_pg3)], ...].

    global_mean_vect: numpy array
        Vector of the same mength as the short_vect_page with the means.

    global_vect_std: numpy array
        Vector of the same mength as the short_vect_page with the stds.

    verbose: int >= 0
        If verbose > 0, print info.

    Raises
    ------
    Exception
        If there is an issue while computing the variance.

    Returns
    -------
    all_triplets_scores: list of tuples.
        all_triplets_scores = [(score_prop, triplet), ...] with
        triplet = [(sc1, label1, vect1), ..., (sc3, label3, vect3)]

    """
    vect_var = []
    for triplet in all_triplets:
        list_vectors = [vect_pg for _, _, vect_pg in triplet]
        mean_local_vect = np.mean(list_vectors, axis=0)
        norm_mean_local_vect = (mean_local_vect - global_mean_vect)/global_vect_std
        variance_triplet = 0
        for _, _, vect_page in triplet:
            norm_vect = (vect_page - global_mean_vect)/global_vect_std
            try:
                diff_eucl = np.sqrt((norm_vect - norm_mean_local_vect)**2)
                variance_triplet += np.sum(diff_eucl)
            except Exception as e:
                str_exc = (f"Error with the computation of the variance: {e}."
                           f"norm_vect: {str(norm_vect)}.\n"
                           f"type(norm_v): {type(norm_vect)}.\n"
                           f"norm_mean_local_vect: {norm_mean_local_vect}.\n"
                           f"type(norm_mean_loc_v): "
                           f"{type(norm_mean_local_vect)}.")
                raise MyException(str_exc)
        vect_var.append(variance_triplet)
    # Here, I need to standardize the variance
    mean_var, std_var = np.mean(vect_var, axis=0), np.std(vect_var, axis=0)

    if verbose > 0:
        print("{:<30} {:>15.2f}.".format("The mean var", mean_var))
        print("{:<30} {:>15.2f}.".format("The std of the var", std_var))

    if np.isclose(std_var, 0):
        std_var = 1
    norm_vect_var = (vect_var - mean_var)/std_var
    all_triplets_scores = []
    for triplet, var in zip(all_triplets, norm_vect_var):
        score_prop = sum((sc for sc, _, _ in triplet))
        # We ponderate the variance by 0.5
        all_triplets_scores.append((score_prop + 0.5*var, triplet))
    # Ultimately, we take the triplet with the best score
    all_triplets_scores.sort(key=itemgetter(0), reverse=True)
    best_triplet = all_triplets_scores[0]

    if verbose > 0:
        ShowsBestTriplet(best_triplet)

    return all_triplets_scores


def GetMeanStdFromList(big_list_sc_label_vectsh):
    """
    Returns the mean and std vect of a list of vectors with dim = 2.
    Here, it is numpy vectors of dim (1, 15).

    Parameters
    ----------
    big_list_sc_label_vectsh: list of tuples
        big_list_sc_label_vectsh = [(score, label, vectsh), ...].

    Returns
    -------
    global_mean_vect: numpy array
        Vector line with the means (dim=2).

    global_vect_std: numpy array
        Vector line with the stds (dim=2).

    """
    # Creation of a big matrix in order to compute the global mean and std
    for i, tuple_page in enumerate(big_list_sc_label_vectsh):
        if i == 0:
            full_matrix = tuple_page[-1]
        else:
            full_matrix = np.concatenate((full_matrix, tuple_page[-1]))
    global_vect_std = np.std(full_matrix, axis=0)
    for i in range(len(global_vect_std)):
        if np.isclose(global_vect_std[i], 0):
            global_vect_std[i] = 1
    global_mean_vect = np.mean(full_matrix, axis=0)
    return global_mean_vect, global_vect_std


def XScores(dict_arts_input, list_features):
    """
    Builds the matrix X with all the articles input. The point is to use this
    matrix to predict the score.

    Parameters
    ----------
    dict_arts_input: dict
        dict_arts_input = {id_article: dict_article, ...}.

    list_features: list of strings
        The features used to build X.

    Returns
    -------
    X: numpy array
        The matrix with all the vectors input.
        dim(X) = (nb_articles_input, len(list_features)).

    """
    for k, dicta in enumerate(dict_arts_input.values()):
        vect_art = np.array([dicta[x] for x in list_features], ndmin=2)
        if k == 0:
            X = vect_art
        else:
            X = np.concatenate((X, vect_art), axis=0)
    return X


def AttributesScoresToArticlesInput(dict_pages, list_features,
                                    dict_arts_input, verbose):
    """
    Gives a score to each article input and add this score to the
    dict_arts_input.

    Parameters
    ----------
    dict_pages: dict of dicts
        The big dict with all the data about the pages.

    list_features: list of strings
        The usual list with the features used to predict the score.

    dict_arts_input: dict of dicts
        dict_arts_input = {id_article: dict_article, ...}.

    verbose: int >= 0
        If verbose > 0, print info.

    Returns
    -------
    dict_arts_input: dict
        Same as in the input but with the key 'score' added.

    """
    X_input = XScores(dict_arts_input, list_features)

    args_mat = [dict_pages, list_features]
    Xreg, Yreg = module_art_type.MatricesTypeArticle(*args_mat, score=True)

    if verbose > 0:
        print(f"Shape of X_input: {X_input.shape}")
        print(f"Shape of Xreg: {Xreg.shape}")

    # Training of the model
    xgbreg = XGBRegressor()
    t0 = time.time()
    xgbreg.fit(Xreg, Yreg)
    if verbose > 0:
        print(f"Duration fit reg score art.: {time.time() - t0} sec.")

    # We predict the scores for the new articles
    preds_xgbreg = xgbreg.predict(X_input)
    for ida, score_pred in zip(dict_arts_input.keys(), preds_xgbreg):
        dict_arts_input[ida]['score'] = score_pred
        if verbose > 0:
            for key, val in dict_arts_input[ida].items():
                print(f"{key}: {val}.")
            print("\n\n")

    return dict_arts_input


def ProposalsWithoutGivenLayout(file_in,
                                file_out,
                                dict_pages,
                                dict_arts,
                                list_mdp_data,
                                dict_gbc,
                                list_features,
                                tol_total_area,
                                tol_nb_images,
                                tol_score_min,
                                verbose):
    """
    Parameters
    ----------
    file_in: str
        Path to the folder with the xml files with the articles input.

    file_out: str
        Path to a file that will contain the results of this function.

    dict_pages: dictionary
        The keys are the ids of the pages and the value a dictionary with all
        the elements of the articles of that page.

    dict_arts: dictionary
        {id_article: dict_art}.

    list_mdp_data: list
        [(nb_pages, array_mdp, list_ids), ...]

    dict_gbc: dictionary of gradient boosting classifier
        {2: gbc2, 3: gbc3, 4:gbc4}

    list_features: list of strings
        The list with the features used to form the vectors and train the
        models.

    tol_total_area: float
        The tolerance between the area of the module and of the article.
        between 0 and 1

    tol_nb_images: int
        The minimum number of images of the pages results.

    tol_score_min: float (between 0 and 1, default: 0.2)
        The min score of the articles in the page.

    verbose: int
        Whether to print something or not.

    Returns
    -------
    all_triplets_scores: list
        [(score_triplet, triplet)] with
        triplet = [(sc_page, label_page, array_page), (), ()]

    """
    start_time = time.time()
    dict_arts_input = CreationDDArt(file_in)
    dur = time.time() - start_time
    if verbose > 0:
        print("Duration loading input: {:*^12.2f} sec.".format(dur))

    # Here, I can build the matrices X (concatenation art), Y (scores)
    # The point is to use the model to predict the score
    args_sc = [dict_pages, list_features, dict_arts_input, verbose]
    dict_arts_input = AttributesScoresToArticlesInput(*args_sc)

    big_list_sc_label_vectsh = []
    list_vectsh_ids = []
    # A dictionary of dictionary
    dd_labels = {}

    # We create pages with a number of articles between 2 and 5
    for nb_arts in range(2, 5):

        t1 = time.time()
        if verbose > 0:
            print("\n{:-^80}\n".format("PAGES WITH {} ARTS".format(nb_arts)))
            print(f"Trying to build pages with {nb_arts} articles.\n")

        # The list all_ntuples corresponds to all nb_arts-tuple possible
        all_ntuples = GenerationAllNTupleFromFiles(dict_arts_input, nb_arts,
                                                   verbose)

        # The set set_ids_page is made of list of ids that respect the basic
        # constraints of a page
        set_ids_page = GetSelectionOfArticlesFromFiles(all_ntuples,
                                                       tol_total_area,
                                                       tol_nb_images,
                                                       tol_score_min,
                                                       verbose)
        if len(set_ids_page) == 0:
            if verbose > 0:
                print("As there are no possibilities, we go to the next number.")
            continue

        # All the short vectors page. For now it is vectors of size 15 which
        # are the sum of all vectors articles in the page
        args_fill = [set_ids_page, list_features, dict_arts_input]
        list_short_vect_page = FillShortVectorPage(*args_fill)

        # We add the couple (sh_vect, ids) to all_list_couples_sh_vect_ids.
        # This will be used to find back the ids of the articles which form
        # the page.
        for short_vect, ids_page in zip(list_short_vect_page, set_ids_page):
            list_vectsh_ids.append((short_vect, ids_page))

        # But I also need the long vectors page for the predictions
        args_files = [set_ids_page, dict_arts_input, list_features]
        list_long_vect_page = CreationListVectorsPageFromFiles(*args_files)

        if len(list_long_vect_page) == 0:
            continue

        # Now, I train the model. Well, first the matrices
        nb_pgmin = 20
        args_cr = [dict_pages, dict_arts, list_mdp_data, nb_arts, nb_pgmin]
        X, Y, dict_labels = CreateXYFromScratch(*args_cr, list_features, verbose)

        t_model = time.time()
        args_pr = [X, Y, list_long_vect_page, dict_gbc[nb_arts], verbose]
        preds_gbc = PredsModel(*args_pr)
        d_mod = time.time() - t_model

        if verbose > 0:
            str_prt = f"Duration model {nb_arts} articles"
            print(f"{str_prt} {d_mod:*^12.2f} sec.")

        dd_labels[nb_arts] = dict_labels

        # I add the score given by preds_gbc to the short vect_page and the
        # index of the max because it is the label predicted.
        list_score_label_vectsh = []
        for short_vect_pg, line_pred in zip(list_short_vect_page, preds_gbc):
            label_pred = list(preds_gbc[0]).index(max(preds_gbc[0]))
            sc_page = max(preds_gbc[0])
            list_score_label_vectsh.append((sc_page, label_pred, short_vect_pg))
        big_list_sc_label_vectsh += list_score_label_vectsh

        if verbose > 0:
            str_prt = "Total duration pages {} arts".format(nb_arts)
            print("{} {:*^12.2f} sec.".format(str_prt, time.time() - t1))
            print("{:-^80}".format(""))

    t2 = time.time()
    gbal_mean_vect, gbal_vect_std = GetMeanStdFromList(big_list_sc_label_vectsh)
    all_triplets = [[p1, p2, p3]
                    for i, p1 in enumerate(big_list_sc_label_vectsh)
                    for j, p2 in enumerate(big_list_sc_label_vectsh[i + 1:])
                    for p3 in big_list_sc_label_vectsh[i + j + 2:]]

    args_all_nb = [all_triplets, gbal_mean_vect, gbal_vect_std, verbose]
    all_triplets_scores = SelectBestTripletAllNumbers(*args_all_nb)

    if verbose > 0:
        str_prt = "Duration computation scores triplet"
        print("{} {:*^12.2f} sec.".format(str_prt, time.time() - t2))

    return all_triplets_scores, dd_labels, dict_arts_input, list_vectsh_ids


def ShowResultsProWGLay(all_triplets_scores, dd_labels,
                        list_vectsh_ids, verbose):
    """
    Returns the ids of the best pages.

    Parameters
    ----------
    all_triplets_scores: list of tuples
        [(score_triplet, triplet=[(sc_page, label_page, array_page), (), ()])].

    dd_labels: dictionary of dictionaries
        {nb_arts_pg: {0: array_mdp, ...}, ...}.

    list_vectsh_ids: list
        list_vectsh_ids = [(vectsh, tuple_ids_articles_page), ...].

    verbose: int
        Whether to print something.

    Returns
    -------
    list_ids_bestpage: list of tuples
        list_ids_bestpage = [(array_layout, ids_art), ...].

    """
    list_ids_found = []
    list_ids_bestpage = []
    for k in range(3):
        if verbose > 0:
            print("{:-^80}".format("PAGE {}".format(k + 1)))
        for i, (_, label, vect_page) in enumerate(all_triplets_scores[k][1]):
            try:
                f = lambda x: np.allclose(x[0], vect_page)
                ids_found = tuple(filter(f, list_vectsh_ids))
            except Exception as e:
                str_exc = (f"An exception occurs: {e}.\n"
                           f"The vect_page: {vect_page}")
                raise MyException(str_exc)
            list_ids_found.append(ids_found)
            if verbose > 0:
                print("ids page {} result: {}".format(i, ids_found[0][1]))
            if k == 0:
                ids_pg = ids_found[0][1]
                layout = dd_labels[len(ids_found[0][1])][label]
                list_ids_bestpage.append((layout, ids_pg))
    return list_ids_bestpage


def FindLocationArticleInLayout(dico_bdd,
                                list_mdp_data,
                                list_ids_bestpage,
                                list_features,
                                dicta_input,
                                verbose):
    """

    This function is used once we have the association layout/ids of articles.
    It finds the best location of the articles in the different modules of the
    layout.

    Parameters
    ----------
    dico_bdd: dictionary
        The data dictionary with the infos about the pages.

    list_mdp_data: list
        The data with the the layouts.
        list_mdp_data = [(nb, array, list_ids), ...]


    list_ids_bestpage: list
        List of the form [(array_mdp, (id_art1, id_art2, ...)), ...].
        It contains three elements, it is the best triplet of pages.

    list_features: list of strings
        The list of features used for the model that predict the best location
        of an article in the page.

    Returns
    -------
    list_xmlout: list
        List of dictionaries of the form {(x, y, w, h): id_art, ...}.
        Each dictionary corresponds to a page result.
    """
    list_xmlout = []
    for layout_array, list_ids in list_ids_bestpage:
        nb_art = len(list_ids)
        min_nb_art = 15
        args_select_model = [list_mdp_data, nb_art, min_nb_art]
        liste_tuple_mdp = propositions.SelectionModelesPages(*args_select_model)
        res_found = propositions.TrouveMDPSim(liste_tuple_mdp, layout_array)
        mdp_ref = res_found[0][1]
        liste_ids_found = res_found[0][2]
        # We build the matrices X and Y to train the model
        args = [dico_bdd, liste_ids_found, mdp_ref, list_features]
        X, Y = methods.GetTrainableData(*args)

        if verbose > 0:
            print(f"The shape of X in FINDLOC: {X.shape}\n{X[0]}")
            print(f"The list of features used FINDLOC: {list_features}")

        # I need to have also the X_input
        # First thing first, The first page of the triplet result
        for i, id_art in enumerate(list_ids):
            vect_art = [dicta_input[id_art][x] for x in list_features]
            vect_art_array = np.array(vect_art, ndmin=2)
            if i == 0:
                X_input = vect_art_array
            else:
                X_input = np.concatenate((X_input, vect_art_array))
        clf = RandomForestClassifier()
        clf.fit(X, Y)
        # Matrix with the predicted probas for each article input
        probas_preds = clf.predict_proba(X_input)
        # From that we generate all possible combinations of articles and we
        # keep the one with the best score
        # 1. First I generate a list of tuples with ids of articles
        if len(list_ids) == 2:
            poss = [list_ids, (list_ids[1], list_ids[0])]
        elif len(list_ids) == 3:
            seti = set([0, 1, 2])
            poss = []
            for j, id1 in enumerate(list_ids):
                for k, id2 in enumerate(list_ids[:j] + list_ids[j + 1:]):
                    l3 = [list_ids[p] for p in list(seti - set([k, j]))]
                    for id3 in l3:
                        poss.append((id1, id2, id3))
        elif len(list_ids) == 4:
            set_ind = set([0, 1, 2, 3])
            poss = []
            for j, id1 in enumerate(list_ids):
                for k, id2 in enumerate(list_ids[:j] + list_ids[j + 1:]):
                    l3 = [list_ids[p] for p in list(set_ind - set([k, j]))]
                    for l, id3 in enumerate(l3):
                        l4 = [list_ids[p] for p in list(set_ind - set([l, k, j]))]
                        for id4 in l4:
                            poss.append((id1, id2, id3, id4))
        # Now we attribuate a score to each possibility by using the predicted
        # probas
        # First a dict with idart: line_pred
        dict_preds = {ida: pred for ida, pred in zip(list_ids, probas_preds)}
        list_score_poss = []
        for possibility in poss:
            score = sum([dict_preds[ida][i] for i, ida in enumerate(possibility)])
            list_score_poss.append((score, possibility))
        # We sort that list and keep only the page with the best score
        list_score_poss.sort(key=itemgetter(0), reverse=True)
        best_page = list_score_poss[0][1]
        args_zip = [layout_array, best_page]
        dict_xmlout = {tuple(module): ida for module, ida in zip(*args_zip)}
        list_xmlout.append(dict_xmlout)
    return list_xmlout


def CreateXmlOutNoLayout(list_xmlout, file_out):
    """
    The function creates an xml of the form:
    <PagesLayout>
        <PageLayout>
            <Module x=... y=... width=... height=... id_article=...>
            <Module x=... ...>
        </PageLayout>
        ...
    </PageLayout>

    Parameters
    ----------
    list_xmlou: list of dictionaries
        [
        {(x, y, w, h): (id_art_1, ...), ..., idLayout: id or "None"},
        {...},
         ...
         ].
    file_out: str
        path/to/file/out.xml.

    Returns
    -------
    bool
    """
    PagesLayout = ET.Element("PagesLayout")
    for i, dico_page_result in enumerate(list_xmlout):
        PageLayout = ET.SubElement(PagesLayout, "PageLayout", name=str(i))
        PageLayout.set("idLayout", dico_page_result['idLayout'])
        for tuple_emp, ida in dico_page_result.items():
            if tuple_emp != "idLayout":
                Module = ET.SubElement(PageLayout, "Module")
                Module.set("x", str(tuple_emp[0]))
                Module.set("y", str(tuple_emp[1]))
                Module.set("width", str(tuple_emp[2]))
                Module.set("height", str(tuple_emp[3]))
                Module.set("id_article", str(ida))
    tree = ET.ElementTree(PagesLayout)
    tree.write(file_out, encoding="UTF-8")
    return True


def FinalResultsMethodNoLayout(file_in,
                               file_out,
                               dico_bdd,
                               dict_arts,
                               list_mdp_data,
                               dict_gbc,
                               dict_layouts,
                               list_features,
                               tol_total_area,
                               tol_nb_images,
                               tol_score_min,
                               verbose):
    """

    Parameters
    ----------
    file_in: str
        The folder with one xml file for each articles.

    file_out: str
        The path of a xml file that will be created by the app.

    dico_bdd: dictionary
        The dictionary with the pages already published.

    dict_arts: dictionary
        The dictionary of the form {id_art: dico_art, ...}.

    list_mdp_data: list
        A list of tuple [(nb_arts, array_mdp, list_ids), ...].

    dict_gbc: dictionary
        Dictionary with the gradient boosting classifier of the form
        {2: GBC2, 3: GBC3, ...}

    dict_layouts: dictionary
        A dictionary of the form:
            {id_layout: {'nameTemplate': str,
                         'cartons': list,
                         'id_pages': list,
                         'array': np.array},
            ...}

    list_features: list of strings
        The list with the features used to train the model and predict the
        layouts.

    tol_total_area: float
        The tolerance between the area of the module and of the article. between 0 and 1

    tol_nb_images: int
        The minimum number of images of the pages results.

    tol_score_min: float (between 0 and 1, default: 0.2)
        The min score of the articles in a created page.

    verbose: int
        Whether to print something

    Returns
    -------
    str
        Indicates that everything went well.

    """
    start_time = time.time()
    args = [file_in, file_out, dico_bdd, dict_arts, list_mdp_data, dict_gbc]
    args += [list_features, tol_total_area, tol_nb_images, tol_score_min]
    res_proposals = ProposalsWithoutGivenLayout(*args, verbose)
    triplets_scores, dd_labels, dicta_input, list_vectsh_ids = res_proposals

    # Shows the results in the prompt and isolates the final ids of articles
    args = [triplets_scores, dd_labels, list_vectsh_ids, verbose]
    list_ids_bestpage = ShowResultsProWGLay(*args)

    # Extracts the infos about layout to build the xml output
    t_location = time.time()
    args_fd = [dico_bdd, list_mdp_data, list_ids_bestpage, list_features]
    list_xmlout = FindLocationArticleInLayout(*args_fd, dicta_input, verbose)
    d_location = time.time() - t_location
    if verbose > 0:
        print(f"Duration find location articles: {d_location:*^12.2f} sec.")

    # Try to associate an id of layout to the layouts used.
    for dict_page_result in list_xmlout:
        layout_output = [module for module in dict_page_result.keys()]
        array_layout_output = np.array(layout_output)
        match = False
        # We search for a match in the database
        for id_layout_data, dicolay in dict_layouts.items():
            layout_data = dicolay['array']
            if layout_data.shape == array_layout_output.shape:
                args = [array_layout_output, layout_data, 10]
                if module_montage.CompareTwoLayouts(*args):
                    # If match, we add the id of the layout
                    dict_page_result['idLayout'] = id_layout_data
                    match = True
                    break
        if match is False:
            dict_page_result['idLayout'] = "None"

    if verbose > 0:
        print(f"\nThe list_xmlout after addition of idLayout:\n")
        for dict_lay in list_xmlout:
            for key, val in dict_lay.items():
                print(f"{key}: {val}")
            print("\n")

    # Creates the xml output in the repository file_out
    CreateXmlOutNoLayout(list_xmlout, file_out)
    tot_dur = time.time() - start_time

    if verbose > 0:
        print("{:-^80}".format("TOTAL DURATION - {:.2f} sec.".format(tot_dur)))

    return "Xml file with the results created"
