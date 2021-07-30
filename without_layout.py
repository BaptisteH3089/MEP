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


class MyException(Exception):
    pass


def ConstraintArea(all_ntuple):
    # Now we select the ones that respect the constraint
    mean_area = 95000
    possib_area = []
    all_sums = []
    for ntuple in all_ntuple:
        sum_art = sum((art[0] for art in ntuple))
        all_sums.append(sum_art)
        if 0.92 * mean_area <= sum_art <= 1.08 * mean_area:
            # If constraint ok, we add the id of the article
            possib_area.append(tuple([art[-1] for art in ntuple]))
    print("{:<35} {:>25}".format("Constraint area", len(possib_area)))
    print("{:<35} {:>25}".format("Mean area", np.mean(all_sums)))
    return possib_area


# What's the next constraint ? -> Number of images ? (> 2)
# Now we select the ones that respect the constraint
def ConstraintImgs(all_ntuple):
    possib_imgs, all_nb_imgs = [], []
    for ntuple in all_ntuple:
        try:
            nb_imgs = sum((art[1] for art in ntuple))
        except Exception as e:
            print("Error with the sum of the nbImg")
            print(e)
            print("ntuple", ntuple)
            print("nb_imgs", nb_imgs)
        all_nb_imgs.append(nb_imgs)
        if nb_imgs >= 1:
            possib_imgs.append(tuple([art[-1] for art in ntuple]))
    print("{:^35} {:>25}".format("Constraint images", len(possib_imgs)))
    print("{:^35} {:>25}".format("Mean nb of images", np.mean(all_nb_imgs)))
    return possib_imgs


def ConstraintScore(all_ntuple):
    # Importance of the articles
    # -> At least one article with score > 0.5
    possib_score, all_max_score = [], []
    for ntuple in all_ntuple:
        max_score = max((art[2] for art in ntuple))
        all_max_score.append(max_score)
        if max_score >= .5:
            possib_score.append(tuple([art[-1] for art in ntuple]))
    print("{:^35} {:>25}".format("Constraint score", len(possib_score)))
    print("{:^35} {:>25}".format("Mean score", np.mean(all_max_score)))
    return possib_score


##############################################################################
#                                                                            #
#           ALL THE FUNCTIONS NECESSARY FOR CreateXYFromScratch              #
#                                                                            #
##############################################################################


def SelectionPagesNarts(dico_bdd, nb_arts):
    dico_narts = {}
    for key in dico_bdd.keys():
        if len(dico_bdd[key]['articles']) == nb_arts:
            dico_narts[key] = dico_bdd[key]['articles']
    return dico_narts


def SelectionMDPNarts(list_mdp_data, nb_arts):
    list_mdp_narts = []
    for nb, mdp, list_ids in list_mdp_data:
        if mdp.shape[0] == nb_arts:
            list_mdp_narts.append([mdp, list_ids])
    return list_mdp_narts


def VerificationPageMDP(dico_narts, list_mdp_narts):
    list_all_matches = []
    for key in dico_narts.keys():
        list_match = [1 for _, list_ids in list_mdp_narts if key in list_ids]
        list_all_matches.append(sum(list_match))
    str_prt = "The nb of 0 match found"
    # print("{:<35} {:>35}".format(str_prt, list_all_matches.count(0)))
    str_prt = "The total nb of pages in our data"
    return f"{str_prt:<35} {len(dico_narts):>10}"


# OK, there are some duplicates
# I need to create the vector page
# But what to put inside them ?
def CreationVectXY(dico_narts,
                   dict_arts,
                   list_mdp_narts,
                   list_features,
                   nb_pages_min):
    print(f"CreationVectXY. Initial length of dico_narts: {len(dico_narts)}")
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
    print("{:<35} {:>35}".format("The number of pages kept", len(vect_XY)))
    return vect_XY


# Now, we create a dico with the mdp
def CreationDictLabels(vect_XY):
    """
    Returns {0: np.array([...]),
            1: np.array([...]),
            ...}
    The values correspond to the layouts.
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
    print("{:<35} {:>35}".format("The nb of labels in that dict", label + 1))
    return dict_labels


def CreationXY(vect_XY, dict_labels):
    """
    Creation of the matrix X and Y used to train a model.
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
    # For now, there is an issue with big_Y since we can have something like :
    # set(big_Y) = {0, 1, 4, 5, 8}
    # I want to have set(big_Y) = {0, 1, 2, 3, 4}
    dict_corres_labels = {label: i for i, label in enumerate(set(big_Y))}
    better_big_Y = [dict_corres_labels[label] for label in big_Y]
    # print(f"without_layout:CreationXY set(better_big_Y): {set(better_big_Y)}")
    return big_X, np.array(better_big_Y)


def CreateXYFromScratch(dico_bdd,
                        dict_arts,
                        list_mdp,
                        nb_arts,
                        nbpg_min,
                        list_feats):
    """
    Used in the method with files.

    Parameters
    ----------
    dico_bdd : TYPE
        DESCRIPTION.
    dict_arts : TYPE
        DESCRIPTION.
    list_mdp : TYPE
        DESCRIPTION.
    nb_arts : TYPE
        DESCRIPTION.
    nbpg_min : TYPE
        DESCRIPTION.
    list_feats : TYPE
        DESCRIPTION.

    Returns
    -------
    X : TYPE
        DESCRIPTION.
    Y : TYPE
        DESCRIPTION.
    dict_labels : TYPE
        DESCRIPTION.

    """
    dico_narts = SelectionPagesNarts(dico_bdd, nb_arts)
    print(f"SelectionPagesNarts return a dict of length: {len(dico_narts)}")
    list_mdp_narts = SelectionMDPNarts(list_mdp, nb_arts)
    print(f"SelectionMDPNarts returns an object of length: {len(list_mdp_narts)}")
    print(f"VerificationPageMDP: {VerificationPageMDP(dico_narts, list_mdp_narts)}")
    args_xy = [dico_narts, dict_arts, list_mdp_narts, list_feats, nbpg_min]
    vect_XY = CreationVectXY(*args_xy)
    dict_labels = CreationDictLabels(vect_XY)
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


def PredsModel(X, Y, list_vect_page, trained_gbc):
    """
    Used in the method with files.

    Parameters
    ----------
    X : TYPE
        DESCRIPTION.
    Y : TYPE
        DESCRIPTION.
    list_vect_page : TYPE
        DESCRIPTION.
    trained_gbc :
        A trained Gradient Boosting Classifier.

    Returns
    -------
    preds_gbc : TYPE
        DESCRIPTION.

    """
    # gbc = GradientBoostingClassifier().fit(X, Y)
    for i, vect_page in enumerate(list_vect_page):
        if i == 0: matrix_input = vect_page
        else: matrix_input = np.concatenate((matrix_input, vect_page))
    str_prt = "The shape of the matrix input"
    print("{:<35} {:>35}".format(str_prt, str(matrix_input.shape)))
    preds_gbc = trained_gbc.predict_proba(matrix_input)
    preds = trained_gbc.predict(matrix_input)
    print("{:^80}".format("The variety of the predictions (classes, count)"))
    list_sh = [(x, list(preds).count(x)) for x in set(list(preds))]
    print("{:^80}".format(str(list_sh)))
    return preds_gbc


def CreationListDictArt(rep_data):
    """
    Used for method files.

    Parameters
    ----------
    rep_data : str
        The folder with the articles input

    Returns
    -------
    dict_arts_input : dict
        A dictionary of the form {ida: dicoa, ...}
    """
    dict_arts_input = {}
    for file_path in Path(rep_data).glob('./**/*'):
        if file_path.suffix == '.xml':
            try:
                dict_art = propositions.ExtractDicoArtInput(file_path)
                dict_arts_input[dict_art['melodyId']] = dict_art
            except Exception as e:
                print(f"An error with an article input: {e}")
    return dict_arts_input


def GenerationAllNTupleFromFiles(dict_arts_input, nb_arts):
    """
    Used for method files.
    I should add the score.
    """
    list_feat = ['aireTot', 'nbPhoto', 'melodyId']
    val_dict = dict_arts_input.values()
    select_arts = [[dicoa[x] for x in list_feat] for dicoa in val_dict]
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
        str_exc = 'Wrong nb of arts: {}. Must be in [2, 5]'
        raise MyException(str_exc.format(nb_arts))
    print(f"The total nb of {nb_arts}-tuple generated: {len(all_ntuple)}.")
    return all_ntuple


def GetSelectionOfArticlesFromFiles(all_ntuples):
    """
    Used for method files.

    Parameters
    ----------
    all_ntuples : TYPE
        DESCRIPTION.

    Returns
    -------
    final_obj : set
        A set with all the tuples containing th ids of articles that together
        respect the constraints.

    """
    # Use of the constraints
    possib_area = ConstraintArea(all_ntuples)
    possib_imgs = ConstraintImgs(all_ntuples)
    # CONSTRAINT SCORE ??? -> I should add that
    # Intersection of each result obtained
    final_obj = set(possib_imgs) & set(possib_area)
    str_prt = "The number of ntuples that respect the constraints"
    print("{:<55} {:>5}".format(str_prt, len(final_obj)))
    return final_obj


def CreationListVectorsPageFromFiles(final_obj, artd_input, list_feat):
    """
    Used in the method with files.
    """
    list_vect_page = []
    for triplet in final_obj:
        for i, ida in enumerate(triplet):
            artv = np.array([artd_input[ida][x] for x in list_feat], ndmin=2)
            if i == 0: vect_page = artv
            else: vect_page = np.concatenate((vect_page, artv), axis=1)
        list_vect_page.append(vect_page)
    return list_vect_page


def SelectBestTriplet(all_triplets):
    """
    Used in the method with files.

    Computation of the variance of a proposition and then the final score of a
    triplet.
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
            weights += [weight] * 15
        if len(weights) != size_vect_pg:
            print("Something's wrong with the weights")
            print("weights\n", weights)
            print("weights_tmp\n", weights_tmp)
            print("array_weights", array_weights)
            print("triplet: {}".format(triplet))
            break
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
    str_triplet = "The best triplet has the score"
    print("{:<35} {:>35.2f}.".format(str_triplet, best_triplet[0]))
    print("{:-^80}".format("The score of each page"))
    for sc, _ in best_triplet[1]: print("{:40.4f}".format(sc))
    print("{:-^80}".format("The vectors page"))
    for _, page in best_triplet[1]: print("{:^40}".format(str(page)))
    return best_triplet


def GenerationOfAllTriplets(preds_gbc, list_vect_page):
    """
    Used in the method with files.

    Parameters
    ----------
    preds_gbc : TYPE
        DESCRIPTION.
    list_vect_page : TYPE
        DESCRIPTION.

    Returns
    -------
    all_triplets : TYPE
        DESCRIPTION.

    """
    pages_with_score = []
    for line, vect_pg in zip(preds_gbc, list_vect_page):
        pages_with_score.append((round(max(line), 4), vect_pg))
    # Generation of every triplet of pages
    all_triplets = [[p1, p2, p3] for i, p1 in enumerate(pages_with_score)
                    for j, p2 in enumerate(pages_with_score[i + 1:])
                    for p3 in pages_with_score[i + j +2:]]
    return all_triplets


def FillShortVectorPage(final_obj, dict_arts):
    """
    Used in method with files.

    Parameters
    ----------
    final_obj : TYPE
        DESCRIPTION.
    dict_arts : TYPE
        DESCRIPTION.

    Returns
    -------
    list_short_vect_page : TYPE
        DESCRIPTION.

    """
    list_short_vect_page = []
    for ntuple in final_obj:
        # Line vector
        short_vect_page = np.zeros((1, 15))
        # nbSign
        nbSign_page = sum((dict_arts[ida]['nbSign'] for ida in ntuple))
        short_vect_page[0][0] = nbSign_page
        # nbBlock
        nbBlock_page = sum((dict_arts[ida]['nbBlock'] for ida in ntuple))
        short_vect_page[0][1] = nbBlock_page
        # abstract
        abstract_page = sum((dict_arts[ida]['abstract'] for ida in ntuple))
        short_vect_page[0][2] = abstract_page
        # syn
        syn_page = sum((dict_arts[ida]['syn'] for ida in ntuple))
        short_vect_page[0][3] = syn_page
        # exergue
        exergue_page = sum((dict_arts[ida]['exergue'] for ida in ntuple))
        short_vect_page[0][4] = exergue_page
        # title - NOT INTERESTING
        title_page = sum((dict_arts[ida]['title'] for ida in ntuple))
        short_vect_page[0][5] = title_page
        # secTitle
        secTitle_page = sum((dict_arts[ida]['secTitle'] for ida in ntuple))
        short_vect_page[0][6] = secTitle_page
        # supTitle
        supTitle_page = sum((dict_arts[ida]['supTitle'] for ida in ntuple))
        short_vect_page[0][7] = supTitle_page
        # subTitle
        subTitle_page = sum((dict_arts[ida]['subTitle'] for ida in ntuple))
        short_vect_page[0][8] = subTitle_page
        # nbPhoto
        nbPhoto_page = sum((dict_arts[ida]['nbPhoto'] for ida in ntuple))
        short_vect_page[0][9] = nbPhoto_page
        # aireImg
        aireImg_page = sum((dict_arts[ida]['aireImg'] for ida in ntuple))
        short_vect_page[0][10] = aireImg_page
        # aireTot
        aireTot_page = sum((dict_arts[ida]['aireTot'] for ida in ntuple))
        short_vect_page[0][11] = aireTot_page
        # petittitre
        petittitre_page = sum((dict_arts[ida]['petittitre'] for ida in ntuple))
        short_vect_page[0][12] = petittitre_page
        # quest_rep
        quest_rep_page = sum((dict_arts[ida]['quest_rep'] for ida in ntuple))
        short_vect_page[0][13] = quest_rep_page
        # intertitre
        intertitre_page = sum((dict_arts[ida]['intertitre'] for ida in ntuple))
        short_vect_page[0][14] = intertitre_page
        # We add that vector to the list
        list_short_vect_page.append(short_vect_page)
    return list_short_vect_page


def SelectBestTripletAllNumbers(all_triplets, global_mean_vect, global_vect_std):
    """
    Computation of the variance of a proposition and then the final score of a
    triplet.
    """
    vect_var = []
    for triplet in all_triplets:
        list_vectors = [vect_pg for _, _, vect_pg in triplet]
        mean_local_vect = np.mean(list_vectors, axis=0)
        norm_mean_local_vect = (mean_local_vect - global_mean_vect) / global_vect_std
        variance_triplet = 0
        for _, _, vect_page in triplet:
            norm_vect = (vect_page - global_mean_vect) / global_vect_std
            try:
                diff_eucl = np.sqrt((norm_vect - norm_mean_local_vect) ** 2)
                variance_triplet += np.sum(diff_eucl)
            except Exception as e:
                args_n = ["norm_mean_local_vect", norm_mean_local_vect]
                args_t = ["type(norm_mean_loc_v)", type(norm_mean_local_vect)]
                print("Error with the computation of the variance \n", e)
                print("{:^35} {:>40}".format("norm_vect", str(norm_vect)))
                print("{:^35} {:>40}".format("type(norm_v)", type(norm_vect)))
                print("{:^35} {:>40}".format(*args_n))
                print("{:^35} {:>40}".format(*args_t))
                raise Exception
        vect_var.append(variance_triplet)
    # Here, I need to standardize the variance
    mean_var, std_var = np.mean(vect_var, axis=0), np.std(vect_var, axis=0)
    print("{:<40} {:>30.2f}".format("The mean var", mean_var))
    print("{:<40} {:>30.2f}".format("The std of the var", std_var))
    if np.isclose(std_var, 0): std_var = 1
    norm_vect_var = (vect_var - mean_var) / std_var
    all_triplets_scores = []
    for triplet, var in zip(all_triplets, norm_vect_var):
        score_prop = sum((sc for sc, _, _ in triplet))
        # We ponderate the variance by 0.5
        all_triplets_scores.append((score_prop + 0.5 * var, triplet))
    # Ultimately, we take the triplet with the best score
    all_triplets_scores.sort(key=itemgetter(0), reverse=True)
    best_triplet = all_triplets_scores[0]
    str_prt = "The best triplet has the score"
    print("{:<50} {:>20.2f}.".format(str_prt, best_triplet[0]))
    print("{:-^80}".format("The score of each page"))
    for sc, _, _ in best_triplet[1]: print("{:40.4f}".format(sc))
    print("{:-^80}".format("The vectors page"))
    for _, _, page in best_triplet[1]:
        x0, x1, x14, x15 = page[0][0], page[0][1], page[0][-2], page[0][-1]
        print(f"{x0:>15} {x1:6} ... {x14:6} {x15:6}")
    return all_triplets_scores


def GetMeanStdFromList(big_list_sc_label_vectsh):
    """
    Returns the mean and std vect of a list of vectors with dim = 2.
    Here, it is numpy vectors of dim (1, 15).
    """
    # Creation of a big matrix in order to compute the global mean and std
    for i, tuple_page in enumerate(big_list_sc_label_vectsh):
        if i == 0: full_matrix = tuple_page[-1]
        else: full_matrix = np.concatenate((full_matrix, tuple_page[-1]))
    global_vect_std = np.std(full_matrix, axis=0)
    for i in range(len(global_vect_std)):
        if np.isclose(global_vect_std[i], 0): global_vect_std[i] = 1
    global_mean_vect = np.mean(full_matrix, axis=0)
    return global_mean_vect, global_vect_std


def ProposalsWithoutGivenLayout(file_in,
                                file_out,
                                dico_bdd,
                                dict_arts,
                                list_mdp_data,
                                dict_gbc):
    """
    Parameters
    ----------
    file_in : str
        Path to the folder with the xml files with the articles input.
    file_out : str
        Path to a file that will contain the results of this function.
    dico_bdd : dictionary
        The keys are the ids of the pages and the value a dictionary with all
        the elements of the articles of that page.
    dict_arts : dictionary
        {id_article: dict_art}.
    list_mdp_data : list
        [(nb_pages, array_mdp, list_ids), ...]
    dict_gbc : dictionary of gradient boosting classifier
        {2: gbc2, 3: gbc3, 4:gbc4}

    Returns
    -------
    all_triplets_scores : list
        [(score_triplet, triplet)] with
        triplet = [(sc_page, label_page, array_page), (), ()]

    """
    start_time = time.time()
    dict_arts_input = CreationListDictArt(file_in)
    print("Duration loading input: {:*^12.2f}".format(time.time() - start_time))
    # Ok, now I should convert each page of the objects final_object into a vector
    # page of size 15 with the following features
    list_features = ['nbSign', 'nbBlock', 'abstract', 'syn']
    list_features += ['exergue', 'title', 'secTitle', 'supTitle']
    list_features += ['subTitle', 'nbPhoto', 'aireImg', 'aireTot']
    list_features += ['petittitre', 'quest_rep', 'intertitre']
    big_list_sc_label_vectsh = []
    list_vectsh_ids = []
    # A dictionary of dictionary
    dd_labels = {}
    for nb_arts in range(2, 5):
        t1 = time.time()
        print("{:-^80}".format("PAGES WITH {} ARTS".format(nb_arts)))
        # The list all_ntuples corresponds to all nb_arts-tuple possible
        all_ntuples = GenerationAllNTupleFromFiles(dict_arts_input, nb_arts)
        # The set set_ids_page is made of list of ids that respect the basic
        # constraints of a page
        set_ids_page = GetSelectionOfArticlesFromFiles(all_ntuples)
        # All the short vectors page. For now it is vectors of size 15 which are
        # the sum of all vectors articles in the page
        list_short_vect_page = FillShortVectorPage(set_ids_page, dict_arts_input)
        # We add the couple (sh_vect, ids) to all_list_couples_sh_vect_ids
        # This will be used to find back the ids of the articles which form the
        # page.
        for short_vect, ids_page in zip(list_short_vect_page, set_ids_page):
            list_vectsh_ids.append((short_vect, ids_page))
        # But I also need the long vectors page for the predictions
        args_files = [set_ids_page, dict_arts_input, list_features]
        list_long_vect_page = CreationListVectorsPageFromFiles(*args_files)
        if len(list_long_vect_page) == 0: continue
        # Now, I train the model. Well, first the matrices
        nb_pgmin = 20
        args_cr = [dico_bdd, dict_arts, list_mdp_data, nb_arts, nb_pgmin]
        X, Y, dict_labels = CreateXYFromScratch(*args_cr, list_features)
        t_model = time.time()
        preds_gbc = PredsModel(X, Y, list_long_vect_page, dict_gbc[nb_arts])
        d_mod = time.time() - t_model
        str_prt = f"Duration model {nb_arts} articles"
        print(f"{str_prt:<35} {d_mod:*^12.2f} sec.")
        dd_labels[nb_arts] = dict_labels
        # I add the score given by preds_gbc to the short vect_page and the
        # index of the max because it is the label predicted.
        list_score_label_vectsh = []
        for short_vect_pg, line_pred in zip(list_short_vect_page, preds_gbc):
            label_pred = list(preds_gbc[0]).index(max(preds_gbc[0]))
            sc_page = max(preds_gbc[0])
            list_score_label_vectsh.append((sc_page, label_pred, short_vect_pg))
        big_list_sc_label_vectsh += list_score_label_vectsh
        str_prt = "Total duration pages {} arts".format(nb_arts)
        print("{:<35} {:*^12.2f} sec.".format(str_prt, time.time() - t1))
        print("{:-^80}".format(""))
    t2 = time.time()
    gbal_mean_vect, gbal_vect_std = GetMeanStdFromList(big_list_sc_label_vectsh)
    all_triplets = [[p1, p2, p3]
                    for i, p1 in enumerate(big_list_sc_label_vectsh)
                    for j, p2 in enumerate(big_list_sc_label_vectsh[i + 1:])
                    for p3 in big_list_sc_label_vectsh[i + j + 2:]]
    args_all_nb = [all_triplets, gbal_mean_vect, gbal_vect_std]
    all_triplets_scores = SelectBestTripletAllNumbers(*args_all_nb)
    str_prt = "Duration computation scores triplet"
    print("{:<50} {:*^12.2f} sec.".format(str_prt, time.time() - t2))
    # It remains to create the xml output
    # Since I have no id of modules, I can create for now false ids
    # But I think this is useless
    return all_triplets_scores, dd_labels, dict_arts_input, list_vectsh_ids


def ShowResultsProWGLay(all_triplets_scores, dd_labels, list_vectsh_ids):
    """
    Used in the method with files.

    Parameters
    ----------
    all_triplets_scores : list of tuples
        [(score_triplet, triplet=[(sc_page, label_page, array_page), (), ()])].
    dd_labels : dictionary of dictionaries
        {nb_arts_pg: {0: array_mdp, ...}, ...}.
    list_vectsh_ids : TYPE
        DESCRIPTION.

    Returns
    -------
    list_ids_bestpage : list of tuples
        [(array_layout, ids_art), ...].

    """
    list_ids_found = []
    list_ids_bestpage = []
    for k in range(3):
        print("{:-^80}".format("PAGE {}".format(k + 1)))
        for i, (_, label, vect_page) in enumerate(all_triplets_scores[k][1]):
            try:
                f = lambda x: np.allclose(x[0], vect_page)
                ids_found = tuple(filter(f, list_vectsh_ids))
            except Exception as e:
                print("An exception occurs", e)
                print("The vect_page: {}".format(vect_page))
            list_ids_found.append(ids_found)
            print("The ids of the page {} result: {}".format(i, ids_found[0][1]))
            if k == 0:
                ids_pg = ids_found[0][1]
                layout = dd_labels[len(ids_found[0][1])][label]
                list_ids_bestpage.append((layout, ids_pg))
    return list_ids_bestpage


def FindLocationArticleInLayout(dico_bdd,
                                list_mdp_data,
                                list_ids_bestpage,
                                list_features,
                                dicta_input):
    """
    Used in the method with files.

    This function is used once we have the association layout/ids of articles.
    It finds the best location of the articles in the different modules of the
    layout.

    Parameters
    ----------
    dico_bdd : dictionary
        DESCRIPTION.
    list_mdp_data : list
        DESCRIPTION.
    list_ids_bestpage : list
        List of the form [(array_mdp, (id_art1, id_art2, ...)), ...].
        It contains three elements, it is the best triplet of pages.
    list_features : list
        The list of features used for the model that predict the best location
        of an article in the page.
    Returns
    -------
    list_xmlout : list
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
        # I need to have also the X_input
        # First thing first, The first page of the triplet result
        for i, id_art in enumerate(list_ids):
            vect_art = [dicta_input[id_art][x] for x in list_features]
            if i == 0:
                X_input = np.array(vect_art, ndmin=2)
            else:
                X_input = np.concatenate((X_input, np.array(vect_art, ndmin=2)))
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
    Used in the method with files.

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
    list_xmlout : list of dictionaries
        [
        {(x, y, w, h): (id_art_1, ...), ..., idLayout: id or "None"},
        {...},
         ...
         ].
    file_out : str
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
                               dict_layouts):
    """

    Parameters
    ----------
    file_in : str
        The folder with one xml file for each articles.
    file_out : str
        The path of a xml file that will be created by the app.
    dico_bdd : dictionary
        The dictionary with the pages already published.
    dict_arts : dictionary
        The dictionary of the form {id_art: dico_art, ...}.
    list_mdp_data : list
        A list of tuple [(nb_arts, array_mdp, list_ids), ...].
    dict_gbc : dictionary
        Dictionary with the gradient boosting classifier of the form
        {2: GBC2, 3: GBC3, ...}
    dict_layouts : dictionary
        A dictionary of the form:
            {id_layout: {'nameTemplate': str,
                         'cartons': list,
                         'id_pages': list,
                         'array': np.array}, ...}

    Returns
    -------
    str
        Indicates that everything went well.

    """
    start_time = time.time()
    args = [file_in, file_out, dico_bdd, dict_arts, list_mdp_data, dict_gbc]
    res_proposals = ProposalsWithoutGivenLayout(*args)
    triplets_scores, dd_labels, dicta_input, list_vectsh_ids = res_proposals
    # Shows the results in the prompt and isolates the final ids of articles
    args = [triplets_scores, dd_labels, list_vectsh_ids]
    list_ids_bestpage = ShowResultsProWGLay(*args)
    # Extracts the infos about layout to build the xml output
    t_location = time.time()
    args_fd = [dico_bdd, list_mdp_data, list_ids_bestpage, list_features]
    list_xmlout = FindLocationArticleInLayout(*args_fd, dicta_input)
    print(f"The list_xmlout: {list_xmlout}")
    d_location = time.time() - t_location
    print(f"Duration find location articles: {d_location:*^12.2f} sec.")
    # HERE I SHOULD TRY TO FIND CORRESPONDANCES WITH THE TRUE LAYOUTS
    # IF WE FIND A MATCH, WE ADD THE MELODYID IN <PageLayout MelodyId=...>
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
    print(f"The list_xmlout after addition of idLayout: {list_xmlout}")
    # Creates the xml output in the repository file_out
    CreateXmlOutNoLayout(list_xmlout, file_out)
    tot_dur = time.time() - start_time
    print("{:-^80}".format("TOTAL DURATION - {:.2f} sec.".format(tot_dur)))
    return "Xml file with the results created"


#%%

list_features = ['nbSign', 'nbBlock', 'abstract', 'syn']
list_features += ['exergue', 'title', 'secTitle', 'supTitle']
list_features += ['subTitle', 'nbPhoto', 'aireImg', 'aireTot']
list_features += ['petittitre', 'quest_rep', 'intertitre']
path_mep = '/Users/baptistehessel/Documents/DAJ/MEP/'
file_in = path_mep + 'ArticlesOptions/'
file_out = path_mep + 'montageIA/out/outNoLayout.xml'
# args = [file_in, file_out, dico_bdd, dict_arts, list_mdp_data, dict_gbc]
# FinalResultsMethodNoLayout(*args)
