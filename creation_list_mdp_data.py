#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 14:45:44 2021

@author: baptistehessel

Script that creates the object list_mdp, which is of the form:
    - [(nb_pages_using_layout,
        np.array(layout),
        list_ids_page_using_layout),
       ...]

Imports module_montage.

Objects necessary:
    - dict_page_array_fast


"""
import pickle
import module_montage
import time


def ArrayLayoutInList(array_layout, list_already_added):
    """

    Parameters
    ----------
    array_layout: numpy array
        A layout input.

    list_already_added: list
        A list of layouts.

    Returns
    -------
    bool
        True if array_layout is in list_already_added, else otherwise.

    """
    for array_layout_added in list_already_added:
        args = [array_layout, array_layout_added, 1]
        if module_montage.CompareTwoLayouts(*args):
            return True
    return False


def CreationListLayouts(dict_pg_ar_fast):
    """

    Parameters
    ----------
    dict_pg_ar_fast: dictionary
        Dictionary of the form {nb_modules: {id_page: array_page, ...}, ...}
        that corresponds to all the data we have.

    Returns
    -------
    list_tuple_layout: list of list
        A list of the form [[nb_pages, array_page, list_ids], ...].
        The list_ids contains all the pages that use the layout array_page.

    This function is very long.

    """

    list_tuple_layout = []

    print(f"Beggining of the function CreationListLayouts")

    for nb_modules, dict_nb_modules in dict_pg_ar_fast.items():

        # We only focus on layouts with nb_modules between 2 and 5.
        if nb_modules in range(2, 6):
            print(f"Layouts with {nb_modules} modules.")

            n = len(list(dict_nb_modules.keys()))

            # We go through the layouts with nb_modules modules.
            for i, (idpage1, layout1) in enumerate(dict_nb_modules.items()):
                list_layout = [1, layout1, [idpage1]]

                for idpage2, layout2 in list(dict_nb_modules.items())[i:]:
                    args = [layout1, layout2, 10]

                    if module_montage.CompareTwoLayouts(*args):
                        list_layout[0] += 1
                        list_layout[2].append(idpage2)

                list_tuple_layout.append(list_layout)

                if i % (n // 100) == 0:
                    print(f"CreationListLayouts {nb_modules}: {i/n:.2%}")

    return list_tuple_layout


def CreationListMdp(dict_page_array_fast, path_customer):
    """
    Save the list_mdp.

    Parameters
    ----------
    dict_page_array_fast: dict
        The dict with the pages and the arrays gather by number of modules.

    path_customer: str
        The path to the data of a customer.

    Returns
    -------
    None.

    """
    t0 = time.time()
    nb_exc = 0
    list_tuple_layout = CreationListLayouts(dict_page_array_fast)
    # Intermediary list used to find the indexes of some elements.
    list_listids = [list_ids for _, _, list_ids in list_tuple_layout]
    print(f"Duration CreationListLayouts: {time.time() - t0} sec.")
    for i, (nb_pages, array_lay, list_ids) in enumerate(list_tuple_layout):
        print(f"The number of pages: {nb_pages}")
        print(f"{array_lay}")
        print(f"An extract of the list_ids: {list_ids[:4]}")
        if i == 10:
            break

    print(f"The length of list_tuple_layout: {len(list_tuple_layout)}")
    total_pages = sum((elt for elt, _, _ in list_tuple_layout))
    print(f"The total number of pages: {total_pages}")

    # Well, we need to remove duplicates.
    # Some difficulties here because it deletes almost all ids
    remove_dup = False
    if remove_dup:
        t1 = time.time()
        for nb_pages, array_layout, list_ids in list_tuple_layout:
            for idpage in list_ids:
                # We look if the id is also associated to an other layout
                res_found = list(filter(lambda x: idpage in x[2], list_tuple_layout))
                # If len(res_found) > 0. We add all the other list_ids to the current
                # and we delete the other lists.
                for tuple_found in res_found:
                    # We add all these ids.
                    list_ids += tuple_found[2]
                    # We delete the tuple_found.
                    ### list_tuple_layout.remove(tuple_found)
                    try:
                        index_tuple_found = list_listids.index(tuple_found[2])
                        # We delete the elements in both lists
                        list_tuple_layout.pop(index_tuple_found)
                        list_listids.pop(index_tuple_found)
                    except Exception as e:
                        str_exc = f"An exception while poping tuple_found: {e}\n"
                        str_exc += f"The tuple found: \n{tuple_found}\n"
                        nb_exc += 1
                        if nb_exc < 5:
                            print(str_exc)

            # We delete duplicates in the list_ids.
            list_ids = list(set(list_ids))

        print(f"Duration removal duplicates: {time.time() - t1} sec.")
        print(f"The new length of list_tuple_layout: {len(list_tuple_layout)}")
        total_pages = sum((elt for elt, _, _ in list_tuple_layout))
        print(f"The new total number of pages: {total_pages}")

        for i, (nb_pages, array_lay, list_ids) in enumerate(list_tuple_layout):
            print(f"The number of pages: {nb_pages}")
            print(f"{array_lay}")
            print(f"An extract of the list_ids: {list_ids[:4]}")
            if i == 5:
                break

    with open(path_customer + 'list_mdp', 'wb') as f:
        pickle.dump(list_tuple_layout, f)
