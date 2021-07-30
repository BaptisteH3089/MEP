#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 12:48:31 2021

@author: baptistehessel

"""
import xml.etree.cElementTree as ET
import numpy as np


def Creationxml(statut, file_out):
    """
    Creates an xml file of the form:
        - <root><statut><'blablabla'></statut></root>.

    Parameters
    ----------
    statut : str
        Just some text.
    file_out : str
        Where we want to create the xml file.

    Returns
    -------
    bool
        True

    """
    root = ET.Element("root")
    ET.SubElement(root, "statut").text = statut
    tree = ET.ElementTree(root)
    tree.write(file_out)
    return True


def CompareTwoLayouts(layout1, layout2, tol=20):
    """

    Useful function in many cases.

    Parameters
    ----------
    layout1 : numpy array
        np.array([[x, y, w, h], [x, y, w, h], ...]).
    layout2 : numpy array
        np.array([[x, y, w, h], [x, y, w, h], ...]).

    Returns
    -------
    bool
        True if the two layouts are approximately the same.

    """
    if layout1.shape != layout2.shape:
        return False
    n = len(layout1)
    nb_correspondances = 0
    for i in range(n):
        for j in range(n):
            try:
                if np.allclose(layout1[j], layout2[i], atol=tol):
                    nb_correspondances += 1
            except Exception as e:
                str_exc = (f"An exception occurs with np.allclose: \n{e}\n"
                           f"layout1: {layout1}\n"
                           f"j: {j}, i: {i}\n"
                           f"layout2: \n{layout2}")
                print(str_exc)
    return True if nb_correspondances == n else False