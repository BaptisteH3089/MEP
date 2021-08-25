#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The files Dico, Avoid and MC_Avoid are the same as the ones used for
similarity. For now, they are not necessary.

The sections FEATURES and CONSTRAINTS correspond to some parameters of the
algorithm.

The section verbose contains:
    - verbose: integer >= 0.
    If 0 we don't print anything, else we print intermediate results in the
    prompt.

"""
[LOG]
path_log: /var/log/daj/montageia/montage_ia.log

[DATA]
path_data: /data/montageia/corpus/

[FILES]
chemin_dico: /data/montageia/bin/Dico
chemin_avoid: /data/montageia/bin/Avoid
chemin_mcavoid: /data/montageia/bin/MC_Avoid

[FEATURES]
list_features: ["nbSign", "nbBlock", "abstract", "syn", "exergue",
                "title", "secTitle", "supTitle", "subTitle", "nbPhoto",
                "aireImg", "aireTot", "petittitre", "quest_rep", "intertitre"]

[CONSTRAINTS]
# Constraints for the layouts
tol_total_area: 0.08
tol_nb_images: 1
tol_score_min: 0.2
# Constraints for the modules
tol_area_images_mod: 0.5
tol_area_text_mod: 0.4
tol_total_area_mod: 0.3

[VERBOSE]
verbose: 0

[CUSTOMERS]
CM: 6001
CF: 6002





