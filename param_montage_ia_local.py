#!/usr/bin/env python3
# -*- coding: utf-8 -*-
[LOG]
path_log: /Users/baptistehessel/Documents/DAJ/MEP/montageIA/log/montage_ia.log

[DATA]
path_data: /Users/baptistehessel/Documents/DAJ/MEP/montageIA/data

[FILES]
chemin_dico: /Users/baptistehessel/Documents/DAJ/SIM/fichiers_pickle/FlatDict
chemin_avoid: /Users/baptistehessel/Documents/DAJ/SIM/fichiers_pickle/Avoid
chemin_mcavoid: /Users/baptistehessel/Documents/DAJ/SIM/fichiers_pickle/MC_Avoid

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

[OPTIMIZATION]
# The importance we give to the variance with respect to the score of fit.
# float between 0 and 1. The closer to one, the more important.
coef_var: 0.5

[VERBOSE]
verbose: 1

[CUSTOMERS]
CM: 6001
CM1:6002
CF: 6003