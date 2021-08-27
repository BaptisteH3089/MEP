#!/usr/bin/env python3
# -*- coding: utf-8 -*-
[LOG]
path_log: /var/log/daj/montageia/montage_ia.log

[DATA]
path_data: /data/montageia/corpus/

[FEATURES]
list_features: ["nbSign", "nbBlock", "abstract", "syn", "exergue",
                "title", "secTitle", "supTitle", "subTitle", "nbPhoto",
                "aireImg", "aireTot", "petittitre", "quest_rep", "intertitre"]

[CONSTRAINTS]
# Constraints for the layouts
tol_total_area: 0.7
tol_nb_images: 1
tol_score_min: 0.05
# Constraints for the modules
tol_area_images_mod: 0.7
tol_area_text_mod: 0.8
tol_total_area_mod: 0.5

[OPTIMIZATION]
# The importance we give to the variance with respect to the score of fit.
# float between 0 and 1. The closer to one, the more important.
coef_var: 0.5

[VERBOSE]
verbose: 0

[CUSTOMERS]
CM: 6001
CF: 6002
