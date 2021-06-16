#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 15:57:50 2021

@author: baptistehessel
"""

############################################
#
# Script Creation databse files
#
############################################

import pickle


dico_arts = {}

rep_data = '/Users/baptistehessel/Documents/DAJ/MEP/montageIA copie/data/CM'
with open(rep_data + '/dico_articles', 'wb') as file:
    pickle.dump(dico_arts, file)



# I have to be sure of the kind of input I will receive, if one day I receive
# something.