#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 16:51:06 2021

@author: baptiste hessel

Juste some changes about the distribution of the labels minor, ter, sec

"""
import pickle
import os
os.chdir('/Users/baptistehessel/Documents/DAJ/MEP/montageIA/bin/')
import creation_dict_bdd
# Histogram to see the distribution of the scores of the articles
import matplotlib.pyplot as plt


path_customer = '/Users/baptistehessel/Documents/DAJ/MEP/montageIA/data/CM2/'

with open(path_customer + 'dict_pages', 'rb') as file:
    dict_pages = pickle.load(file)

list_scores = []
for id_page, dict_page in dict_pages.items():
    for id_article, dict_article in dict_page['articles'].items():
        list_scores.append(dict_article['score'])

plt.hist(list_scores, bins=30)
plt.show()


# Let's see the distribution between the labels
list_labels = []
for id_page, dict_page in dict_pages.items():
    for ida, dicta in dict_page['articles'].items():
        if dicta['isPrinc'] == 1:
            list_labels.append(0)
        if dicta['isSec'] == 1:
            list_labels.append(1)
        if dicta['isSub'] == 1:
            list_labels.append(1)
        if dicta['isTer'] == 1:
            list_labels.append(2)
        if dicta['isMinor'] == 1:
            list_labels.append(3)
count_labels = [(label, list_labels.count(label)) for label in set(list_labels)]
print(f"The distribution of the labels: {count_labels}")

# Change of the labels between minor and ter
for id_page, dict_page in dict_pages.items():
    dict_page = creation_dict_bdd.Determ_Nat_Art(dict_page)

# Let's see the distribution between the labels
list_labels = []
for id_page, dict_page in dict_pages.items():
    for ida, dicta in dict_page['articles'].items():
        if dicta['isPrinc'] == 1:
            list_labels.append(0)
        elif dicta['isSec'] == 1:
            list_labels.append(1)
        elif dicta['isSub'] == 1:
            list_labels.append(1)
        elif dicta['isTer'] == 1:
            dicta['isMinor'] = 0
            list_labels.append(2)
        elif dicta['isMinor'] == 1:
            list_labels.append(3)
        else:
            print("No label")
count_labels = [(label, list_labels.count(label))
                for label in set(list_labels)]
print(f"The distribution of the labels: {count_labels}")


with open(path_customer + 'dict_pages', 'wb') as file:
    pickle.dump(dict_pages, file)



