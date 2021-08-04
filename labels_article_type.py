#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 16:51:06 2021

@author: baptiste hessel

Juste some changes about the distribution of the labels minor, ter, sec

"""
import pickle
import matplotlib.pyplot as plt
import shows_image_page
import numpy as np

path_customer = '/Users/baptistehessel/Documents/DAJ/MEP/montageIA/data/CM/'

with open(path_customer + 'dict_pages', 'rb') as file:
    dict_pages = pickle.load(file)

with open(path_customer + 'dict_layouts', 'rb') as file:
    dict_layouts = pickle.load(file)


#%%

##############################################################################
#                           DISTRIBUTION SCORES                              #
##############################################################################

list_scores = []
for id_page, dict_page in dict_pages.items():
    for id_article, dict_article in dict_page['articles'].items():
        list_scores.append(dict_article['score'])
plt.hist(list_scores, bins=50)
plt.xlabel('Score')
plt.ylabel('Articles')
plt.show()


#%%

##############################################################################
#                           DISTRIBUTION LABELS                              #
##############################################################################

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

rbar = [list_labels.count(lab) / len(list_labels) for lab in set(list_labels)]
plt.bar(range(4), rbar, width=0.3,
        edgecolor='black', linewidth=0.5)
plt.xticks(range(4), ['Main', 'Sec', 'Ter', 'Minor'])
plt.show()


#%%

##############################################################################
#                              INFOS RUBRICS                                 #
##############################################################################

# Let's gather pages by rubric
rubrics = {}
for idp, dicop in dict_pages.items():
    rub = dicop['catName']
    if rub in rubrics.keys():
        rubrics[dicop['catName']].append(dicop['pageTemplate'])
    else:
        rubrics[dicop['catName']] = [dicop['pageTemplate']]

print(f"All the rubrics:")
for i in range(0, len(list(rubrics.keys())), 2):
    size = len(list(rubrics.keys())[i:i + 2])
    print(("{:<35} "*size).format(*list(rubrics.keys())[i:i + 2]))
print("\n\n")

for key, list_ids_layout in rubrics.items():
    count_list = [(x, list_ids_layout.count(x)) for x in set(list_ids_layout)]
    count_list.sort(key=lambda x: x[1], reverse=True)
    tot_rub = sum((count for _, count in count_list))
    # We print only rubrics with more than 50 pages published
    if tot_rub > 100:
        print(f"The rubric: {key}")
        print("The list with the count:")
        for tem, count in count_list:
            if count > 5:
                print(f"id template: {tem:<15} {count:>5} pages")
        print("\n\n")


#%%

##############################################################################
#                         IMAGES TEMPLATES RUBRICS                           #
##############################################################################

dict_rb = {}
for key, l_ids_layout in rubrics.items():
    count_list = [(idl, l_ids_layout.count(idl)) for idl in set(l_ids_layout)]
    dict_rb[key] = {'ids_layout': list_ids_layout, 'count_list': count_list}

for rub, dict_rub in dict_rb.items():
    for id_lay, countl in dict_rub['count_list']:
        if countl > 10:
            array_to_show = dict_layouts[id_lay]['array']
            try:
                shows_image_page.RepPageSlide(array_to_show,
                                              ["euh"]*len(array_to_show),
                                              title=rub)
            except Exception as e:
                print(f"An exception: {e}")
                print(f"The array: {array_to_show}")
                print(f"The rubric: {rub}")

# CM_LOC_SARTENE
# CM_LOC_PORTO-VECCHIO
# CM_LOC_AJACCIO
# CM_LOC_PLAINE_ORIENTALE
# CM_LOC_BASTIA
# CM_LOC_CORTE
# CM_LOC_CALVI
# -> Some similarities in the layouts used


#%%

##############################################################################
#                               ANALYSIS PAGE                                #
##############################################################################

# We gather the pages that have the same distribution and we plot that.
for rub, dict_rub in dict_rb.items():
    dict_rb[rub]['count'] = sum((nb for _, nb in dict_rub['count_list']))
    dict_rb[rub]['variety'] = len(dict_rb[rub]['count_list'])

dict_rb = dict(sorted(dict_rb.items(), key=lambda x: x[1]['count']))

for rub, dict_rub in dict_rb.items():
    sorted_list = dict_rub['count_list']
    sorted_list.sort(key=lambda x: x[1], reverse=True)
    f = lambda x: (x[0], round(x[1] / dict_rub['count'], 2))
    sorted_list = list(map(f, sorted_list))
    dict_rub['count_list'] = sorted_list

# Let's look at the distribution of the type of the articles in the page
list_types = ['isPrinc', 'isSec', 'isSub', 'isTer', 'isMinor']
list_type_simple = [0, 1, 1, 2, 3]
list_distributions = []
for idp, dicop in dict_pages.items():
    list_types_page = []
    for ida, dica in dicop['articles'].items():
        type_current = [dica[x] for x in list_types]
        # Depending on where is the one
        int_type = list_type_simple[type_current.index(1)]
        list_types_page.append(int_type)
    list_types_page.sort()
    list_distributions.append(tuple(list_types_page))

list_count = [(dis, list_distributions.count(dis))
              for dis in set(list_distributions)]

list_count.sort(key=lambda x: x[1], reverse=True)
print("The list with the counts.")
for dis, countl in list_count:
    if countl > 100:
        print(f"The distri {dis}. The count: {countl}")

# Dict to simplify the distri
dict_distri = {dis: i for i, (dis, _) in enumerate(list_count)}

list_simpler = []
for dis, countl in list_count:
    if countl > 50:
        list_simpler += [dict_distri[dis]] * countl

plt.hist(list_simpler, bins=100)


#%%

##############################################################################
#                  ANALYSIS PAGES WITH N ARTICLES                            #
##############################################################################


dict_stats_rep = {i: [] for i in range(2, 9)}

for rep, count in list_count:
    try:
        dict_stats_rep[len(rep)].append((rep, count))
    except:
        pass

for nbarts, listrep in dict_stats_rep.items():
    listrep.sort(key=lambda x: x[1], reverse=True)
    dict_stats_rep[nbarts] = listrep


for nbarts, listrep in dict_stats_rep.items():
    tot = sum((count for _, count in listrep))
    print(f"the TOTAL: {tot}")

    rbar = []
    leg = []
    for i, (rep, count) in enumerate(listrep):
        print(f"The repartition: {rep}. PCTG: {count/tot:.1%}")
        rbar.append(count/tot)
        leg.append(str(rep))

        if i == 6:
            break
        elif count/tot < 0.03:
            break

    if i > 4:
        plt.bar(range(len(rbar)), rbar, width=0.3, edgecolor='black',
                linewidth=0.5)
        plt.xticks(range(len(leg)), leg, rotation=45)
        plt.show()
    else:
        plt.bar(range(len(rbar)), rbar, width=0.3, edgecolor='black',
                linewidth=0.5)
        plt.xticks(range(len(leg)), leg)
        plt.show()

#%%

##############################################################################
#                     MEAN VECTOR FOR EACH RUBRIC                            #
##############################################################################

list_features = ["nbSign","nbBlock", "abstract", "syn", "exergue",
                 "title", "secTitle", "supTitle", "subTitle", "nbPhoto",
                 "aireImg", "aireTot", "petittitre", "quest_rep", "intertitre"]

dict_rubric_pages = {}
for id_page, dicop in dict_pages.items():
    rubric_page = dicop['catName']
    id_page = dicop['melodyId']
    try:
        dict_rubric_pages[rubric_page][id_page] = dicop
    except:
        dict_rubric_pages[rubric_page] = {id_page: dicop}

for key, dict_rub in dict_rubric_pages.items():
    print(f"{key}: {len(dict_rub)}")

# Construction of the matrix with all the vectors
dict_rub_matrix = {}
for rubric, dict_rubric in dict_rubric_pages.items():

    for i, (id_page, dicop) in enumerate(dict_rubric.items()):
        for j, dicta in enumerate(dicop['articles'].values()):
            vect_art = np.array([dicta[x] for x in list_features],
                                ndmin=2)
            if j == 0:
                vect_page = vect_art
            else:
                vect_page = np.concatenate((vect_page, vect_art))
        mean_vect_page = np.mean(vect_page, axis=0, keepdims=True)
        if i == 0:
            matrix_rubric = mean_vect_page
        else:
            matrix_rubric = np.concatenate((matrix_rubric, mean_vect_page))
    dict_rub_matrix[rubric] = np.mean(matrix_rubric, axis=0)
    print(f"{rubric}: {np.mean(matrix_rubric, axis=0)}")

# Construction of a dict with the rubric associated to all the mean values
dd_rubrics = {}
for rubric, matrix_mean in dict_rub_matrix.items():
    dd_rubrics[rubric] = {}
    for feature, mean_value in zip(list_features, matrix_mean):
        dd_rubrics[rubric][feature] = mean_value

for rub, dict_mean in dd_rubrics.items():
    print(f"{rub}")
    for key, val in dict_mean.items():
        print(f"{key}: {val:.2f}")
    print("\n\n")












