#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 16:51:06 2021

@author: baptiste hessel

Juste some changes about the distribution of the labels minor, ter, sec

"""
import pickle
import matplotlib.pyplot as plt
# %cd /Users/baptistehessel/Documents/DAJ/MEP/montageIA/bin
import shows_image_page
import numpy as np
import matplotlib.patches as mpatches

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
    dict_rub_matrix[rubric] = {}

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
    dict_rub_matrix[rubric]['mean'] = np.mean(matrix_rubric, axis=0)
    dict_rub_matrix[rubric]['matrix'] = matrix_rubric
    print(f"{rubric}: {np.mean(matrix_rubric, axis=0)}")

# Construction of a dict with the rubric associated to all the mean values
dd_rubrics = {}
for rubric, dict_matrix_rubric in dict_rub_matrix.items():
    dd_rubrics[rubric] = {}
    for feature, mean_value in zip(list_features, dict_matrix_rubric['mean']):
        dd_rubrics[rubric][feature] = mean_value

for rub, dict_mean in dd_rubrics.items():
    nb_pages = len(dict_rubric_pages[rub])
    if nb_pages > 200:
        print(f"Number of pages using the rubric: {nb_pages}")
        print(f"{rub}")
        for key, val in dict_mean.items():
            print(f"{key}: {val:.2f}")
        print("\n\n")


#%%

##############################################################################
#                       MEAN VECTOR FOR EACH TYPE                            #
##############################################################################


# Building of the big matrix X with all the articles
for i, (id_page, dicop) in enumerate(dict_pages.items()):
    for j, (ida, dicta) in enumerate(dicop['articles'].items()):
        vect_art = np.array([dicta[x] for x in list_features], ndmin=2)
        if i + j == 0:
            Xfull = vect_art
        else:
            Xfull = np.concatenate((Xfull, vect_art))

print(f"Xfull shape: {Xfull.shape}")

# PCA fit with that big matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
Z  = sc.fit_transform(Xfull)

pca = PCA(n_components=2)
pca.fit(Z)

print(f"Explained variance ration: {np.cumsum(pca.explained_variance_ratio_)}")


#%%

# Mean Vectors for each type
dict_type = {'main': {}, 'sec': {}, 'ter': {}, 'minor': {}}
list_types = ['isPrinc', 'isSec', 'isSub', 'isTer', 'isMinor']
for i, (id_page, dicop) in enumerate(dict_pages.items()):
    for j, (ida, dicta) in enumerate(dicop['articles'].items()):
        if dicta['isPrinc'] == 1:
            art_type = 'main'
        elif dicta['isSec'] == 1:
            art_type = 'sec'
        elif dicta['isSub'] == 1:
            art_type = 'sec'
        elif dicta['isTer'] == 1:
            art_type = 'ter'
        elif dicta['isMinor'] == 1:
            art_type = 'minor'
        else:
            print(f"Article without type: {ida}")
        dict_type[art_type][ida] = dicta

for key, val in dict_type.items():
    print(f"{key}. {len(val)}")

# Addition of the big matrix
for art_type, dict_one_type in dict_type.items():
    print(f"Type: {art_type}")
    for i, (ida, dicta) in enumerate(dict_one_type.items()):
        vect_art = np.array([dicta[x] for x in list_features], ndmin=2)
        if i == 0:
            Xtype = vect_art
        else:
            Xtype = np.concatenate((Xtype, vect_art))
    print(f"Shape of Xtype: {Xtype.shape}")
    dict_type[art_type]['matrix'] = Xtype
    mean_vect = sc.transform(np.mean(Xtype, axis=0, keepdims=True))
    dict_type[art_type]['mean_vector'] = mean_vect
    print(f"The mean vector: {mean_vect}")



#%%

# Reduction of the mean vector with PCA
for art_type, dict_one_type in dict_type.items():
    Xred = pca.transform(np.array(dict_one_type['mean_vector'], ndmin=2))
    dict_type[art_type]['mean_reduced'] = Xred
    print(f"Type: {art_type}")
    print(f"The reduction gives: {Xred}")


#%%

# Representation of the means
for art_type, dict_one_type in dict_type.items():
    plt.scatter(dict_one_type['mean_reduced'][0][0],
                dict_one_type['mean_reduced'][0][1])


#%%

colors = ['darkred', 'navy', 'green', 'orange']
# Representation of each vector
for type_art, color in zip(dict_type.keys(), colors):
    X_sc = sc.transform(dict_type[type_art]['matrix'])
    X_red = pca.transform(X_sc)
    plt.scatter(X_red[:, 0], X_red[:, 1],
                s=.5, alpha=0.8, c=color, label=type_art)
plt.legend()

print(f"Explained variance ration: {np.cumsum(pca.explained_variance_ratio_)}")


#%%

##############################################################################
#                 REPRESENTATION WITH PCA BY RUBRIC                          #
##############################################################################


mat_rub = dict_rub_matrix['CM_CORSE INFOS']['matrix']
Xrub_sc = sc.transform(mat_rub)
Xrub_red = pca.transform(Xrub_sc)
plt.scatter(Xrub_red[:, 0], Xrub_red[:, 1], c='purple', linewidths=1,
            s=1, alpha=0.3, label='CM_CORSE INFOS', marker='o')

mat_rub = dict_rub_matrix['CM_SPORTS']['matrix']
Xrub_sc = sc.transform(mat_rub)
Xrub_red = pca.transform(Xrub_sc)
plt.scatter(Xrub_red[:, 0], Xrub_red[:, 1], c='green', linewidths=1,
            s=2, alpha=0.3, label='CM_SPORTS', marker='.')

mat_rub = dict_rub_matrix['CM_LOC_SARTENE']['matrix']
Xrub_sc = sc.transform(mat_rub)
Xrub_red = pca.transform(Xrub_sc)
plt.scatter(Xrub_red[:, 0], Xrub_red[:, 1], c='red', linewidths=1,
            s=0.8, alpha=0.1, label='CM_LOC_SARTENE', marker='x')

mat_rub = dict_rub_matrix['CM_UNE']['matrix']
Xrub_sc = sc.transform(mat_rub)
Xrub_red = pca.transform(Xrub_sc)
plt.scatter(Xrub_red[:, 0], Xrub_red[:, 1], c='orange', linewidths=1,
            s=1.5, alpha=1, label='CM_UNE', marker='o')

mat_rub = dict_rub_matrix['CM_LOC_CALVI']['matrix']
Xrub_sc = sc.transform(mat_rub)
Xrub_red = pca.transform(Xrub_sc)
plt.scatter(Xrub_red[:, 0], Xrub_red[:, 1], c='blue', linewidths=1,
            s=1, alpha=0.075, label='CM_LOC_CALVI', marker='o')

purple_patch = mpatches.Patch(color='purple', label='CM_CORSE INFOS')
green_patch = mpatches.Patch(color='green', label='CM_SPORTS')
orange_patch = mpatches.Patch(color='orange', label='CM_UNE')
red_patch = mpatches.Patch(color='red', label='CM_LOC_SARTENE')
blue_patch = mpatches.Patch(color='blue', label='CM_LOC_CALVI')
plt.legend(handles=[purple_patch, green_patch, orange_patch, red_patch,
                    blue_patch])



