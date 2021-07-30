#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 10:42:11 2021

@author: baptistehessel

All these functions are used to validate the models that predict the type of
an article.

"""
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import MultinomialNB, GaussianNB, ComplementNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import sys
import re
import xgboost as xgb
import time


class MyException(Exception):
    pass


def Clean(txt):
    """
    Remplace les caractères qui ne sont pas alpha-numériques par des espaces.
    """
    return re.sub(re.compile('\W+'), ' ', txt)


def CleanCar(txt):
    """
    Enlève les caractères entre deux espaces.
    """
    return ' '.join([word for word in txt.split() if len(word) > 1])


def LemTxt(txt, dict_lem):
    return ' '.join([dict_lem[word] if word in dict_lem.keys() else word
                     for word in txt.split()])


def CreationCorpusLabels(dict_pages):
    init = False
    for id_page, dict_page in dict_pages.items():
        for id_article, dict_article in dict_page['articles'].items():
            if dict_article['isPrinc'] == 1:
                label = 0
            elif dict_article['isSec'] == 1:
                label = 1
            elif dict_article['isSub'] == 1:
                label = 1
            elif dict_article['isTer'] == 1:
                label = 2
            elif dict_article['isMinor'] == 1:
                label = 3
            else:
                # Article with no label
                print("Article with no label")
                continue
            if len(dict_article['content']) > 10:
                if init is False:
                    corpus = [dict_article['content']]
                    Y = [label]
                    init = True
                else:
                    corpus.append(dict_article['content'])
                    Y.append(label)
    return corpus, Y


def CleanCorpus(corpus, dict_lem):
    # Remove the non-alphanumeric characters
    corpus = list(map(lambda x: CleanCar(Clean(x.lower())), corpus))
    # Replace words by their lemma
    corpus = list(map(lambda x: LemTxt(x, dict_lem), corpus))
    return corpus


def MatricesXYArticleContent(dict_pages, dict_lem, max_features=None,
                             ngram_range=(1, 2)):
    corpus, Y = CreationCorpusLabels(dict_pages)
    print(f"The length of corpus: {len(corpus)}.")
    print(f"The length of Y: {len(Y)}.")
    # The content of the articles
    corpus = CleanCorpus(corpus, dict_lem)
    # Converting text into vector with tf-idf
    vectorizer = TfidfVectorizer(max_features=max_features,
                                 ngram_range=ngram_range)
    X = vectorizer.fit_transform(corpus)
    return X, np.array(Y)


# Creation of the vectors X and Y
def MatricesTypeArticle(dict_pages, list_features):
    """
    Parameters
    ----------
    dict_pages : dictionary
        The usual dict with all pages.
    list_features : list of strings
        The features to use. They correspond to the keys of the dicta.

    Raises
    ------
    MyException
        DESCRIPTION.

    Returns
    -------
    X : numpy array
        A matrix X with nbcols = len(list_features) + 3 (nb_nums, nb_mails,
        mean number of words by paragraph text).
    Y : numpy array
        A matrix column with the labels (0, 1, 2, or 3).
    """
    X, Y = [], []
    for idp, dicop in dict_pages.items():
        # We check that there is one article with more than 2000 signs
        # nb_signs = [dicta['nbSign'] for dicta in dicop['articles'].values()]
        # if (np.array(nb_signs) > 2000).any():
        for ida, dicta in dicop['articles'].items():
            if len(dicta['content']) > 10:
                txt = dicta['content']
                pattern_num = r'(\d\d\.\d\d\.\d\d\.\d\d\.\d\d\.)'
                nb_nums = len(re.findall(pattern_num , txt))
                pattern_mel = r'\w+@\w+\.com'
                nb_mels = len(re.findall(pattern_mel , txt))
                dicta['aireTot'] = dicta['width'] * dicta['height']
                # Add the mean number of signs of the blocks 'texte'
                l_nbs = []
                args_z = [dicta['typeBlock'], dicta['nbSignBlock']]
                for type_block, nb_sign in zip(*args_z):
                    if type_block == 'texte':
                        l_nbs.append(nb_sign)
                mean_nbs = sum(l_nbs) / len(l_nbs) if len(l_nbs) > 0 else 0
                try:
                    other_features = [nb_nums, nb_mels, mean_nbs]
                    x = [dicta[z] for z in list_features] + other_features
                    x = np.array(x)
                except Exception as e:
                    print(e)
                    print(dicta)
                    sys.exit()
                if dicta['isPrinc'] == 1:
                    y = 0
                elif max(dicta['isSec'], dicta['isSub']) == 1:
                    y = 1
                elif dicta['isTer'] == 1:
                    y = 2
                elif dicta['isMinor'] == 1:
                    y = 3
                else:
                    print(dicta)
                    raise MyException("No label for this article")
                X.append(x)
                Y.append(y)
    return np.array(X), np.array(Y)


##############################################################################
#                                                                            #
#                 MODEL PREDS CONTENT AND FEATURES                           #
#                                                                            #
##############################################################################

def ModelFeatures(X, Y):
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    rfc = RandomForestClassifier()
    linsvc = make_pipeline(StandardScaler(), LinearSVC())
    svc = make_pipeline(StandardScaler(), SVC())
    mnb = MultinomialNB()
    gnb = GaussianNB()
    cnb = ComplementNB()
    bnb = BernoulliNB()
    ada = AdaBoostClassifier()
    gbc = GradientBoostingClassifier()
    mean_lr, mean_rfc, mean_linsvc, mean_svc, mean_xgb = 0, 0, 0, 0, 0
    mean_mnb, mean_gnb, mean_cnb, mean_bnb, mean_ada = 0, 0, 0, 0, 0
    mean_gbc = 0
    dict_duration = {}
    skf = StratifiedKFold(n_splits=3, shuffle=True)
    for i, (train, test) in enumerate(skf.split(X, Y)):

        t = time.time()
        gbc.fit(X[train], Y[train])
        preds_gbc = gbc.predict(X[test])
        score_gbc = f1_score(Y[test], preds_gbc, average='macro')
        mean_gbc += score_gbc
        dict_duration['GBC'] = time.time() - t

        t_xgb = time.time()
        dtrain = xgb.DMatrix(X[train], label=Y[train])
        dtest = xgb.DMatrix(X[test], label=Y[test])
        param = {'objective': 'multi:softmax',
                 'num_class': max(set(Y)) + 1,
                 'eval_metric': 'mlogloss'}
        print(f"param['num_class']: {param['num_class']}")
        bst = xgb.train(param, dtrain, num_boost_round=15)
        preds_xgb = bst.predict(dtest)
        score_xgb = f1_score(Y[test], preds_xgb, average='macro')
        mean_xgb += score_xgb
        dict_duration['XGB'] = time.time() - t_xgb

        t = time.time()
        lr.fit(X[train], Y[train])
        preds_lr = lr.predict(X[test])
        score_lr = f1_score(Y[test], preds_lr, average='macro')
        mean_lr += score_lr
        dict_duration['LR'] = time.time() - t

        t = time.time()
        rfc.fit(X[train], Y[train])
        preds_rfc = rfc.predict(X[test])
        score_rfc = f1_score(Y[test], preds_rfc, average='macro')
        mean_rfc += score_rfc
        dict_duration['RFC'] = time.time() - t

        t = time.time()
        linsvc.fit(X[train], Y[train])
        preds_linsvc = linsvc.predict(X[test])
        score_linsvc = f1_score(Y[test], preds_linsvc, average='macro')
        mean_linsvc += score_linsvc
        dict_duration['LINSVC'] = time.time() - t

        t = time.time()
        svc.fit(X[train], Y[train])
        preds_svc = svc.predict(X[test])
        score_svc = f1_score(Y[test], preds_svc, average='macro')
        mean_svc += score_svc
        dict_duration['SVC'] = time.time() - t

        t = time.time()
        mnb.fit(X[train], Y[train])
        preds_mnb = mnb.predict(X[test])
        score_mnb = f1_score(Y[test], preds_mnb, average='macro')
        mean_mnb += score_mnb
        dict_duration['MNB'] = time.time() - t

        t = time.time()
        gnb.fit(X[train], Y[train])
        preds_gnb = gnb.predict(X[test])
        score_gnb = f1_score(Y[test], preds_gnb, average='macro')
        mean_gnb += score_gnb
        dict_duration['GNB'] = time.time() - t

        t = time.time()
        cnb.fit(X[train], Y[train])
        preds_cnb = cnb.predict(X[test])
        score_cnb = f1_score(Y[test], preds_cnb, average='macro')
        mean_cnb += score_cnb
        dict_duration['CNB'] = time.time() - t

        t = time.time()
        bnb.fit(X[train], Y[train])
        preds_bnb = bnb.predict(X[test])
        score_bnb = f1_score(Y[test], preds_bnb, average='macro')
        mean_bnb += score_bnb
        dict_duration['BNB'] = time.time() - t

        t = time.time()
        ada.fit(X[train], Y[train])
        preds_ada = ada.predict(X[test])
        score_ada = f1_score(Y[test], preds_ada, average='macro')
        mean_ada += score_ada
        dict_duration['ADA'] = time.time() - t

        print(f"FOLD {i}")
        print(f"F1-Score {'LR':>15} {score_lr:>15.3f}")
        print(f"F1-Score {'LINSVC':>15} {score_linsvc:>15.3f}")
        print(f"F1-Score {'SVC':>15} {score_svc:>15.3f}")
        print(f"F1-Score {'MNB':>15} {score_mnb:>15.3f}")
        print(f"F1-Score {'GNB':>15} {score_gnb:>15.3f}")
        print(f"F1-Score {'CNB':>15} {score_bnb:>15.3f}")
        print(f"F1-Score {'BNB':>15} {score_bnb:>15.3f}")
        print(f"F1-Score {'ADA':>15} {score_ada:>15.3f}")
        print(f"F1-Score {'RFC':>15} {score_rfc:>15.3f}")
        print(f"F1-Score {'GBC':>15} {score_gbc:>15.3f}")
        print(f"F1-Score {'XGB':>15} {score_xgb:>15.3f}")
        target_names = ['Main', 'Sec', 'Ter', 'Minor']
        args_cl = [Y[test], preds_rfc]
        report = classification_report(*args_cl, target_names=target_names)
        print(f"RFC classification report: \n{report}\n\n")

    print("{:-^80}".format("MEAN SCORES"))
    print(f"F1-Score {'LR':>15} {mean_lr/3:>15.3f}")
    print(f"F1-Score {'LINSVC':>15} {mean_linsvc/3:>15.3f}")
    print(f"F1-Score {'SVC':>15} {mean_svc/3:>15.3f}")
    print(f"F1-Score {'MNB':>15} {mean_mnb/3:>15.3f}")
    print(f"F1-Score {'GNB':>15} {mean_gnb/3:>15.3f}")
    print(f"F1-Score {'CNB':>15} {mean_bnb/3:>15.3f}")
    print(f"F1-Score {'BNB':>15} {mean_bnb/3:>15.3f}")
    print(f"F1-Score {'ADA':>15} {mean_ada/3:>15.3f}")
    print(f"F1-Score {'GBC':>15} {mean_gbc/3:>15.3f}")
    print(f"F1-Score {'XGB':>15} {mean_xgb/3:>15.3f}")
    print(f"F1-Score {'RFC':>15} {mean_rfc/3:>15.3f}")

    for key, val in dict_duration.items():
        print(f"CLF {key:>20}, duration: {val:.3f} sec.")


##############################################################################
#                                                                            #
#                 MODEL PREDS CONTENT AND FEATURES                           #
#                                                                            #
##############################################################################


def ModelMix(X, Y, Xc, Yc):
    """
    Parameters
    ----------
    X : numpy array
        Matrix with the features.
    Y : numpy array
        Matrix column with the labels.
    Xc : sparse matrix
        Matrix obtained with the scikit-learn tfidfvectorizer.
    Yc : numpy array
        Should be exactly the same as Y.

    Returns
    -------
    None.

    Cross-validate the several models combining the predictions of both
    methods. It gives the scores for the model with the content and with the
    features.
    It also does a kind a combination of the predictions of the two models.
    We select the prediction of the model content only if the max proba pred
    of the model content > 0.85 and the one of the model features < 0.55.
    For now, the combination of the two models gives a score very similar but
    a bit lower.

    """
    lr = make_pipeline(StandardScaler(with_mean=False),
                       LogisticRegression(max_iter=1000))
    rfc = RandomForestClassifier()
    lin_svc = LinearSVC()
    ss = ShuffleSplit(n_splits=3, random_state=0)
    mean_rfc = []
    mean_lr = []
    mean_lin_svc = []
    mean_mix = []
    list_classes = ['Main', 'Sec', 'Ter', 'Minor']
    for train, test in ss.split(X, Y):
        # MODEL FEATURES - RFC
        rfc.fit(X[train], Y[train])
        preds_rfc = rfc.predict(X[test])
        probas_rfc = rfc.predict_proba(X[test])
        score_rfc = f1_score(Y[test], preds_rfc, average='macro')
        conf_svc = confusion_matrix(Y[test], preds_rfc)
        args_cl = [Y[test], preds_rfc]
        report = classification_report(*args_cl, target_names=list_classes)
        print(f"F1-score features: {score_rfc:.3f}")
        print(f"\nConfusion Matrix RFC: \n{conf_svc}\n")
        print(f"Classification report features \n {report}")
        # MODEL CONTENT - LR, LINEAR SVC
        lr.fit(Xc[train], Yc[train])
        lin_svc.fit(Xc[train], Yc[train])
        preds_lin_svc = lin_svc.predict(Xc[test])
        preds_lr = lr.predict(Xc[test])
        probas_lr = lr.predict_proba(Xc[test])
        # f1-scores, confusion matrix and
        score_lr = f1_score(Y[test], preds_lr, average='macro')
        score_lin_svc = f1_score(Y[test], preds_lin_svc, average='macro')
        conf_lr = confusion_matrix(Y[test], preds_lr)
        args_rep = [Y[test], preds_lr]
        rep_ctt = classification_report(*args_rep, target_names=list_classes)
        print(f"F1-score LINEAR SVC content: {score_lin_svc:.3f}")
        print(f"F1-score LR content: {score_lr:.3f}")
        print(f"Confusion Matrix LR: \n{conf_lr}\n")
        print(f"Classification report content \n {rep_ctt}")
        # Average of the probas
        mean_preds = np.mean((probas_rfc, probas_lr), axis=0)
        preds_avg = np.argmax(mean_preds, axis=1)
        score_avg = f1_score(Y[test], preds_avg, average='macro')
        # We only select the strong predictions of the model content.
        preds_mix = []
        for preds_ctt, preds_ft in zip(probas_lr, probas_rfc):
            if max(preds_ctt) > 0.85:
                if max(preds_ft) < 0.55:
                    preds_mix.append(np.argmax(preds_ctt, axis=0))
                else:
                    preds_mix.append(np.argmax(preds_ft, axis=0))
            else:
                preds_mix.append(np.argmax(preds_ft, axis=0))
        score_mix = f1_score(Y[test], preds_mix, average='macro')
        print(f"Score average: {score_avg:.3f}")
        print(f"Score mix: {score_mix:.3f}")
        mean_lin_svc.append(score_lin_svc)
        mean_mix.append(score_mix)
        mean_rfc.append(score_rfc)
        mean_lr.append(score_lr)
    # The mean predictions
    print(f"The mean score LINEAR SVC: {np.mean(mean_lin_svc):.3f}")
    print(f"The mean score MIX: {np.mean(mean_mix):.3f}")
    print(f"The mean score RFC: {np.mean(mean_rfc):.3f}")
    print(f"The mean score LR: {np.mean(mean_lr):.3f}")


def OptiThresholds(X, Y, Xc, Yc):
    """

    Parameters
    ----------
    X : numpy array
        Matrix with the features.
    Y : numpy array
        Matrix column with the labels.
    Xc : sparse matrix
        Matrix obtained with the scikit-learn tfidfvectorizer.
    Yc : numpy array
        Should be exactly the same as Y.

    Returns
    -------
    best_params : TYPE
        DESCRIPTION.

    Try several combinations of threshold for the selection of the prediction
    of the model content.

    """
    lr = make_pipeline(StandardScaler(with_mean=False),
                       LogisticRegression(max_iter=1000))
    rfc = RandomForestClassifier()
    ss = ShuffleSplit(n_splits=2)
    for train, test in ss.split(X, Y):
        # MODEL FEATURES
        rfc.fit(X[train], Y[train])
        preds_rfc = rfc.predict(X[test])
        score_features = f1_score(Y[test], preds_rfc, average='macro')
        probas_rfc = rfc.predict_proba(X[test])
        # MODEL CONTENT
        lr.fit(Xc[train], Yc[train])
        preds_lr = lr.predict(Xc[test])
        score_content = f1_score(Y[test], preds_lr, average='macro')
        probas_lr = lr.predict_proba(Xc[test])
        str_print = "The reference score obtained with only the"
        print(f"{str_print} features: {score_features:.4f}")
        print(f"{str_print} content: {score_content:.4f}")
        # We only select the strong predictions of the model content.
        best_params = []
        best_of_all = []
        for proba_ctt in np.linspace(0.55, 0.95, 20):
            for proba_feat in np.linspace(0.55, 0.85, 20):
                preds_mix = []
                for preds_ctt, preds_ft in zip(probas_lr, probas_rfc):
                    if max(preds_ctt) > proba_ctt:
                        if max(preds_ft) < proba_feat:
                            preds_mix.append(np.argmax(preds_ctt, axis=0))
                        else:
                            preds_mix.append(np.argmax(preds_ft, axis=0))
                    else:
                        preds_mix.append(np.argmax(preds_ft, axis=0))
                score_mix = f1_score(Y[test], preds_mix, average='macro')
                best_params.append((score_mix, proba_ctt, proba_feat))
        best = max(best_params, key=lambda x: x[0])
        print(f"The best score obtained with the combination of preds is: {best}")
        best_of_all.append(best)
    return best_of_all

