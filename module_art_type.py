#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 10:42:11 2021

@author: baptistehessel

All these functions are used to validate the models that predict the type of
an article.

"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
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
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import explained_variance_score
from xgboost import XGBRegressor
import xgboost as xgb
import numpy as np
import sys
import re
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


def MatricesXYArticleContent(dict_pages,
                             dict_lem,
                             max_features=None,
                             ngram_range=(1, 2)):
    """
    Creates the matrices X and Y to train the model that predicts the type of
    an article with the content of the article.

    Parameters
    ----------
    dict_pages: dict
        The big dict with the data.

    dict_lem: dict
        The dict used to lemmatized. It is part of our data.
        It associates words and lemma.

    max_features: int, optional
        The max number of features (words) in X.
        The default is None.

    ngram_range: tuple with 2 integers, optional
        If (1, 1) we only use single words, if (1, 2) we use the single words
        and the 2 grams.
        The default is (1, 2).

    Returns
    -------
    X: sparse matrix
        Contains the tf-idf score of each word for each article.

    Y: numpy array
        Vector column with the labels of the articles.

    """
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



def MatricesTypeArticle(dict_pages, list_features, score=False):
    """
    The matrices X and Y used to validate the models with the features that
    predict the type of an article.

    Parameters
    ----------
    dict_pages: dictionary
        The usual dict with all pages.

    list_features: list of strings
        The features to use. They correspond to the keys of the dicta.

    score: bool, optional
        If score is False, we store the labels in Y (classification).
        If score is True, we store the scores in Y (regression).

    Raises
    ------
    MyException
        If there is an article without label.

    Returns
    -------
    X: numpy array
        A matrix X with nbcols = len(list_features) + 3 (nb_nums, nb_mails,
        mean number of words by paragraph text).

    Y: numpy array
        A matrix column with the labels (0, 1, 2, or 3).
    """
    X, Y = [], []
    for idp, dicop in dict_pages.items():
        # We check that there is one article with more than 2000 signs
        # nb_signs = [dicta['nbSign'] for dicta in dicop['articles'].values()]
        # if (np.array(nb_signs) > 2000).any():
        for ida, dicta in dicop['articles'].items():
            if len(dicta['content']) > 10:
                # txt = dicta['content']
                # pattern_num = r'(\d\d\.\d\d\.\d\d\.\d\d\.\d\d\.)'
                # nb_nums = len(re.findall(pattern_num , txt))
                # pattern_mel = r'\w+@\w+\.com'
                # nb_mels = len(re.findall(pattern_mel , txt))
                dicta['aireTot'] = dicta['width'] * dicta['height']
                # Add the mean number of signs of the blocks 'texte'
                l_nbs = []
                args_z = [dicta['typeBlock'], dicta['nbSignBlock']]
                for type_block, nb_sign in zip(*args_z):
                    if type_block == 'texte':
                        l_nbs.append(nb_sign)
                # mean_nbs = sum(l_nbs) / len(l_nbs) if len(l_nbs) > 0 else 0
                try:
                    # other_features = [nb_nums, nb_mels, mean_nbs]
                    # x = [dicta[z] for z in list_features] + other_features
                    x = [dicta[z] for z in list_features]
                    x = np.array(x)
                except Exception as e:
                    print(e)
                    print(dicta)
                    sys.exit()
                # We store the score of the article.
                if score:
                    try:
                        y = dicta['score']
                    except Exception as e:
                        print(f"Exception MatricesTypeArticle: {e}")
                        print(f"dict_article: {dicta}")
                # We store the label of the article (the type).
                else:
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


def CrossValidationRegression(X, Y):
    """
    Cross-validation of the regression models to predict the scores.

    Parameters
    ----------
    X: numpy array
        The matrix with the vector articles concatenated.

    Y: numpy array
        A column vector with the scores.

    Returns
    -------
    dict_results: dict
        The dict with all the scores and durations.

    """

    linreg = linear_model.LinearRegression()
    ridge = linear_model.Ridge(alpha=.5)
    lasso = linear_model.Lasso(alpha=0.1)
    elastic = linear_model.ElasticNet(random_state=0)
    rfr = RandomForestRegressor()
    xgbreg = XGBRegressor()
    ss = ShuffleSplit(n_splits=5)

    regr_names = ['linreg', 'ridge', 'lasso', 'elastic', 'rfr', 'xgbreg']
    dict_results = {x: {'mse': [], 'mae': [], 'evs': [], 'duration': []}
                    for x in regr_names}

    for i, (train, test) in enumerate(ss.split(X, Y)):

        print(f"FOLD: {i}")

        # Training of the models
        t = time.time()
        linreg.fit(X[train], Y[train])
        t_linreg = time.time() - t
        dict_results['linreg']['duration'].append(t_linreg)

        t = time.time()
        ridge.fit(X[train], Y[train])
        t_ridge = time.time() - t
        dict_results['ridge']['duration'].append(t_ridge)

        t = time.time()
        lasso.fit(X[train], Y[train])
        t_lasso = time.time() - t
        dict_results['lasso']['duration'].append(t_lasso)

        t = time.time()
        elastic.fit(X[train], Y[train])
        t_elastic = time.time() - t
        dict_results['elastic']['duration'].append(t_elastic)

        t = time.time()
        rfr.fit(X[train], Y[train])
        t_rfr = time.time() - t
        dict_results['rfr']['duration'].append(t_rfr)

        t = time.time()
        xgbreg.fit(X[train], Y[train])
        t_xgbreg = time.time() - t
        dict_results['xgbreg']['duration'].append(t_xgbreg)

        # Preds of the models
        preds_linreg = linreg.predict(X[test])
        preds_ridge = ridge.predict(X[test])
        preds_lasso = lasso.predict(X[test])
        preds_elastic = elastic.predict(X[test])
        preds_rfr = rfr.predict(X[test])
        preds_xgbreg = xgbreg.predict(X[test])

        # Scores of the models
        # MSE
        mse_linreg = mean_squared_error(Y[test], preds_linreg)
        mse_ridge = mean_squared_error(Y[test], preds_ridge)
        mse_lasso = mean_squared_error(Y[test], preds_lasso)
        mse_elastic = mean_squared_error(Y[test], preds_elastic)
        mse_rfr = mean_squared_error(Y[test], preds_rfr)
        mse_xgbreg = mean_squared_error(Y[test], preds_xgbreg)

        # MAE
        mae_linreg = mean_absolute_error(Y[test], preds_linreg)
        mae_ridge = mean_absolute_error(Y[test], preds_ridge)
        mae_lasso = mean_absolute_error(Y[test], preds_lasso)
        mae_elastic = mean_absolute_error(Y[test], preds_elastic)
        mae_rfr = mean_absolute_error(Y[test], preds_rfr)
        mae_xgbreg = mean_absolute_error(Y[test], preds_xgbreg)

        # Explained variance
        evs_linreg = explained_variance_score(Y[test], preds_linreg)
        evs_ridge = explained_variance_score(Y[test], preds_ridge)
        evs_lasso = explained_variance_score(Y[test], preds_lasso)
        evs_elastic = explained_variance_score(Y[test], preds_elastic)
        evs_rfr = explained_variance_score(Y[test], preds_rfr)
        evs_xgbreg = explained_variance_score(Y[test], preds_xgbreg)

        # Add to the dict_results
        # MSE
        dict_results['linreg']['mse'].append(mse_linreg)
        dict_results['ridge']['mse'].append(mse_ridge)
        dict_results['lasso']['mse'].append(mse_lasso)
        dict_results['elastic']['mse'].append(mse_elastic)
        dict_results['rfr']['mse'].append(mse_rfr)
        dict_results['xgbreg']['mse'].append(mse_xgbreg)

        # MAE
        dict_results['linreg']['mae'].append(mae_linreg)
        dict_results['ridge']['mae'].append(mae_ridge)
        dict_results['lasso']['mae'].append(mae_lasso)
        dict_results['elastic']['mae'].append(mae_elastic)
        dict_results['rfr']['mae'].append(mae_rfr)
        dict_results['xgbreg']['mae'].append(mae_xgbreg)

        # EVS
        dict_results['linreg']['evs'].append(evs_linreg)
        dict_results['ridge']['evs'].append(evs_ridge)
        dict_results['lasso']['evs'].append(evs_lasso)
        dict_results['elastic']['evs'].append(evs_elastic)
        dict_results['rfr']['evs'].append(evs_rfr)
        dict_results['xgbreg']['evs'].append(evs_xgbreg)

    # Add mean scores to the dict and show them
    # Mean evs
    dict_results['linreg']['mean_evs'] = np.mean(dict_results['linreg']['evs'])
    dict_results['ridge']['mean_evs'] = np.mean(dict_results['ridge']['evs'])
    dict_results['lasso']['mean_evs'] = np.mean(dict_results['lasso']['evs'])
    dict_results['elastic']['mean_evs'] = np.mean(dict_results['elastic']['evs'])
    dict_results['rfr']['mean_evs'] = np.mean(dict_results['rfr']['evs'])
    dict_results['xgbreg']['mean_evs'] = np.mean(dict_results['xgbreg']['evs'])

    # Mean mse
    dict_results['linreg']['mean_mse'] = np.mean(dict_results['linreg']['mse'])
    dict_results['ridge']['mean_mse'] = np.mean(dict_results['ridge']['mse'])
    dict_results['lasso']['mean_mse'] = np.mean(dict_results['lasso']['mse'])
    dict_results['elastic']['mean_mse'] = np.mean(dict_results['elastic']['mse'])
    dict_results['rfr']['mean_mse'] = np.mean(dict_results['rfr']['mse'])
    dict_results['xgbreg']['mean_mse'] = np.mean(dict_results['xgbreg']['mse'])

    # Mean mae
    dict_results['linreg']['mean_mae'] = np.mean(dict_results['linreg']['mae'])
    dict_results['ridge']['mean_mae'] = np.mean(dict_results['ridge']['mae'])
    dict_results['lasso']['mean_mae'] = np.mean(dict_results['lasso']['mae'])
    dict_results['elastic']['mean_mae'] = np.mean(dict_results['elastic']['mae'])
    dict_results['rfr']['mean_mae'] = np.mean(dict_results['rfr']['mae'])
    dict_results['xgbreg']['mean_mae'] = np.mean(dict_results['xgbreg']['mae'])

    # Mean duration
    mean_dura_linreg = np.mean(dict_results['linreg']['duration'])
    mean_dura_ridge = np.mean(dict_results['ridge']['duration'])
    mean_dura_lasso = np.mean(dict_results['lasso']['duration'])
    mean_dura_elastic = np.mean(dict_results['elastic']['duration'])
    mean_dura_rfr = np.mean(dict_results['rfr']['duration'])
    mean_dura_xgbreg = np.mean(dict_results['xgbreg']['duration'])
    dict_results['linreg']['mean_duration'] = mean_dura_linreg
    dict_results['ridge']['mean_duration'] = mean_dura_ridge
    dict_results['lasso']['mean_duration'] = mean_dura_lasso
    dict_results['elastic']['mean_duration'] = mean_dura_elastic
    dict_results['rfr']['mean_duration'] = mean_dura_rfr
    dict_results['xgbreg']['mean_duration'] = mean_dura_xgbreg

    # Shows mean results
    for regr in regr_names:
        for sc in ['mean_mse', 'mean_mae', 'mean_evs', 'mean_duration']:
            print(f"{regr:<15}: {sc:<18} = {dict_results[regr][sc]:>10.3f}")

    return dict_results


def TuningParamModelPredictionScore(X, Y):
    """
    Do a gridsearchcv for the model XGBoost that predicts the score of an
    article.

    Parameters
    ----------
    X: numpy array
        Concatenation of vectors article.

    Y: numpy array
        Column vector with the scores of the articles.

    Returns
    -------
    dict
        the best parameters found by the gridsearchcv.

    """
    xgbreg = XGBRegressor()
    parameters = {'eta': [0.2, 0.3, 0.4],
                  'max_depth': [6, 7],}
    clf = GridSearchCV(xgbreg, parameters, verbose=4,
                       scoring='neg_mean_absolute_error')
    clf.fit(X, Y)
    print(f"The best parameters: {clf.best_params_}")
    print(f"The best scores: {clf.best_score_}")
    return clf.best_params_
