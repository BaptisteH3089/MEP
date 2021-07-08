#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from bs4 import BeautifulSoup as bs
from operator import itemgetter
from pathlib import Path
import numpy as np
import pickle
import re


def FillDicoPage(content_xml, page_soup, file_name):
    dico_page = {}
    dico_page['pageName'] = str(file_name)
    dico_page['melodyId'] = int(page_soup.get('melodyid'))
    dico_page['cahierCode'] = page_soup.get('cahiercode')
    dico_page['customerCode'] = page_soup.get('customercode')
    dico_page['pressTitleCode'] = page_soup.get('presstitlecode')
    dico_page['editionCode'] = page_soup.get('editioncode')
    dico_page['folio'] = int(page_soup.get('folio'))
    dico_page['width'] = int(page_soup.get('docwidth'))
    dico_page['height'] = int(page_soup.get('docheight'))
    dico_page['paddingTop'] = int(page_soup.get('paddingtop'))
    dico_page['paddingLeft'] = int(page_soup.get('paddingleft'))
    dico_page['catName'] = page_soup.get('printcategoryname')
    dico_page['nbArt'] = len(content_xml.find_all('article'))
    pagetemp = content_xml.find('pagetemplate')
    if pagetemp is not None:
        dico_page['pageTemplate'] = int(pagetemp.get('melodyid'))
        dico_page['nameTemplate'] = pagetemp.find('name').text
    return dico_page


def FillDicoCarton(carton_soup):
    dico_carton = {}
    dico_carton['x'] = int(carton_soup.get('x'))
    dico_carton['y'] = int(carton_soup.get('y'))
    dico_carton['height'] = int(carton_soup.get('mmheight'))
    dico_carton['width'] = int(carton_soup.get('mmwidth'))
    dico_carton['nbCol'] = int(carton_soup.get('nbcol'))
    dico_carton['nbPhoto'] = int(carton_soup.get('nbphotos'))
    return dico_carton


def FillDicoArticle(art_soup):
    dico_art = {}
    dico_art['melodyId'] = int(art_soup.get('melodyid'))
    dico_art['nbCol'] = int(art_soup.get('nbcols'))
    if art_soup.get('x') != '':
        dico_art['x'] = int(art_soup.get('x'))
        dico_art['y'] = int(art_soup.get('y'))
    else:
        dico_art['x'] = - 1
        dico_art['y'] = -1
    if len(art_soup.get('mmwidth')) > 0:
        dico_art['width'] = int(art_soup.get('mmwidth'))
        dico_art['height'] = int(art_soup.get('mmheight'))
    else:
        dico_art['width'] = -1
        dico_art['height'] = -1
    dico_art['nbSign'] = int(art_soup.get('nbsignes'))
    synopsis = art_soup.find('field', {'type':'synopsis'})
    dico_art['syn'] = int(synopsis.get('nbsignes'))
    suptit = art_soup.find('field', {'type':'supTitle'})
    dico_art['supTitle'] = int(suptit.get('nbsignes'))
    tit = art_soup.find('field', {'type':'title'})
    dico_art['title'] = int(tit.get('nbsignes'))
    sect = art_soup.find('field', {'type':'secondaryTitle'})
    dico_art['secTitle'] = int(sect.get('nbsignes'))
    subt = art_soup.find('field', {'type':'subTitle'})
    dico_art['subTitle'] = int(subt.get('nbsignes'))
    abst = art_soup.find('field', {'type':'abstract'})
    dico_art['abstract'] = int(abst.get('nbsignes'))
    txt = art_soup.find('field', {'type':'text'})
    dico_art['nbSignTxt'] = int(txt.get('nbsignes'))
    # Article content
    if art_soup.find('text') is not None:
        dico_art['content'] = art_soup.find('text').text
    else:
        dico_art['content'] = "I don't have much to say"
    # Title content
    dico_art['titleContent'] = art_soup.find('title').text
    # Number of blocks
    dico_art['nbBlock'] = len(art_soup.find_all('block'))
    # Type of the blocks
    l_blocks = []
    l_type_blocks = []
    for block in art_soup.find_all('block'):
        l_blocks.append(int(block.get('nbsignes')))
        match = re.search('(type)="(\w+)"', str(block))
        if match:
            l_type_blocks.append(match.group(2))
    dico_art['nbSignBlock'] = l_blocks
    dico_art['typeBlock'] = l_type_blocks
    auth = art_soup.find('field', {'type': 'author'})
    dico_art['author'] = int(auth.get('nbsignes'))
    commu = art_soup.find('field', {'type': 'commune'})
    dico_art['commune'] = int(commu.get('nbsignes'))
    exerg = art_soup.find('field', {'type': 'exergue'})
    dico_art['exergue'] = int(exerg.get('nbsignes'))
    note_sp = art_soup.find('field', {'type': 'note'})
    dico_art['note'] = int(note_sp.get('nbsignes'))
    dico_art['nbPhoto'] = len(art_soup.find_all('photo'))
    # Détermination de si l'article est en haut ou non
    dico_art['posArt'] = 1 if float(article_current.y) < 90 else 0
    # Initialisation de la hiérarchie des articles
    dico_art['isPrinc'] = 0
    dico_art['isSec'] = 0
    dico_art['isMinor'] = 0
    dico_art['isSub'] = 0
    dico_art['isTer'] = 0
    return dico_art


def FillDicoPhoto(img_soup):
    dico_img = {}
    try:
        dico_img['x'] = int(img_soup.get('x'))
        dico_img['y'] = int(img_soup.get('y'))
    except Exception as e:
        dico_img['x'] = -1
        dico_img['y'] = -1
    try:
        dico_img['xAbs'] = int(img_soup.get('absolute_x'))
        dico_img['yAbs'] = int(img_soup.get('absolute_y'))
    except Exception as e:
        dico_img['xAbs'] = -1
        dico_img['yAbs'] = -1
    if float(img_soup.get('width')) > 0:
        dico_img['width'] = int(img_soup.get('width'))
    else:
        dico_img['width'] = -1
    if float(img_soup.get('height')) > 0:
        dico_img['height'] = int(img_soup.get('height'))
    else:
        dico_img['height'] = -1
    try:
        dico_img['credit'] = int(img_soup.find('credit').get('nbsignes'))
        dico_img['legende'] = int(img_soup.find('legend').get('nbsignes'))
    except Exception as e:
        dico_img['credit'] = -1
        dico_img['legende'] = -1
    return dico_img


def ExtrationData(rep_data):
    """
    Extraction des données des xml.
    """
    info_extract_pages = []
    # Boucle pour l'extraction des xml
    for p in Path(rep_data).glob('./**/*'):
        if p.suffix == '.xml':
            with open(str(p), "r", encoding='utf-8') as file:
                content = file.readlines()
                content = "".join(content)
            soup = bs(content, "lxml")
            pg = soup.find('page')
            # Remplissage Dico page
            dico_page = FillDicoPage(ntuple_page, soup, pg, p)
            dico_page['cartons'] = []
            for carton_soup in soup.find_all('carton'):
                dico_carton = FillDicoCarton(carton_soup)
                ntuple_page['cartons'].append(dico_carton)
            # Remplissage des dicos associés aux articles
            dico_page['art'] = []
            for art_soup in soup.find_all('article'):
                dico_article = FillDicoArticle(art_soup)
                liste_dico_imgs = []
                for img_soup in art.find_all('photo'):
                    dico_img = FillDicoPhoto(img_soup)
                    liste_dico_imgs.append(dico_img)
                dico_article['photo'] = liste_dico_imgs
                dico_page['art'].append(dico_article)
        info_extract_pages.append(dico_page)
    return info_extract_pages


def DetermPrinc(dico_page):
    """
    Donne le score d'être l'art princ pour chaque article de la page.
    """
    probas = []
    for ind_art, dico_art in enumerate(dico_page['art']):
        p = 0
        # Si nb_col max p += 0.2
        nb_col = [dico_art['nbCol'] for dico_art in dico_page['art']]
        ind_max_col = [i for i, col in enumerate(nb_col) if col == max(nb_col)]
        if ind_art in ind_max_col:
            p += 0.2
        # Nb de caractères maximum
        nb_car = [dico_art['nbSign'] for dico_art in dico_page['art']]
        ind_max_car = [i for i, col in enumerate(nb_car) if col == max(nb_car)]
        if ind_art in ind_max_car:
            p += 0.10
        # Largeur de l'article
        larg = [dico_art['width'] for dico_art in dico_page['art']]
        ind_max_larg = [i for i, lg in enumerate(larg) if lg == max(larg)]
        if ind_art in ind_max_larg:
            p += 0.15
        # Diminution score s'il existe un art avec plus grande larg et htr
        dim = [(d_art['width'], d_art['height']) for d_art in dico_page['art']]
        dim_art = dim[ind_art]
        ind_sup_dim = [i for i, dim in enumerate(dim)
                       if dim[0] >= dim_art[0] and dim[1] >= dim_art[1]]
        if ind_sup_dim != []:
            p -= 0.2
        # Augmentation de la proba si article le plus haut
        htr = [tuple_art.y for tuple_art in ntuple_page.art]
        ind_max_htr = [i for i, ht in enumerate(htr) if ht == max(htr)]
        if ind_art in ind_max_htr:
            p += 0.15
        # Augmentation de la proba si article avec la plus grande aire
        aires = [d_art['width']*d_art['height'] for d_art in dico_page['art']]
        ind_max_aires = [i for i, area in enumerate(aires)
                         if area == max(aires)]
        if ind_art in ind_max_aires:
            p += 0.15
        # Détermination si l'article possède un/des article(s) lié(s)
        coord_art = (dico_art['x'], dico_art['y'])
        largeur, hauteur = dico_art['width'], dico_art['height']
        coord = [(d_art['x'], d_art['y']) for d_art in dico_page['art']]
        try:
            coord.remove(coord_art)
        except:
            pass
        art_bound = []
        for vect in coord:
            if TestInt(vect, coord_art, largeur, hauteur):
                art_bound.append(vect)
        if art_bound != []:
            p += 0.2
            dico_art['has_incl'] = len(art_bound)
        if dico_art['nbPhoto'] == 0: p -= 0.1
        elif dico_art['nbPhoto'] > 0: p += 0.1
        if dico_art['syn'] > 0: p += 0.05
        if dico_art['title'] == 0: p -= 0.2
        if dico_art['supTitle'] > 0: p += 0.05
        if dico_art['secTitle'] > 0: p += 0.05
        if dico_art['subTitle'] > 0: p += 0.05
        if dico_art['abstract'] > 0: p += 0.05
        if dico_art['exergue'] > 0: p += 0.05
        if 'intertitre' in dico_art['typeBlock']: p += 0.05
        if 'petittitre' in dico_art['typeBlock']: p += 0.05
        if dico_art['posArt'] > 0: p += 0.2
        probas.append((round(p, 2), (dico_art['x'], dico_art['y'])))
        dico_art['score'] = p
    # En cas d'égalité
    pmax = max(probas) if probas != [] else 10
    ind_max = [i for i, p in enumerate(probas) if p == pmax]
    if len(ind_max) > 1:
        # Il faut prendre l'article le plus haut
        htrs = [dico_art['y'] for i, dico_art in enumerate(dico_page['art'])
                if i in ind_max]
        ind_max_htr = [i for i, h in enumerate(htrs) if h == max(htrs)]
        if len(ind_max_htr) > 1:
            # Cas où 2 articles même hauteur même score -> le plus large
            larg = [dic_art['x'] for i, dic_art in enumerate(dico_page['art'])
                    if i in ind_max_htr]
            art_princ = larg.index(max(larg))
        else:
            art_princ = htrs.index(max(htrs))
    else:
        art_princ = probas.index(max(probas)) if probas != [] else None
    # Mise à jour de l'attribut isPrinc
    if art_princ is not None:
        dico_page['art'][art_princ]['isPrinc'] = 1
    return probas, art_princ


def DicoCorpus(corpus):
    d = {}
    for text in corpus:
        for word in text.split():
            d[word] = d.get(word, 0) + 1
    return d


def DicoDoc(liste_mots):
    return {mot: liste_mots.count(mot) for mot in liste_mots}


def ConvTxtIntoVect(l_mots, txt_lem):
    """
    l_mots correspond à tous les mots du corpus.
    Convertit un texte en vecteur numérique.
    """
    # La taille des vecteurs
    n = len(l_mots)
    # Initialisation des vecteurs
    Vect = np.zeros(n)
    mots_vect = set(l_mots)
    # Dico avec les mots et leur position dans la liste pour gagner du temps
    dic = {mot: i for i, mot in enumerate(l_mots)}
    # Remplissage des vecteurs avec les fréquences
    Dico = DicoDoc(txt_lem.split())
    intersect = mots_vect & set(Dico.keys())
    for mot in intersect:
        Vect[dic[mot]] = Dico[mot]
    return Vect


def DistanceCosTuple(dico_page):
    """
    Calcul les distances entre le txt_lem et le sous-cp obtenu.
    Mise à jour de l'élément :
        - distToPrinc
    Étape nécessaire pour déterminer le type des articles (mineurs, sec, ...)
    """
    # Trouver l'article principal
    text_ref = ''
    for dico_article in dico_page['art']:
        if dico_article['isPrinc'] == 1:
            text_ref = dico_article['content']
            break
    corpus = [dico_article['content'] for dico_article in dico_page['art']]
    dico_mots_cp = DicoCorpus(corpus)
    l_mots = [clef for clef in dico_mots_cp.keys()]
    # Il faut que le dernier élément de corpus soit le text_ref
    Vect = [ConvTxt(l_mots, text) for text in corpus]
    Vect.append(ConvTxt(l_mots, text_ref))
    l_dist = []
    N = len(Vect)
    norme1 = np.sqrt(np.dot(Vect[N - 1], Vect[N - 1]))
    for i, (text, dico_art) in enumerate(zip(corpus, dico_page['art'])):
        # Le texte live est rajouté à la fin, d'où le N - 1.
        prod_scalaire = np.dot(Vect[N - 1], Vect[i])
        norme2 = np.sqrt(np.dot(Vect[i], Vect[i]))
        prod_normes = norme1 * norme2 if norme1 * norme2 != 0 else 1
        res = round(prod_scalaire / prod_normes, 2)
        l_dist.append([res, text[:60]])
        # Mise à jour de l'élément distToPrinc
        dico_art['distToPrinc'] = res
    l_dist.sort(reverse=True)
    return l_dist


def DetermineTypeArt(dico_page):
    """
    À utiliser une fois que les élmts score, distToPrinc sont remplis.
    Mise à jour des éléments :
        - isMinor
        - isTer
        - isSec
        - isSub
    """
    for dico_art in dico_page['art']:
        if dico_art['isPrinc'] == 0:
            if dico_art['score'] <= 0.05:
                # Si img : ter, sinon : minor
                if dico_art['nbPhoto'] > 0:
                    dico_art['isTer'] = 1
                else:
                    dico_art['isMinor'] = 1
            else:
                if dico_art['distToPrinc'] >= 0.2:
                    dico_art['isSub'] = 1
                else:
                    dico_art['isSec'] = 1
    return dico_page


def GetPositionArticle(dic_art):
    return [dic_art['x'], dic_art['y'], dic_art['width'], dic_art['height']]


def GetY(dico_page):
    big_y = np.zeros(1)
    for dico_art in dico_page['art']:
        if big_y.any() == 0:
            big_y = np.array([GetPositionArticle(dico_art)])
        else:
            y = np.array([GetPositionArticle(dico_art)])
            big_y = np.concatenate((big_y, y), axis=0)
    return big_y


def SelectionModelesPages(liste_mdp_bdd, nb_art, nb_min_pages):
    """
    Sélectionne parmi tous les MDP dans la BDD, ceux qui ont "nb_art" articles
    et qui sont utilisés pour au moins "nb_min_pages" pages.
    Si pas de correspondance trouvée, la fonction soulève une erreur.
    """
    f = lambda x: (x[1].shape[0] == nb_art) & (x[0] >= nb_min_pages)
    liste_mdp_found = list(filter(f, liste_mdp_bdd))
    if liste_mdp_found == []:
        stc1 = 'No MDP in the databse with {} articles '.format(nb_art)
        stc2 = 'and {} pages using that MDP'.format(nb_min_pages)
        raise MyException(stc1 + stc2)
    return liste_mdp_found


def TrouveMDPSim(liste_tuple_mdp, mdp_input):
    """
    Identification MDP similaire dans la liste l_model_new
    liste_tuple_mdp : l_model_new avec la sélection sur nb d'articles
    Renvoie la liste de tuples [(nbpages, mdp, ids), ...] triée
    """
    mdp_trouve = []
    for nb_pages, mdp, liste_ids in liste_tuple_mdp:
        if np.allclose(mdp, mdp_input, atol=30):
            mdp_trouve.append((nb_pages, mdp, liste_ids))
    if len(mdp_trouve) == 0:
        sentence = "No correspondance found for that MDP:\n"
        raise MyException(sentence + "{}\n in the BDD.".format(mdp_input))
    mdp_trouve.sort(key=itemgetter(0), reverse=True)
    return mdp_trouve


def AddPageBDD(dico_bdd, list_mdp_data, file_in, save_bdd, path_save_bdd):
    """
    Extraction des données du xml qui contient les infos d'une page.
    Disposition des infos extraites dans un dico.
    """
    # Extraction des données du xml
    dico_page_input = ExtrationData(file_in)
    if len(dico_page_input['art']) > 0:
        id_p = dico_page_input['melodyId']
        dico_bdd[id_p] = {}
        dico_bdd[id_p]['catname'] = dico_page_input['catName']
        # Ajout du dico page
        dico_bdd[id_p]['dico_page'] = dico_page_input
        # Il faut maintenant ajouter le MDP utilisé par cette page à la liste
        # des MDP de la BDD qui est de la forme [(nb_page, MDP, liste_ids_pg)]
        mdp_input = GetY(dico_page_input)
        # On regarde si ce MDP est dans la list_mdp_data
        nb_art = mdp_input.shape[0]
        try:
            liste_tuple_mdp = SelectionModelesPages(list_mdp_data, nb_art, 1)
            nb_pg, mdp_fd, l_ids_fd = TrouveMDPSim(liste_tuple_mdp, mdp_input)
            ind_found = list_mdp_data.index((nb_pg, mdp_fd, l_ids_fd))
            nb_pg += 1
            l_ids_fd.append(id_p)
            # Modification de la liste "list_mdp_data"
            list_mdp_data[ind_found] = (nb_pg, mdp_fd, l_ids_fd)
        except Exception as e:
            print(e)
            # Cas où on a rien trouvé. On ajoute ce mdp à la liste_tuple_mdp.
            list_mdp_data.append([1, mdp_input, [id_p]])
    if save_bdd is True:
        # CHANGER NOMS
        with open(path_save_bdd + 'dico_bdd', 'wb') as file:
            pickle.dump(dico_bdd, file)
        with open(path_save_bdd + 'l_model_new', 'wb') as file:
            pickle.dump(list_mdp_data, file)
    return "Page added"
