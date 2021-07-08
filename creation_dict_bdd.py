"""
@author: baptistehessel

Creates the main dictionary with all the information about the pages of a
client.
The dict_pages is of the form:
    {id_page: dict_page, ...}.
The dict_page includes several dictionary or list:
    dict_page = {'arts': ..., 'cartons': ..., ...}.

The name of the object created is dict_pages. The absolute path is:
    dir_out/dict_pages.

"""
from bs4 import BeautifulSoup as bs
from pathlib import Path
import numpy as np
import pickle
import time
import re


class MyException(Exception):
    pass


def FillDictPage(content_xml, infos_page, file_name):
    dict_page = {}
    try:
        dict_page['pageName'] = str(file_name)
    except Exception as e:
        print(e, 'FillDictPage : pageName')
    try:
        dict_page['melodyId'] = float(infos_page.get('melodyid'))
    except Exception as e:
        print(e, 'FillDictPage : melodyId')
    try:
        dict_page['cahierCode'] = infos_page.get('cahiercode')
    except Exception as e:
        print(e, 'FillDictPage : cahierCode')
    try:
        dict_page['customerCode'] = infos_page.get('customercode')
    except Exception as e:
        print(e, 'FillDictPage : customerCode')
    try:
        dict_page['pressTitleCode'] = infos_page.get('presstitlecode')
    except Exception as e:
        print(e, 'FillDictPage : pressTitleCode')
    try:
        dict_page['editionCode'] = infos_page.get('editioncode')
    except Exception as e:
        print(e, 'FillDictPage : editionCode')
    try:
        dict_page['folio'] = int(infos_page.get('folio'))
    except Exception as e:
        print(e, 'FillDictPage : folio')
    try:
        dict_page['width'] = float(infos_page.get('docwidth'))
    except Exception as e:
        print(e, 'FillDictPage : width')
    try:
        dict_page['height'] = float(infos_page.get('docheight'))
    except Exception as e:
        print(e, 'FillDictPage : height')
    try:
        dict_page['paddingTop'] = float(infos_page.get('paddingtop'))
    except Exception as e:
        print(e, 'FillDictPage : paddingTop')
    try:
        dict_page['paddingLeft'] = float(infos_page.get('paddingleft'))
    except Exception as e:
        print(e, 'FillDictPage : paddingLeft')
    try:
        dict_page['catName'] = infos_page.get('printcategoryname')
    except Exception as e:
        print(e, 'FillDictPage : catName')
    try:
        dict_page['nbArt'] = len(content_xml.find_all('article'))
    except Exception as e:
        print(e, 'FillDictPage : nbArt')
    try:
        pagetemp = content_xml.find('pagetemplate')
        if pagetemp is not None:
            dict_page['pageTemplate'] = pagetemp.get('melodyid')
            dict_page['nameTemplate'] = pagetemp.find('name').text
        else:
            dict_page['pageTemplate'] = "NoTemplate"
            dict_page['nameTemplate'] = "NoTemplate"
    except Exception as e:
        print(e, 'FillDictPage : pageTemplate and nameTemplate')
    return dict_page


def FillDictCarton(carton_soup):
    dict_carton = {}
    dict_carton['x'] = round(float(carton_soup.get('x')), 1)
    dict_carton['y'] = round(float(carton_soup.get('y')), 1)
    dict_carton['height'] = round(float(carton_soup.get('mmheight')), 1)
    dict_carton['width'] = round(float(carton_soup.get('mmwidth')), 1)
    dict_carton['nbCol'] = int(carton_soup.get('nbcol'))
    dict_carton['nbPhoto'] = int(carton_soup.get('nbphotos'))
    return dict_carton


def GetSizeDictArticle(dict_article_curr):
    if dict_article_curr['nbSignTxt'] < 500:
        dict_article_curr['taille'] = 0
    elif 500 <= dict_article_curr['nbSignTxt'] < 2000:
        dict_article_curr['taille'] = 1
    elif 2000 <= dict_article_curr['nbSignTxt'] < 3000:
        dict_article_curr['taille'] = 2
    elif 3000 <= dict_article_curr['nbSignTxt'] < 4000:
        dict_article_curr['taille'] = 3
    else:
        dict_article_curr['taille'] = 4
    return dict_article_curr


def FillDictArticle(art_soup):
    dict_article_current = {}
    try:
        dict_article_current['melodyId'] = int(art_soup.get('melodyid'))
    except Exception as e:
        print(e, 'FillDictArticle : melodyId')
    try:
        if len(art_soup.get('nbcols')) > 0:
            dict_article_current['nbCol'] = int(art_soup.get('nbcols'))
        else:
            dict_article_current['nbCol'] = 3
    except:
        print('FillDictArticle : nbCol')
    try:
        if art_soup.get('x') != '':
            dict_article_current['x'] = round(float(art_soup.get('x')), 1)
            dict_article_current['y'] = round(float(art_soup.get('y')), 1)
        else:
            dict_article_current['x'] = - 1
            dict_article_current['y'] = -1
    except Exception as e:
        print(e, 'FillDictArticle : x and y')

    try:
        if len(art_soup.get('mmwidth')) > 0:
            mmWidth = float(art_soup.get('mmwidth'))
            mmHeight = float(art_soup.get('mmheight'))
            dict_article_current['width'] = round(mmWidth, 1)
            dict_article_current['height'] = round(mmHeight, 1)
        else:
            dict_article_current['width'] = -1
            dict_article_current['height'] = -1
    except Exception as e:
        print(e, 'FillDictArticle : width and height')
    # Pb sur cet export (14/12), le nombre de signe art = 0
    try:
        dict_article_current['nbSign'] = int(art_soup.get('nbsignes'))
    except Exception as e:
        print(e, 'FillDictArticle : nbSign')
    try:
        synopsis = art_soup.find('field', {'type':'synopsis'})
        dict_article_current['syn'] = int(synopsis.get('nbsignes'))
    except Exception as e:
        print(e, 'FillDictArticle : syn')
    try:
        suptit = art_soup.find('field', {'type':'supTitle'})
        dict_article_current['supTitle'] = int(suptit.get('nbsignes'))
    except Exception as e:
        print(e, 'FillDictArticle : supTitle')
    try:
        tit = art_soup.find('field', {'type':'title'})
        dict_article_current['title'] = int(tit.get('nbsignes'))
    except Exception as e:
        print(e, 'FillDictArticle : title')
    try:
        sect = art_soup.find('field', {'type':'secondaryTitle'})
        dict_article_current['secTitle'] = int(sect.get('nbsignes'))
    except Exception as e:
        print(e, 'FillDictArticle : secTitle')
    try:
        subt = art_soup.find('field', {'type':'subTitle'})
        dict_article_current['subTitle'] = int(subt.get('nbsignes'))
    except Exception as e:
        print(e, 'FillDictArticle : subTitle')
    try:
        abst = art_soup.find('field', {'type':'abstract'})
        dict_article_current['abstract'] = int(abst.get('nbsignes'))
    except Exception as e:
        print(e, 'FillDictArticle : abstract')
    try:
        txt = art_soup.find('field', {'type':'text'})
        dict_article_current['nbSignTxt'] = int(txt.get('nbsignes'))
    except Exception as e:
        print(e, 'FillDictArticle : nbSignTxt')
    # Article content
    try:
        if art_soup.find('text') is not None:
            dict_article_current['content'] = art_soup.find('text').text
        else:
            dict_article_current['content'] = "I don't have much to say"
    except Exception as e:
        print(e, 'FillDictArticle : content', art_soup.find('text'))
    # Title content
    try:
        dict_article_current['titleContent'] = art_soup.find('title').text
    except Exception as e:
        print(e, 'FillDictArticle : titleContent')
    try:
        dict_article_current['nbBlock'] = len(art_soup.find_all('block'))
    except Exception as e:
        print(e, 'FillDictArticle : nbBlock')
    # Type of the blocks
    try:
        l_blocks = []
        l_type_blocks = []
        for block in art_soup.find_all('block'):
            l_blocks.append(int(block.get('nbsignes')))
            match = re.search('(type)="(\w+)"', str(block))
            if match:
                l_type_blocks.append(match.group(2))
        dict_article_current['nbSignBlock'] = l_blocks
        dict_article_current['typeBlock'] = l_type_blocks
    except Exception as e:
        print(e, 'FillDictArticle : nbSignBlock and typeBlock')
    try:
        auth = art_soup.find('field', {'type': 'author'})
        dict_article_current['author'] = int(auth.get('nbsignes'))
    except Exception as e:
        print(e, 'FillDictArticle : author')
    try:
        commu = art_soup.find('field', {'type': 'commune'})
        dict_article_current['commune'] = int(commu.get('nbsignes'))
    except Exception as e:
        print(e, 'FillDictArticle : commune')
    try:
        exerg = art_soup.find('field', {'type': 'exergue'})
        dict_article_current['exergue'] = int(exerg.get('nbsignes'))
    except Exception as e:
        print(e, 'FillDictArticle : exergue')
    try:
        note_sp = art_soup.find('field', {'type': 'note'})
        dict_article_current['note'] = int(note_sp.get('nbsignes'))
    except Exception as e:
        print(e, 'FillDictArticle : note')
    try:
        dict_article_current['nbPhoto'] = len(art_soup.find_all('photo'))
    except Exception as e:
        print(e, 'FillDictArticle : nbPhoto')
    # Détermination de si l'article est en haut ou non
    try:
        if float(dict_article_current['y']) < 90:
            dict_article_current['posArt'] = 1
        else:
            dict_article_current['posArt'] = 0
    except Exception as e:
        print(e, 'FillDictArticle : posArt')
    # Initialisation de la hiérarchie des articles
    try:
        dict_article_current['isPrinc'] = 0
        dict_article_current['isSec'] = 0
        dict_article_current['isMinor'] = 0
        dict_article_current['isSub'] = 0
        dict_article_current['isTer'] = 0
    except Exception as e:
        print(e, 'FillDictArticle : isPrinc...')
    return dict_article_current


def FillDictPhoto(bl_ph_soup):
    dict_photo = {}
    try:
        dict_photo['x'] = round(float(bl_ph_soup.get('x')), 1)
        dict_photo['y'] = round(float(bl_ph_soup.get('y')), 1)
    except Exception:
        dict_photo['x'] = -1
        dict_photo['y'] = -1
    try:
        dict_photo['xAbs'] = round(float(bl_ph_soup.get('absolute_x')), 1)
        dict_photo['yAbs'] = round(float(bl_ph_soup.get('absolute_y')), 1)
    except Exception:
        dict_photo['xAbs'] = -1
        dict_photo['yAbs'] = -1
    try:
        if float(bl_ph_soup.get('width')) > 0:
            dict_photo['width'] = round(float(bl_ph_soup.get('width')), 1)
        else:
            dict_photo['width'] = 80
        if float(bl_ph_soup.get('height')) > 0:
            dict_photo['height'] = round(float(bl_ph_soup.get('height')), 1)
        else:
            dict_photo['height'] = 60
    except Exception:
        dict_photo['width'] = 80
        dict_photo['height'] = 60
    try:
        dict_photo['credit'] = int(bl_ph_soup.find('credit').get('nbsignes'))
        dict_photo['legende'] = int(bl_ph_soup.find('legend').get('nbsignes'))
    except Exception:
        dict_photo['credit'] = -1
        dict_photo['legende'] = -1
    return dict_photo


def ExtrationData(rep_data):
    """

    Extraction des données des xml.

    """
    info_extract_pages = []
    # Boucle pour l'extraction des xml
    for i, p in enumerate(Path(rep_data).glob('./**/*')):
        if p.suffix == '.xml':
            with open(str(p), "r", encoding='utf-8') as file:
                content = file.readlines()
                content = "".join(content)
                soup = bs(content, "lxml")
                pg = soup.find('page')
                # Remplissage dico_page
                dict_page = FillDictPage(soup, pg, p)
                dict_page['cartons'] = []
                try:
                    for carton_soup in soup.find_all('carton'):
                        dict_carton = FillDictCarton(carton_soup)
                        dict_page['cartons'].append(dict_carton)
                except Exception as e:
                    print(e, 'l.265 FillDictCarton', p)
                # Remplissage des dict ARTICLE
                dict_page['articles'] = {}
                for art_soup in soup.find_all('article'):
                    try:
                        dict_art = FillDictArticle(art_soup)
                        l_dict_photos = []
                        for bl_ph in art_soup.find_all('photo'):
                            try:
                                dict_photo = FillDictPhoto(bl_ph)
                                l_dict_photos.append(dict_photo)
                            except Exception as e:
                                print(e, 'l.282', p)
                        dict_art['photo'] = l_dict_photos
                        dict_page['articles'][dict_art['melodyId']] = dict_art
                    except Exception as e:
                        print(e, 'l.288', p)
            info_extract_pages.append(dict_page)
            if i % 500 == 0:
                print(f"500 files extracted")
    return info_extract_pages


def TestInt(v, ref, lg, ht):
    if ref[0] <= v[0] <= ref[0] + lg and ref[1] <= v[1] <= ref[1] + ht:
            return True
    return False


def DetermPrinc(dict_page):
    """
    Donne le score d'être l'art princ pour chaque article de la page.

    """
    probas = []
    val_dict = dict_page['articles'].values()
    item_dict = dict_page['articles'].items()
    print("{:-^80}".format("The values of the dictionary dict_page"))
    for val in val_dict:
        print(val)
    for id_art, dict_art in item_dict:
        p = 0
        # Si nb_col max p += 0.2
        nb_col = [(dict_art['nbCol'], ida) for ida, dict_art in item_dict]
        max_col = max((col for col, _ in nb_col))
        ind_max_col = [ida for ida, col in nb_col if col == max_col]
        if id_art in ind_max_col:
            p += 0.2
        # Nb de caractères maximum
        nb_car = [(dict_art['nbSign'], ida) for ida, dict_art in item_dict]
        max_nb_car = max((nbcar for nbcar, _ in nb_car))
        ids_max_car = [ida for ida, nbcar in nb_car if nbcar == max_nb_car]
        if id_art in ids_max_car:
            p += 0.10
        # Largeur de l'article
        larg = [(dict_art['width'], ida) for ida, dict_art in item_dict]
        max_larg = max((w for w, _ in larg))
        ids_max_larg = [ida for width, ida in larg if width == max_larg]
        if id_art in ids_max_larg:
            p += 0.15
        # Diminution score s'il existe un article avec plus grande largeur et
        # hauteur
        dim = {ida: (dic['width'], dic['height']) for ida, dic in item_dict}
        dim_art = dim[id_art]
        ids_sup_dim = [ida for ida, dim in dim.items()
                       if dim[0] >= dim_art[0] and dim[1] >= dim_art[1]]
        if ids_sup_dim != []:
            p -= 0.2
        # Augmentation de la proba si article le plus haut
        htr = [(dict_art['y'], ida) for ida, dict_art in item_dict]
        max_htr = max((y for y, _ in htr))
        ids_max_htr = [ida for y, ida in htr if y == max_htr]
        if id_art in ids_max_htr:
            p += 0.15
        # Augmentation de la proba si article avec la plus grande aire
        aires = [(d['width']*d['height'], ida) for ida, d in item_dict]
        max_aires = max((aire for aire, _ in aires))
        ids_max_aires = [ida for area, ida in aires if area == max_aires]
        if id_art in ids_max_aires:
            p += 0.15
        # Détermination si l'article possède un/des article(s) lié(s)
        coord_art = (dict_art['x'], dict_art['y'])
        largeur = dict_art['width']
        hauteur = dict_art['height']
        coord = [(dict_art['x'], dict_art['y']) for dict_art in val_dict]
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
            dict_art['has_incl'] = len(art_bound)
        # Nb images
        if dict_art['nbPhoto'] == 0:
            p -= 0.1
        elif dict_art['nbPhoto'] > 0:
            p += 0.1
        # Synopsis
        if dict_art['syn'] > 0:
            p += 0.05
        # Title
        if dict_art['title'] == 0:
            p -= 0.2
        # SupTitle
        if dict_art['supTitle'] > 0:
            p += 0.05
        # SecondaryTitle
        if dict_art['secTitle'] > 0:
            p += 0.05
        # SubTitle
        if dict_art['subTitle'] > 0:
            p += 0.05
        # Abstract
        if dict_art['abstract'] > 0:
            p += 0.05
        # Exergue
        if dict_art['exergue'] > 0:
            p += 0.05
        # PetitTitre et InterTitre
        if 'intertitre' in dict_art['typeBlock']:
            p += 0.05
        if 'petittitre' in dict_art['typeBlock']:
            p += 0.05
        # Position_Article
        if dict_art['posArt'] > 0:
            p += 0.2
        probas.append((round(p, 2), id_art))
        dict_art['score'] = p
    # En cas d'égalité
    pmax = max(probas)[0]
    ids_max = [ida for proba, ida in probas if proba == pmax]
    if len(ids_max) > 1:
        # Il faut prendre l'article le plus haut
        htrs = [(x['y'], ida) for ida, x in item_dict if ida in ids_max]
        max_htrs = max(htrs)
        ids_max_htr = [ida for htr, ida in htrs if htr == max_htrs]
        if len(ids_max_htr) > 1:
            # Cas où 2 articles même hauteur même score -> le plus large
            lg = [(x['x'], ida) for ida, x in item_dict if ida in ids_max_htr]
            id_art_p = max(lg, key=lambda x: x[0])[1]
        else:
            id_art_p = ids_max_htr[0]
    else:
        id_art_p =  ids_max[0]
    # Mise à jour de l'attribut isPrinc
    dict_page['articles'][id_art_p]['isPrinc'] = 1
    return dict_page


def DetermAire(dict_page):
    """
    Mise à jour de l'élément art.aire dans {1, 2, 3}.
    """
    for dict_art in dict_page['articles'].values():
        aire_art = dict_art['width'] * dict_art['height']
        if aire_art < 19600:
            dict_art['aire'] = 1
        elif 19600 <= aire_art < 45000:
            dict_art['aire'] = 2
        else:
            dict_art['aire'] = 3
    return dict_page


def DetermAireImg(dict_page):
    """
    Mise à jour de l'élément aire photo.
    """
    for dict_art in dict_page['articles'].values():
        for dict_img in dict_art['photo']:
            aire_img = dict_img['width'] * dict_img['height']
            if aire_img < 7000:
                dict_img['aire_img'] = 1
            elif 7000 <= aire_img < 15000:
                dict_img['aire_img'] = 2
            else:
                dict_img['aire_img'] = 3
    return dict_page


def Update(dict_page):
    """
    Mise à jour de distToPrinc, isMinor/sub/sec, pos et aire.

    """
    # score et isPrinc

    dict_page = DetermPrinc(dict_page)
    #except Exception as e:
        #str_exc = (f"An error with Update:DetermPrinc: {e}\n"
                   #f"dict_page: {dict_page}")
        #raise MyException(str_exc)

    # distToPrinc
    try:
        dict_page, _ = DistanceCosTuple(dict_page)
    except Exception as e:
        str_exc = (f"An error with Update:DistanceCosTuple: {e}\n"
                   f"dict_page: {dict_page}")
        raise MyException(str_exc)

    # isMinor/sub/sec
    try:
        dict_page = Determ_Nat_Art(dict_page)
    except Exception as e:
        str_exc = (f"An error with Update:Determ_Nat_Art: {e}\n"
                   f"dict_page: {dict_page}")
        raise MyException(str_exc)

    # art['aire']
    try:
        dict_page = DetermAire(dict_page)
    except Exception as e:
        str_exc = (f"An error with Update:DetermAire: {e}\n"
                   f"dict_page: {dict_page}")
        raise MyException(str_exc)

    # photo['aire_img']
    try:
        dict_page = DetermAireImg(dict_page)
    except Exception as e:
        str_exc = (f"An error with Update:DetermAireImg: {e}\n"
                   f"dict_page: {dict_page}")
        raise MyException(str_exc)

    # art['aireImg']
    try:
        dict_page = UpdateAllAireImg(dict_page)
    except Exception as e:
        str_exc = (f"An error with Update:UpdateAllAireImg: {e}\n"
                   f"dict_page: {dict_page}")
        raise MyException(str_exc)

    # art['petittitre'], art['intertitre'], art['quest_rep']
    try:
        for dict_art in dict_page['articles'].values():
            if 'intertitre' in dict_art['typeBlock']:
                dict_art['intertitre'] = 1
            else:
                dict_art['intertitre'] = 0
            if 'petittitre' in dict_art['typeBlock']:
                dict_art['petittitre'] = 1
            else:
                dict_art['petittitre'] = 0
            if 'question' in dict_art['typeBlock']:
                dict_art['quest_rep'] = 1
            elif 'reponse' in dict_art['typeBlock']:
                dict_art['quest_rep'] = 1
            else:
                dict_art['quest_rep'] = 0
    except Exception as e:
        str_exc = (f"An error with the last 3 features: {e}\n"
                   f"The dict_page['articles']: {dict_page['articles']}")
        raise MyException(str_exc)
    return dict_page


def UpdateAll(dict_pages):
    """
    Update toutes les pages.
    """
    n = len(dict_pages)
    for i, dict_page in enumerate(dict_pages):
        dict_page = Update(dict_page)
        if i % (n//50) == 0:
            print(f"UpdateAll: {100*i/n:.2f}%")
    return dict_pages


def Determ_Nat_Art(dict_page):
    """
    À utiliser une fois que les élmts score, distToPrinc sont remplis.
    """
    for dict_art in dict_page['articles'].values():
        if dict_art['isPrinc'] == 0:
            if dict_art['score'] <= 0.05:
                # Si img : ter, sinon : minor
                if dict_art['nbPhoto'] > 0:
                    dict_art['isTer'] = 1
                else:
                    dict_art['isMinor'] = 1
            else:
                if dict_art['distToPrinc'] >= 0.2:
                    dict_art['isSub'] = 1
                else:
                    dict_art['isSec'] = 1
    return dict_page


def DicoCorpus(corpus):
    d = {}
    for text in corpus:
        for word in text.split():
            d[word] = d.get(word, 0) + 1
    return d


def DicoDoc(liste_mots):
    return {mot: liste_mots.count(mot) for mot in liste_mots}


def ConvTxt(l_mots, txt_lem):
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


def DistanceCosTuple(dict_page):
    """
    Calcul les distances entre le txt_lem et le sous-cp obtenu
    Mise à jour de distToPrinc
    """
    # Trouver l'article principal
    text_ref = ''
    val_dict = dict_page['articles'].values()
    for dict_article in val_dict:
        if dict_article['isPrinc'] == 1:
            text_ref = dict_article['content']
            break
    corpus = [dict_article['content'] for dict_article in val_dict]
    dico = DicoCorpus(corpus)
    l_mots = [clef for clef in dico.keys()]
    # Il faut que le dernier élément de corpus soit le text_ref
    Vect = [ConvTxt(l_mots, text) for text in corpus]
    Vect.append(ConvTxt(l_mots, text_ref))
    l_dist = []
    N = len(Vect)
    norme1 = np.sqrt(np.dot(Vect[N - 1], Vect[N - 1]))
    i = 0
    for text, dict_art in zip(corpus, val_dict):
        # Le texte live est rajouté à la fin, d'où le N - 1.
        prod_scalaire = np.dot(Vect[N - 1], Vect[i])
        norme2 = np.sqrt(np.dot(Vect[i], Vect[i]))
        prod_normes = norme1 * norme2 if norme1 * norme2 != 0 else 1
        i += 1
        res = round(prod_scalaire / prod_normes, 2)
        l_dist.append([res, text[:60]])
        # Mise à jour de l'élément distToPrinc
        dict_art['distToPrinc'] = res
    l_dist.sort(reverse=True)
    return dict_page, l_dist


def UpdateNew(dict_pages):
    """
    Rajout des éléments suivants : 'prop_img', 'nbcar', 'nbimg', 'prop_art'.
    """
    # Rajout de l'élément prop_art à chaque ntuple_page
    for dict_page in dict_pages:
        aire_img_tot, nbcar_tot, nbimg_tot, aire_tot = 0, 0, 0, 0
        for dict_art in dict_page['articles'].values():
            aire_tot += dict_art['width'] * dict_art['height']
            nbcar_tot += dict_art['nbSignTxt']
            nbimg_tot += dict_art['nbPhoto']
            for dict_img in dict_art['photo']:
                aire_img_tot += dict_img['width'] * dict_img['height']
        aire_page = dict_page['width'] * dict_page['height']
        # 1. prop_art et 2. prop_img
        dict_page['propArt'] = round(aire_tot / aire_page, 1)
        dict_page['propImg'] = round(aire_img_tot / aire_page, 1)
        # 3. nbcar et 4. nbimg
        dict_page['nbCar'] = nbcar_tot
        dict_page['nbImg'] = nbimg_tot
    return dict_pages


def GetDistributionPage(dict_page):
    """
    Renvoie la répartition du type d'articles dans la page.
    """
    rep = [0, 0, 0, 0]
    for dict_art in dict_page['articles'].values():
        if dict_art['isPrinc'] == 1:
            rep[0] += 1
        elif sum([dict_art['isSec'], dict_art['isSub']]) > 0:
            rep[1] += 1
        elif dict_art['isTer'] == 1:
            rep[2] += 1
        else:
            rep[3] += 1
    return rep


def filter_set(dico_bdd, search_string):
    def iterator_func(dico):
        for v in [dico['catname']]:
            if search_string in v:
                return True
        return False
    return filter(iterator_func, dico_bdd.values())


def CreationBDD(list_dict_pages):
    """
    Crée un liste contenant les listes des infos sur les pages.
    """
    dico_bdd = {}
    for dict_page in list_dict_pages:
        try:
            if len(dict_page['articles']) > 0:
                id_p = dict_page['melodyId']
                dico_bdd[id_p] = dict_page
        except Exception as e:
            print(e)
    return dico_bdd


def UpdateAireImg(dict_art):
    aire = 0
    for dict_img in dict_art['photo']:
        try:
            aire += dict_img['width'] * dict_img['height']
        except:
            pass
    dict_art['aireImg'] = aire
    return dict_art


def UpdateAllAireImg(dict_page):
    for dict_art in dict_page['articles'].values():
        dict_art = UpdateAireImg(dict_art)
    return dict_page


def ShowElementsDicoBDD(dico_bdd):
    rdm_id_page = np.random.choice(list(dico_bdd.keys()))
    print(rdm_id_page)
    for key, value in dico_bdd[rdm_id_page].items():
        print(key)
        if type(value) is list:
            if len(value) > 0:
                if type(value[0]) is dict:
                    for dict_elt in value:
                        for key, value in dict_elt.items():
                            print(key)
                            print(value)
                else:
                    for elt in value:
                        print(elt)
        else:
            print(value)


##############################################################################
#                                                                            #
#        Extraction of the data and Creation of the dict_pages               #
#                                                                            #
##############################################################################


def CreationDictPages(rep_data, dir_out, save_dict=True):
    t0 = time.time()
    list_dict_pages = ExtrationData(rep_data)
    print(f"ExtrationData duration: {time.time() - t0} sec.")

    t1 = time.time()
    list_dict_pages = UpdateAll(list_dict_pages)
    print("UpdateAll : {:.2f} sec".format(time.time() - t1))

    t2 = time.time()
    list_dict_pages = UpdateNew(list_dict_pages)
    print("UpdateNew: {:.2f} sec".format(time.time() - t2))

    t3 = time.time()
    dico_bdd = CreationBDD(list_dict_pages)
    print("CreationBDD: {:.2f} sec".format(time.time() - t3))

    ShowElementsDicoBDD(dico_bdd)

    if save_dict:
        with open(dir_out + 'dict_pages', 'wb') as f:
            pickle.dump(dico_bdd, f)
