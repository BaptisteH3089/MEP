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
    """
    Fill a dict with all the info we have in infos_page.

    Parameters
    ----------
    content_xml: beautiful soup object
        Only used to get the number of articles.

    infos_page: beautiful soup object
        The parsed content of a xml.

    file_name: str
        The name of the file parsed.

    Returns
    -------
    dict_page: dict
        Big dictionary with infos about the pages.

    """
    dict_page = {}
    try:
        dict_page['pageName'] = str(file_name)
    except Exception as e:
        print(e, 'FillDictPage: pageName. Error.')

    try:
        dict_page['melodyId'] = float(infos_page.get('melodyid'))
    except Exception as e:
        print(e, 'FillDictPage: melodyId. Error.')

    try:
        dict_page['cahierCode'] = infos_page.get('cahiercode')
    except Exception as e:
        print(e, 'FillDictPage: cahierCode. Error.')

    try:
        dict_page['customerCode'] = infos_page.get('customercode')
    except Exception as e:
        print(e, 'FillDictPage: customerCode. Error.')

    try:
        dict_page['pressTitleCode'] = infos_page.get('presstitlecode')
    except Exception as e:
        print(e, 'FillDictPage: pressTitleCode. Error.')

    try:
        dict_page['editionCode'] = infos_page.get('editioncode')
    except Exception as e:
        print(e, 'FillDictPage: editionCode. Error.')

    try:
        dict_page['folio'] = int(infos_page.get('folio'))
    except Exception as e:
        print(e, 'FillDictPage: folio. Error.')

    try:
        dict_page['width'] = float(infos_page.get('docwidth'))
    except Exception as e:
        print(e, 'FillDictPage: width. Error.')

    try:
        dict_page['height'] = float(infos_page.get('docheight'))
    except Exception as e:
        print(e, 'FillDictPage: height. Error.')

    try:
        dict_page['paddingTop'] = float(infos_page.get('paddingtop'))
    except Exception as e:
        print(e, 'FillDictPage: paddingTop. Error.')

    try:
        dict_page['paddingLeft'] = float(infos_page.get('paddingleft'))
    except Exception as e:
        print(e, 'FillDictPage: paddingLeft. Error.')

    try:
        dict_page['catName'] = infos_page.get('printcategoryname')
    except Exception as e:
        print(e, 'FillDictPage: catName. Error.')

    try:
        dict_page['nbArt'] = len(content_xml.find_all('article'))
    except Exception as e:
        print(e, 'FillDictPage: nbArt. Error.')

    try:
        pagetemp = content_xml.find('pagetemplate')
        if pagetemp is not None:
            dict_page['pageTemplate'] = pagetemp.get('melodyid')
            dict_page['nameTemplate'] = pagetemp.find('name').text
        else:
            dict_page['pageTemplate'] = "NoTemplate"
            dict_page['nameTemplate'] = "NoTemplate"
    except Exception as e:
        print(e, 'FillDictPage: pageTemplate and nameTemplate. Error.')

    return dict_page


def FillDictCarton(carton_soup):
    """
    Fill a small dict carton or module.

    Parameters
    ----------
    carton_soup: beautiful soup object
        Parsed content of a module.

    Returns
    -------
    dict_carton: dict
        dict with keys: x, y, height, width, nbCol, nbPhoto.

    """
    dict_carton = {}
    try:
        dict_carton['x'] = round(float(carton_soup.get('x')), 1)
    except:
        dict_carton['x'] = -1
    try:
        dict_carton['y'] = round(float(carton_soup.get('y')), 1)
    except:
        dict_carton['y'] = -1

    try:
        dict_carton['height'] = round(float(carton_soup.get('mmheight')), 1)
    except:
        dict_carton['height'] = -1

    try:
        dict_carton['width'] = round(float(carton_soup.get('mmwidth')), 1)
    except:
        dict_carton['width'] = -1

    try:
        dict_carton['nbCol'] = int(carton_soup.get('nbcol'))
    except:
        dict_carton['nbCol'] = -1

    try:
        dict_carton['nbPhoto'] = int(carton_soup.get('nbphotos'))
    except:
        dict_carton['nbPhoto'] = -1

    return dict_carton


def FillDictArticle(art_soup):
    """

    We use indicators functions for the features relative to the nb of signs
    of elements, except for the content of the article.

    Parameters
    ----------
    art_soup: Beautiful soup object
        Contains the parsed content of the tag article of a xml file.

    Returns
    -------
    dict_article_current: dict
        Dict with all the info about this article.

    """
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
            dict_article_current['aireTot'] = round(mmWidth * mmHeight, 0)
        else:
            dict_article_current['width'] = -1
            dict_article_current['height'] = -1
            dict_article_current['aireTot'] = 0
    except Exception as e:
        print(e, 'FillDictArticle : width and height')

    try:
        dict_article_current['nbSign'] = int(art_soup.get('nbsignes'))
    except Exception as e:
        print(e, 'FillDictArticle : nbSign')

    try:
        synopsis = art_soup.find('field', {'type':'synopsis'})
        dict_article_current['syn_full'] = int(synopsis.get('nbsignes'))
    except Exception as e:
        print(e, 'FillDictArticle : syn_full')

    try:
        suptit = art_soup.find('field', {'type':'supTitle'})
        dict_article_current['supTitle_full'] = int(suptit.get('nbsignes'))
    except Exception as e:
        print(e, 'FillDictArticle : supTitle_full')

    try:
        tit = art_soup.find('field', {'type':'title'})
        dict_article_current['title_full'] = int(tit.get('nbsignes'))
    except Exception as e:
        print(e, 'FillDictArticle : title_full')

    try:
        sect = art_soup.find('field', {'type':'secondaryTitle'})
        dict_article_current['secTitle_full'] = int(sect.get('nbsignes'))
    except Exception as e:
        print(e, 'FillDictArticle : secTitle_full')

    try:
        subt = art_soup.find('field', {'type':'subTitle'})
        dict_article_current['subTitle_full'] = int(subt.get('nbsignes'))
    except Exception as e:
        print(e, 'FillDictArticle : subTitle_full')

    try:
        abst = art_soup.find('field', {'type':'abstract'})
        dict_article_current['abstract_full'] = int(abst.get('nbsignes'))
    except Exception as e:
        print(e, 'FillDictArticle : abstract_full')

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

    # nbBlock
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
        dict_article_current['author_full'] = int(auth.get('nbsignes'))
    except Exception as e:
        print(e, 'FillDictArticle : author_full')

    try:
        commu = art_soup.find('field', {'type': 'commune'})
        dict_article_current['commune_full'] = int(commu.get('nbsignes'))
    except Exception as e:
        print(e, 'FillDictArticle : commune_full')

    try:
        exerg = art_soup.find('field', {'type': 'exergue'})
        dict_article_current['exergue_full'] = int(exerg.get('nbsignes'))
    except Exception as e:
        print(e, 'FillDictArticle : exergue_full')

    try:
        note_sp = art_soup.find('field', {'type': 'note'})
        dict_article_current['note_full'] = int(note_sp.get('nbsignes'))
    except Exception as e:
        print(e, 'FillDictArticle : note_full')

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
    dict_article_current['isPrinc'] = 0
    dict_article_current['isSec'] = 0
    dict_article_current['isMinor'] = 0
    dict_article_current['isSub'] = 0
    dict_article_current['isTer'] = 0

    # Transformation into indicators functions
    if dict_article_current['note_full'] > 0:
        dict_article_current['note'] = 1
    else:
        dict_article_current['note'] = 0

    if dict_article_current['exergue_full'] > 0:
        dict_article_current['exergue'] = 1
    else:
        dict_article_current['exergue'] = 0

    if dict_article_current['commune_full'] > 0:
        dict_article_current['commune'] = 1
    else:
        dict_article_current['commune'] = 0

    if dict_article_current['author_full'] > 0:
        dict_article_current['author'] = 1
    else:
        dict_article_current['author'] = 0

    if dict_article_current['abstract_full'] > 0:
        dict_article_current['abstract'] = 1
    else:
        dict_article_current['abstract'] = 0

    if dict_article_current['subTitle_full'] > 0:
        dict_article_current['subTitle'] = 1
    else:
        dict_article_current['subTitle'] = 0

    if dict_article_current['secTitle_full'] > 0:
        dict_article_current['secTitle'] = 1
    else:
        dict_article_current['secTitle'] = 0

    if dict_article_current['title_full'] > 0:
        dict_article_current['title'] = 1
    else:
        dict_article_current['title'] = 0

    if dict_article_current['supTitle_full'] > 0:
        dict_article_current['supTitle'] = 1
    else:
        dict_article_current['supTitle'] = 0

    if dict_article_current['syn_full'] > 0:
        dict_article_current['syn'] = 1
    else:
        dict_article_current['syn'] = 0

    return dict_article_current


def FillDictPhoto(bl_ph_soup):
    """
    Fill a dict photo.

    Parameters
    ----------
    bl_ph_soup: beautiful soup object.
        Parsed content of the tag photos.

    Returns
    -------
    dict_photo: dict
        Dict with infos about the images of an article.

    """
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
    Extracts the data of the of the xml files

    Parameters
    ----------
    rep_data: str
        Repertory with the xml files.

    Returns
    -------
    info_extract_pages: list of dict_page
        A big list with the structured data of all the files in rep_data.

    """

    info_extract_pages = []
    # Boucle pour l'extraction des xml
    for i, p in enumerate(Path(rep_data).glob('**/*.xml')):
        # Opening of the file
        with open(str(p), "r", encoding='utf-8') as file:
            content = file.readlines()

        # Parsing of the content
        content = "".join(content)
        soup = bs(content, "lxml")
        pg = soup.find('page')

        # Fill dict_page
        dict_page = FillDictPage(soup, pg, p)

        # Fill dict_carton
        dict_page['cartons'] = []
        for carton_soup in soup.find_all('carton'):
            dict_carton = FillDictCarton(carton_soup)
            dict_page['cartons'].append(dict_carton)

        # Fill dict article
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
        # Addition of the dict_page to the long list.
        info_extract_pages.append(dict_page)

        if i % 500 == 0:
            print(f"{i} files extracted")

    return info_extract_pages


def TestInt(v, ref, lg, ht):
    """

    Parameters
    ----------
    v: list, tuple or numpy array
        A vector.

    ref: list of float
        A reference x and y.

    lg: float
        length.

    ht: float
        height.

    Returns
    -------
    bool
        Whether v is in the interval.

    """
    if ref[0] <= v[0] <= ref[0] + lg and ref[1] <= v[1] <= ref[1] + ht:
            return True
    return False


def DetermPrinc(dict_page):
    """
    Fill the key score of the dict_page and the field "isPrinc"

    Parameters
    ----------
    dict_page: dict
        Dict with info about a page.

    Returns
    -------
    dict_page: dict
        Updated dict with info about a page.

    """

    probas = []
    val_dict = dict_page['articles'].values()
    item_dict = dict_page['articles'].items()

    for id_art, dict_art in item_dict:

        p = 0

        # Si nb_col max p += 0.2
        nb_col = [(dict_art['nbCol'], ida) for ida, dict_art in item_dict]
        max_col = max((col for col, _ in nb_col))
        ind_max_col = [ida for ida, col in nb_col if col == max_col]
        if id_art in ind_max_col:
            p += 0.1
        elif dict_art['nbCol'] > 4:
            p += 0.1
        elif dict_art['nbCol'] < 3:
            p -= 0.1

        # Number of blocks
        if np.mean(dict_art['nbSignBlock']) > 50:
            std_nb_blocks =  dict_art['nbBlock']/40 - 0.15
            p += std_nb_blocks / 5
        else:
            p -= 0.15

        # Nb de caractères maximum
        nb_car = [(dict_art['nbSign'], ida) for ida, dict_art in item_dict]
        max_nb_car = max((nbcar for nbcar, _ in nb_car))
        ids_max_car = [ida for ida, nbcar in nb_car if nbcar == max_nb_car]
        if id_art in ids_max_car:
            p += 0.10

        # Increase of the score proportional to the number of signs
        std_nb_car = dict_art['nbSign']/8000 - 0.125
        p += std_nb_car / 2

        # Augmentation de la proba si article le plus haut
        htr = [(dict_art['y'], ida) for ida, dict_art in item_dict]
        max_htr = max((y for y, _ in htr))
        ids_max_htr = [ida for y, ida in htr if y == max_htr]
        if id_art in ids_max_htr:
            p += 0.15

        # Augmentation de la proba si article avec la plus grande aire
        # Max area possible = 126 000
        # If area <= 10 000, small article
        aire_std_art = (dict_art['width']*dict_art['height'])/126000 - 0.08
        p += aire_std_art / 3

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

        # Petittitre
        if 'petittitre' in dict_art['typeBlock']:
            p += 0.05

        # Synopsis
        if dict_art['syn'] > 0:
            p += 0.1

        # Abstract
        if dict_art['abstract'] > 0:
            p += 0.1

        # Exergue
        if dict_art['exergue'] > 0:
            p += 0.1

        # Position_Article
        if dict_art['posArt'] > 0:
            p += 0.2

        # Addition of the score of this article to the list.
        probas.append((round(p, 2), id_art))
        dict_art['score'] = p

    # En cas d'égalité
    pmax = max(probas, default=(0, 0))[0]
    if pmax == 0:
        return dict_page

    ids_max = [ida for proba, ida in probas if proba == pmax]
    if len(ids_max) > 1:
        # Il faut prendre l'article le plus haut
        htrs = [(x['y'], ida) for ida, x in item_dict if ida in ids_max]
        max_htrs = max(htrs)[0]
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
    Update the element "aire" of the dict article.

    Parameters
    ----------
    dict_page: dict
        Dict with info about the page.

    Returns
    -------
    dict_page: dict
        Updated dict.

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
    Update the element aire_img.

    Parameters
    ----------
    dict_page: dict
        The dict with the pages.

    Returns
    -------
    dict_page: dict
        Updated dict.

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
    Update distToPrinc, isMinor/sub/sec, pos et aire.

    Parameters
    ----------
    dict_page: dict
        A dict page.

    Raises
    ------
    MyException
        If an arreor during the process.

    Returns
    -------
    dict_page: dict
        Updated dict_page.

    """

    # score et isPrinc
    dict_page = DetermPrinc(dict_page)


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

    # Formerly it was the function UpdateNew
    # 'prop_img', 'nbcar', 'nbimg', 'prop_art'
    try:
        dict_page = UpdateNew4ft(dict_page)
    except Exception as e:
        str_exc = (f"An exception occurs with Update:UpdateNew4ft: \n{e}\n"
                   f"The dict_page['articles']: {dict_page['articles']}")
        raise MyException(str_exc)

    return dict_page


def UpdateAll(dict_pages):
    """
    Update dict_pages.

    Parameters
    ----------
    dict_pages: dict
        Global dict with infos about pages.

    Returns
    -------
    dict_pages: dict
        Upadted dict.

    """

    n = len(dict_pages)
    for i, dict_page in enumerate(dict_pages):
        dict_page = Update(dict_page)
        if i % (n//50) == 0:
            print(f"UpdateAll: {i/n:.2%}")
    return dict_pages


def Determ_Nat_Art(dict_page):
    """
    Update the elements isMinor, isTer, isSec, isSub.

    Parameters
    ----------
    dict_page: dict
        Global dict with infos about the pages.

    Returns
    -------
    dict_page: dict
        Updated dict.

    """

    for dict_art in dict_page['articles'].values():
        if dict_art['isPrinc'] == 0:
            # Just for the cases where we update some elements
            dict_art['isTer'] = 0
            dict_art['isSub'] = 0
            dict_art['isSec'] = 0
            dict_art['isMinor'] = 0
            if dict_art['score'] <= 0.15:
                if dict_art['score'] >= -0.05:
                    dict_art['isTer'] = 1
                else:
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
    """

    Parameters
    ----------
    corpus: list
        list of strings.

    Returns
    -------
    d: dict
        {word: nb_occurrences_word, ...}.

    """
    d = {}
    for text in corpus:
        for word in text.split():
            d[word] = d.get(word, 0) + 1
    return d


def DicoDoc(liste_mots):
    """

    Parameters
    ----------
    liste_mots: list of strings

    Returns
    -------
    dict
        {word: nb_occ_word_in_list_mots, ...}.

    """
    return {mot: liste_mots.count(mot) for mot in liste_mots}


def ConvTxt(l_mots, txt_lem):
    """
    Convert a str into a numeric vector.

    Parameters
    ----------
    l_mots: list of strings
        All the words in the corpus.

    txt_lem: str
        The str to convert into vector.

    Returns
    -------
    Vect: numpy array
        numpy.array([nb_occ_w1, nb_occ_w2, ...]).

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
    Compute the distances between the lemmatized txt and all the texts in a
    corpus.
    Update the element dist_to_princ.

    Parameters
    ----------
    dict_page: dict
        Dict with info about the pages.

    Returns
    -------
    dict_page: dict
        Updated dict.

    l_dist: list of lists
        [[score_distance, str_60_first_words], ...].

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

    for i, (text, dict_art) in enumerate(zip(corpus, val_dict)):
        # Le texte live est rajouté à la fin, d'où le N - 1.
        prod_scalaire = np.dot(Vect[N - 1], Vect[i])
        norme2 = np.sqrt(np.dot(Vect[i], Vect[i]))
        prod_normes = norme1 * norme2 if norme1 * norme2 != 0 else 1
        res = round(prod_scalaire / prod_normes, 2)
        l_dist.append([res, text[:60]])
        # Mise à jour de l'élément distToPrinc
        dict_art['distToPrinc'] = res

    # We sort the list
    l_dist.sort(reverse=True)

    return dict_page, l_dist


def UpdateNew4ft(dict_page):
    """
    Update the elements:
        - 'prop_img'
        - 'nbcar'
        - 'nbimg'
        - 'prop_art'

    Parameters
    ----------
    dict_page: dict
        Global dict with all the pages.

    Returns
    -------
    dict_page: dict
        Updated dict.

    """

    # Rajout de l'élément prop_art à chaque ntuple_page
    aire_img_tot, nbcar_tot, nbimg_tot, aire_tot = 0, 0, 0, 0
    for dict_art in dict_page['articles'].values():
        aire_tot += dict_art['width'] * dict_art['height']
        nbcar_tot += dict_art['nbSign']
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

    return dict_page


def CreationBDD(list_dict_pages):
    """
    From the list of dicts about the pages.

    Parameters
    ----------
    list_dict_pages: list of dicts
        Global list with all the dicts about the pages.

    Returns
    -------
    dico_bdd: dict
        Global dict_pages.

    """

    dico_bdd = {}
    for dict_page in list_dict_pages:
        try:
            if len(dict_page['articles']) > 0:
                id_p = dict_page['melodyId']
                dico_bdd[id_p] = dict_page
        except:
            pass
    return dico_bdd


def UpdateAireImg(dict_art):
    """
    Update the element aireImg

    Parameters
    ----------
    dict_art: dict
        A dict article.

    Returns
    -------
    dict_art: dict
        An updated dict article.

    """
    aire = 0
    for dict_img in dict_art['photo']:
        try:
            aire += dict_img['width'] * dict_img['height']
        except:
            pass
    dict_art['aireImg'] = aire
    return dict_art


def UpdateAllAireImg(dict_page):
    """
    Update all the dict_article in a dict_page.

    Parameters
    ----------
    dict_page: dict
        A dict page.

    Returns
    -------
    dict_page: dict
        An updated dict page.

    """
    for dict_art in dict_page['articles'].values():
        dict_art = UpdateAireImg(dict_art)
    return dict_page


def ShowElementsDicoBDD(dico_bdd):
    """
    Shows a random element of the dict.

    Parameters
    ----------
    dico_bdd: dict
        Global dict pages.

    Returns
    -------
    None.

    """
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


def CreationDictPages(rep_data, dir_out):
    """
    Create the dict_pages and the dict_arts.

    Parameters
    ----------
    rep_data: str
        The repertory with the files.

    dir_out: str
        The repertory where we store the data.

    Returns
    -------
    None.

    """
    t0 = time.time()
    list_dict_pages = ExtrationData(rep_data)
    print(f"ExtrationData duration: {time.time() - t0} sec.")

    t1 = time.time()
    list_dict_pages = UpdateAll(list_dict_pages)
    print("UpdateAll : {:.2f} sec".format(time.time() - t1))

    t3 = time.time()
    dico_bdd = CreationBDD(list_dict_pages)
    # Creation of the dict_arts
    dict_arts = {}
    for idp, dicop in dico_bdd.items():
        for ida, dicta in dicop['articles'].items():
            dict_arts[ida] = dicta
    print("CreationBDD and dict_arts: {:.2f} sec".format(time.time() - t3))

    # shows the elements in the prompt
    ShowElementsDicoBDD(dico_bdd)

    # save the dicts
    with open(dir_out + 'dict_pages', 'wb') as f:
        pickle.dump(dico_bdd, f)
    with open(dir_out + 'dict_arts', 'wb') as f:
        pickle.dump(dict_arts, f)

