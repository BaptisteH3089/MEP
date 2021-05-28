#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from flask import Flask
from flask_restful import reqparse, Api, Resource
from logging.handlers import RotatingFileHandler
import configparser
import argparse
import logging
import pickle
import time
import propositions


# Classe pour écrire des exceptions personnalisées
class MyException(Exception):
    pass


# app est une instance de la classe Flask
app = Flask(__name__)
api = Api(app)

# ARG PARSER - SERVEUR. Initialisation des arguments du script pour le
# lancement du webservice montage IA.
main_parser = argparse.ArgumentParser(description='Lancmt serveur montageIA')
main_parser.add_argument('customer_code',
                         help="Le code du client pour sélection données.",
                         type=str)
main_parser.add_argument('path_param_file',
                         help="Le chemin du fichier de paramètres.",
                         type=str)
main_args = main_parser.parse_args()

# REQ PARSER - REQUEST. Les arguments qui seront utilisés pour les requêtes.
str_h_file_in = "Le chemin de l'archive avec les fichiers xml associés aux "
str_h_file_in += "articles et les MDP"
str_h_file_out = "Le chemin du fichier xml avec les placements des articles"
parser = reqparse.RequestParser()
parser.add_argument('file_in', help=str_h_file_in, type=str)
parser.add_argument('file_out', help=str_h_file_out, type=str)

# FICHIER PARAM. Parsage du fichier de paramètres.
config = configparser.RawConfigParser()
# Lecture du fichier de paramètres situé dans le même répertoire que le script
config.read(main_args.path_param_file)
# Les paramètres extrait du fichier de configurations
param = {'path_log': config.get('LOG', 'path_log'),
         'path_data': config.get('DATA', 'path_data')}

# LOGGER. Initialisation du Logger
logger = logging.getLogger('my_logger')
str_fmt = '%(asctime)s :: %(filename)s :: %(levelname)s :: %(message)s'
formatter = logging.Formatter(str_fmt)
handler = RotatingFileHandler(param['path_log'], maxBytes=30000,
                              backupCount=3)
logger.setLevel(logging.DEBUG)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info('Start of the app montage IA.')

# Ouverture des fichiers data associés à ce client
# Création du path_customer suivant si path_data finit par '/' ou non.
if param['path_data'][-1] == '/':
    path_customer = param['path_data'] + main_args.customer_code + '/'
else:
    path_customer = param['path_data'] + '/' + main_args.customer_code + '/'

# Ouverture de la BDD avec les pages archivées de ce client et toutes les
# infos associées à ces pages
try:
    with open(path_customer + 'dico_pages', 'rb') as file:
        dico_bdd = pickle.load(file)
except Exception as e:
    logger.error(e, exc_info=True)
    logger.debug('Path to dico_bdd: {}'.format(path_customer + 'dico_pages'))
# Ouverture de la liste des MDP utilisés par ce client
try:
    with open(path_customer + 'list_mdp', 'rb') as file:
        list_mdp_data = pickle.load(file)
except Exception as e:
    logger.error(e, exc_info=True)
    logger.debug('Path to list_mdp: {}'.format(path_customer + 'list_mdp'))


class GetLayout(Resource):
    """
    - Lit les fichiers xml dans l'archive située à l'endroit file_in
    - Détermine les pages qu'on peut créer avec ces articles INPUT et les MDP
    INPUT
    - Crée un fichier xml à l'endroit file_out avec les placements des
    articles INPUT dans les MDP INPUT
    """
    def get(self):
        t0 = time.time()
        args = parser.parse_args()
        try:
            propositions.ExtractAndComputeProposals(dico_bdd,
                                                    list_mdp_data,
                                                    args['file_in'],
                                                    args['file_out'])
            logger.info('End of GET')
        except Exception as e:
            logger.error(e, exc_info=True)
            logger.debug("args['file_in']: {}".format(args['file_in']))
            logger.debug("args['file_out']: {}".format(args['file_out']))
        return "{:-^35.2f} sec".format(time.time() - t0)


class AddPage(Resource):
    """
    Ajoute les informations d'une page au dico_bdd et à la liste des MDP
    list_mdp_data.
    """
    def post(self):
        t0 = time.time()
        args = parser.parse_args()
        if COUNTER % 100 == 0:
            save_bdd = True
            logger.info('After addition, the database will be saved.')
        else:
            save_bdd = False
        try:
            add_page.AddPageBDD(dico_bdd,
                                list_mdp_data,
                                args['file_in'],
                                save_bdd,
                                path_customer)
            COUNTER += 1
            logger.info('End of POST')
        except Exception as e:
            logger.error(e, exc_info=True)
            logger.debug("args['file_in']: {}".format(args['file_in']))
            logger.debug("path_customer: {}".format(path_customer))
        return "{:-^35.2f} sec".format(time.time() - t0)
        

api.add_resource(GetLayout, '/extract')
api.add_resource(AddPage, '/add')

if __name__ == '__main__':
    app.run(debug=True)