#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: baptistehessel

Main script that launches the servor.
The objects necessary are:
    - dict_pages
    - dict_arts
    - list_mdp
    - dict_page_array
	- dict_layouts_small
	- gbc2
	- gbc3
	- gbc4
	- gbc5

"""
from flask import Flask
from flask_restful import reqparse, Api, Resource
from logging.handlers import RotatingFileHandler
import configparser
import argparse
import zipfile
import logging
import pickle
import time
import os
import propositions # script with the code used for the requests with layout
import without_layout # script for the case with no layout input


# Class to write some personalized exceptions
class MyException(Exception):
    pass


# app is an instance of the Flask class
app = Flask(__name__)
api = Api(app)

# ARG PARSER - SERVOR. Initilisation of the arguements of the script
# montage_ia.xml for the start webservice montage IA.
main_parser = argparse.ArgumentParser(description='Lancmt server montageIA')
main_parser.add_argument('customer_code',
                         help="The code of a client to select the good data.",
                         type=str)
main_parser.add_argument('path_param_file',
                         help="The absolute path of the parameters file.",
                         type=str)
main_args = main_parser.parse_args()

str_h_file_in = "The path of the archive zip with the xml files associated to"
str_h_file_in += "the articles and the page layout"
str_h_file_out = ("The path of the xml file output with the pages proposed by"
                  " the algorithm")
# REQ PARSER - REQUEST. The arguments that will be used for the requests.
parser = reqparse.RequestParser()
# The path of the input
parser.add_argument('file_in', help=str_h_file_in, type=str)
# The path of the output
parser.add_argument('file_out', help=str_h_file_out, type=str)

# PARAM FILE. Parsing of the parameters file
config = configparser.RawConfigParser()
# Reading of the parameters file (param_montage_ia.py) of the appli montageIA
config.read(main_args.path_param_file)
# The parameters extracted from the parameters file
param = {'path_log': config.get('LOG', 'path_log'),
         'path_data': config.get('DATA', 'path_data'),
         'port': config.get('CUSTOMERS', main_args.customer_code)}

# LOGGER. Initialisation of the logger
logger = logging.getLogger('my_logger')
str_fmt = '%(asctime)s :: %(filename)s :: %(levelname)s :: %(message)s'
formatter = logging.Formatter(str_fmt)
# We write the logs in the directory given by the parameters file
handler = RotatingFileHandler(param['path_log'], maxBytes=30000,
                              backupCount=3)
logger.setLevel(logging.DEBUG)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info('Start of the app montage IA.')

# Creation of the path_customer
if param['path_data'][-1] == '/':
    path_customer = param['path_data'] + main_args.customer_code + '/'
else:
    path_customer = param['path_data'] + '/' + main_args.customer_code + '/'

# Opening of the Database with the archive files of that client
try:
    with open(path_customer + 'dict_pages', 'rb') as file:
        dico_bdd = pickle.load(file)
except Exception as e:
    logger.error(e, exc_info=True)
    logger.debug('Path to dict_bdd: {}'.format(path_customer + 'dict_pages'))
# Opening of the list of the page layouts used by this client
try:
    with open(path_customer + 'list_mdp', 'rb') as file:
        list_mdp_data = pickle.load(file)
except Exception as e:
    logger.error(e, exc_info=True)
    logger.debug('Path to list_mdp: {}'.format(path_customer + 'list_mdp'))
# The dict {ida: dicoa, ...}
try:
    with open(path_customer + 'dict_arts', 'rb') as file:
        dict_arts = pickle.load(file)
except Exception as e:
    logger.error(e, exc_info=True)
    logger.debug('Path to dict_arts: {}'.format(path_customer + 'dict_arts'))
# Loading dictionary with all the pages
try:
    with open(path_customer + 'dict_page_array', 'rb') as f:
        dict_page_array = pickle.load(f)
except Exception as e:
    logger.error(e, exc_info=True)
    path_dict_p_a = path_customer + 'dict_page_array'
    logger.debug('Path to dict_page_array: {}'.format(path_dict_p_a))
# Loading dictionary with all the layouts with an MelodyId
with open(path_customer + 'dict_layouts_small', 'rb') as f:
    dict_layouts = pickle.load(f)
# The Gradient Boosting Classifier trained with pages with 2 articles
try:
    with open(path_customer + 'gbc2', 'rb') as file:
        gbc2 = pickle.load(file)
except Exception as e:
    logger.error(e, exc_info=True)
    logger.debug('Path to gbc2: {}'.format(path_customer + 'gbc2'))
# The Gradient Boosting Classifier trained with pages with 3 articles
try:
    with open(path_customer + 'gbc3', 'rb') as file:
        gbc3 = pickle.load(file)
except Exception as e:
    logger.error(e, exc_info=True)
    logger.debug('Path to gbc3: {}'.format(path_customer + 'gbc3'))
# The Gradient Boosting Classifier trained with pages with 4 articles
try:
    with open(path_customer + 'gbc4', 'rb') as file:
        gbc4 = pickle.load(file)
except Exception as e:
    logger.error(e, exc_info=True)
    logger.debug('Path to gbc4: {}'.format(path_customer + 'gbc4'))
# The Gradient Boosting Classifier trained with pages with 5 articles
try:
    with open(path_customer + 'gbc5', 'rb') as file:
        gbc5 = pickle.load(file)
except Exception as e:
    logger.error(e, exc_info=True)
    logger.debug('Path to gbc5: {}'.format(path_customer + 'gbc5'))
# Dictionary with all the models
dict_gbc = {2: gbc2, 3: gbc3, 4: gbc4, 5: gbc5}


class GetLayout(Resource):
    """

    - Read the xml files in the archive located in the directory "file_in"
    - Extraction of the propositions of page that we can create with the
    articles input and the page layout input
    - Create a xml file in the directory "file_out" with several propositions
    of pages. In fact we simply associate some ids of articles with some ids
    of cartons.

    """
    def get(self):
        t0 = time.time()
        args = parser.parse_args()
        try:
            file_in = args['file_in']
            file_out = args['file_out']
            logger.info('file_in: {}'.format(file_in))
            logger.info('file_out: {}'.format(file_out))
            try:
                dir_file_in = os.path.dirname(file_in)
            except Exception as e:
                str_exc = (f"Error with os.path.dirname(): {e}\n"
                           "file_in: {file_in}")
                raise MyException(str_exc)
            try:
                basename_file_in = os.path.basename(file_in)
            except Exception as e:
                str_exc = (f"Error with os.path.basename(): {e}\n"
                           "file_in: {file_in}")
                raise MyException(str_exc)
            # We remove the .zip in the basename_file_in
            path_data_input = dir_file_in + '/input/' + basename_file_in[:-4]
            with zipfile.ZipFile(file_in, "r") as z:
                z.extractall(path_data_input)
            print(f"path_data_input: {path_data_input}")
            directories = os.listdir(path_data_input)
            print(f"directories: {directories}")
            if 'pageTemplate' in directories:
                print("{:-^80}".format(""))
                print("{:-^80}".format("| Case with layout input |"))
                print("{:-^80}".format(""))
                args_lay = [dico_bdd, list_mdp_data, path_data_input, file_out]
                propositions.ExtractAndComputeProposals(*args_lay)
            else:
                print("{:-^80}".format(""))
                print("{:-^80}".format("| Case without layout input |"))
                print("{:-^80}".format(""))
                args_nolay = [path_data_input, file_out, dico_bdd, dict_arts]
                args_nolay += [list_mdp_data, dict_gbc, dict_layouts, ]
                without_layout.FinalResultsMethodNoLayout(*args_nolay)
            logger.info('End of GET')
        except Exception as e:
            logger.error(e, exc_info=True)
        return "{:-^35.2f} sec".format(time.time() - t0)


# Initialisation of the counter. Every 100 modifs, we save the database.
COUNTER = 0


class AddPage(Resource):
    """
    Adds the information of a published page in the different files of the
    database.
    Unused for the moment.

    """
    def post(self):
        # Counts the number of modification of the database.
        global COUNTER
        t0 = time.time()
        args = parser.parse_args()
        if COUNTER % 100 == 0:
            save_bdd = True
            logger.info('After addition, the database will be saved.')
        else:
            save_bdd = False
        try:
            logger.info('file_in: {}'.format(args['file_in']))
            add_page.AddPageBDD(dico_bdd,
                                list_mdp_data,
                                args['file_in'],
                                save_bdd,
                                path_customer)
            COUNTER += 1
            logger.info('End of POST')
        except Exception as e:
            logger.error(e, exc_info=True)
            logger.debug("path_customer: {}".format(path_customer))
        return "{:-^35.2f} sec".format(time.time() - t0)


api.add_resource(GetLayout, '/extract')
api.add_resource(AddPage, '/add')

if __name__ == '__main__':
    app.run(debug=True, port=param['port'])
