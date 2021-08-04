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
import json
import propositions # script with the code used for the requests with layout
import without_layout # script for the case with no layout input
import add_page # script used to add a page to the data
import delete_page # script used to delete a page from the data


# Class to write some personalized exceptions
class MyException(Exception):
    pass


# app is an instance of the Flask class
app = Flask(__name__)
api = Api(app)


# ARG PARSER - SERVER

# Initilisation of the arguements of the script montage_ia.xml for the start
# webservice montage IA.
main_parser = argparse.ArgumentParser(description='Lancmt server montageIA')
main_parser.add_argument('customer_code',
                         help="The code of a client to select the good data.",
                         type=str)
main_parser.add_argument('path_param_file',
                         help="The absolute path of the parameters file.",
                         type=str)
main_args = main_parser.parse_args()


# REQ PARSER - REQUESTS

# The arguments that will be used for the requests.
str_h_file_in = "The path of the archive zip with the xml files associated to"
str_h_file_in += "the articles and the page layout"
str_h_file_out = ("The path of the xml file output with the pages proposed by"
                  " the algorithm")
parser = reqparse.RequestParser()
# The path of the input
parser.add_argument('file_in', help=str_h_file_in, type=str)
# The path of the output
parser.add_argument('file_out', help=str_h_file_out, type=str)


# PARAM FILE

# Parsing of the parameters file.
config = configparser.RawConfigParser()
# Reading of the parameters file (param_montage_ia.py) of the appli montageIA.
config.read(main_args.path_param_file)
# The parameters extracted from the parameters file.
param = {'path_log': config.get('LOG', 'path_log'),
         'path_data': config.get('DATA', 'path_data'),
         'port': config.get('CUSTOMERS', main_args.customer_code),
         'list_features': json.loads(config.get('FEATURES', 'list_features'))}

# LOGGER

# Initialisation of the logger
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


# OPENING FILES IN THE DATABASE

# Creation of the path_customer
if param['path_data'][-1] == '/':
    path_customer = param['path_data'] + main_args.customer_code + '/'
else:
    path_customer = param['path_data'] + '/' + main_args.customer_code + '/'

# The dict_pages with all the infos about the pages.
with open(path_customer + 'dict_pages', 'rb') as file:
    dict_pages = pickle.load(file)

# Opening of the list of the page layouts used by this client
try:
    with open(path_customer + 'list_mdp', 'rb') as file:
        list_mdp = pickle.load(file)
except Exception as e:
    logger.error(e, exc_info=True)
    logger.debug('Path to list_mdp: {}'.format(path_customer + 'list_mdp'))

# The dict_arts {ida: dicoa, ...}
try:
    with open(path_customer + 'dict_arts', 'rb') as file:
        dict_arts = pickle.load(file)
except Exception as e:
    logger.error(e, exc_info=True)
    logger.debug('Path to dict_arts: {}'.format(path_customer + 'dict_arts'))

# Loading dictionary with all the pages and the array corresponding
try:
    with open(path_customer + 'dict_page_array', 'rb') as f:
        dict_page_array = pickle.load(f)
except Exception as e:
    logger.error(e, exc_info=True)
    path_dict_p_a = path_customer + 'dict_page_array'
    logger.debug('Path to dict_page_array: {}'.format(path_dict_p_a))

# Loading dictionary with all the pages
try:
    with open(path_customer + 'dict_page_array_fast', 'rb') as f:
        dict_page_array_fast = pickle.load(f)
except Exception as e:
    logger.error(e, exc_info=True)
    path_dict_p_a = path_customer + 'dict_page_array_fast'
    logger.debug('Path to dict_page_array_fast: {}'.format(path_dict_p_a))

# Loading dictionary with all the layouts with an MelodyId
with open(path_customer + 'dict_layouts', 'rb') as f:
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

# The GBC trained with pages with 4 articles
try:
    with open(path_customer + 'gbc4', 'rb') as file:
        gbc4 = pickle.load(file)
except Exception as e:
    logger.error(e, exc_info=True)
    logger.debug('Path to gbc4: {}'.format(path_customer + 'gbc4'))

# The GBC trained with pages with 5 articles
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
        print("\n\n{:-^80}\n".format("BEGINNING ALGO"))

        list_features = param['list_features']
        print(f"The features used:\n{list_features}. {type(list_features)}.")

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
            path_data_input_trunc = dir_file_in + '/input/'
            with zipfile.ZipFile(file_in, "r") as z:
                z.extractall(path=path_data_input_trunc)
            print(f"path_data_input: {path_data_input} \n"
                  f"path_data_input_trunc: {path_data_input_trunc}\n")

            directories = os.listdir(path_data_input)
            print(f"directories: {directories}\n")
            if 'pageTemplate' in directories:
                print("{:-^80}".format(""))
                print("{:-^80}".format("| Case with layout input |"))
                print("{:-^80}\n".format(""))
                args_lay = [dict_pages, list_mdp, path_data_input, file_out]
                args_lay += [list_features]
                propositions.ExtractAndComputeProposals(*args_lay)
            else:
                print("{:-^80}".format(""))
                print("{:-^80}".format("| Case without layout input |"))
                print("{:-^80}\n".format(""))

                args_nolay = [path_data_input, file_out, dict_pages, dict_arts]
                args_nolay += [list_mdp, dict_gbc, dict_layouts, list_features]
                without_layout.FinalResultsMethodNoLayout(*args_nolay)
            logger.info('End of GET')
        except Exception as e:
            logger.error(e, exc_info=True)
        return "{:-^35.2f} sec".format(time.time() - t0)


# Initialisation of the counter. Every 100 modifs, we save the database.
COUNTER = 1


class AddPage(Resource):
    """
    Adds the information of a published page in the different files of the
    database.

    """
    def post(self):
        # Counts the number of modification of the database.
        global dict_pages
        global dict_arts
        global list_mdp
        global dict_page_array
        global dict_page_array_fast
        global dict_layouts
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
            logger.info('file_out: {}'.format(args['file_out']))
            args_add = [args['file_in'], args['file_out'], dict_pages]
            args_add += [dict_arts, list_mdp, dict_page_array]
            args_add += [dict_page_array_fast, dict_layouts]
            # Verification of the initial length of the inputs
            a, b, c = 'dict_pages', 'dict_arts', 'list_mdp'
            d, e, f = 'dict_page_array', 'dict_page_array_fast', 'dict_layouts'
            lena, lenb, lenc = len(dict_pages), len(dict_arts), len(list_mdp)
            lend = len(dict_page_array)
            lene = len(dict_page_array_fast)
            lenf = len(dict_layouts)
            print(f"Initial length of {a:>25} {lena:>10} {b:>25} {lenb:>10}")
            print(f"Initial length of {c:>25} {lenc:>10} {d:>25} {lend:>10}")
            print(f"Initial length of {e:>25} {lene:>10} {f:>25} {lenf:>10}")
            database = add_page.AddOnePageToDatabase(*args_add)
            dict_pages = database[0]
            dict_arts = database[1]
            list_mdp = database[2]
            dict_page_array = database[3]
            dict_page_array_fast = database[4]
            dict_layouts = database[5]
                # Verification of the length after addition
            lena, lenb, lenc = len(dict_pages), len(dict_arts), len(list_mdp)
            lend = len(dict_page_array)
            lene = len(dict_page_array_fast)
            lenf = len(dict_layouts)
            print(f"After length of {a:>27} {lena:>10} {b:>25} {lenb:>10}")
            print(f"After length of {c:>27} {lenc:>10} {d:>25} {lend:>10}")
            print(f"After length of {e:>27} {lene:>10} {f:>25} {lenf:>10}")
            if save_bdd:
                with open(path_customer + 'dict_pages', 'wb') as f:
                    pickle.dump(dict_pages, f)
                with open(path_customer + 'dict_arts', 'wb') as f:
                    pickle.dump(dict_arts, f)
                with open(path_customer + 'list_mdp', 'wb') as f:
                    pickle.dump(list_mdp, f)
                with open(path_customer + 'dict_page_array', 'wb') as f:
                    pickle.dump(dict_page_array, f)
                with open(path_customer + 'dict_page_array_fast', 'wb') as f:
                    pickle.dump(dict_page_array_fast, f)
                with open(path_customer + 'dict_layouts', 'wb') as f:
                    pickle.dump(dict_layouts, f)
                COUNTER = 0
            else:
                COUNTER += 1
            logger.info('End of POST')
        except Exception as e:
            logger.error(e, exc_info=True)
            logger.debug("path_customer: {}".format(path_customer))
        return "{:-^35.2f} sec".format(time.time() - t0)


class DeletePage(Resource):
    def delete(self):
        global COUNTER
        global dict_pages
        global dict_arts
        global list_mdp
        global dict_page_array
        global dict_page_array_fast
        global dict_layouts
        t0 = time.time()
        args = parser.parse_args()
        if COUNTER % 100 == 0:
            save_bdd = True
            logger.info('After deletion, the database will be saved.')
        else:
            save_bdd = False
        logger.info('file_in: {}'.format(args['file_in']))
        # Verification of the intitial length of the inputs.
        a, b, c = 'dict_pages', 'dict_arts', 'list_mdp'
        d, e, f = 'dict_page_array', 'dict_page_array_fast', 'dict_layouts'
        lena, lenb, lenc = len(dict_pages), len(dict_arts), len(list_mdp)
        lend = len(dict_page_array)
        lene = len(dict_page_array_fast)
        lenf = len(dict_layouts)
        print(f"Initial length of {a:>25} {lena:>10} {b:>25} {lenb:>10}")
        print(f"Initial length of {c:>25} {lenc:>10} {d:>25} {lend:>10}")
        print(f"Initial length of {e:>25} {lene:>10} {f:>25} {lenf:>10}")
        database = delete_page.DeleteOnePageFromDatabase(args['file_in'],
                                                         args['file_out'],
                                                         dict_pages,
                                                         dict_arts,
                                                         list_mdp,
                                                         dict_page_array,
                                                         dict_page_array_fast,
                                                         dict_layouts)
        dict_pages = database[0]
        dict_arts = database[1]
        list_mdp = database[2]
        dict_page_array = database[3]
        dict_page_array_fast = database[4]
        dict_layouts = database[5]
        # Verification of the lengths after deletion.
        lena, lenb, lenc = len(dict_pages), len(dict_arts), len(list_mdp)
        lend = len(dict_page_array)
        lene = len(dict_page_array_fast)
        lenf = len(dict_layouts)
        print(f"After length of {a:>27} {lena:>10} {b:>25} {lenb:>10}")
        print(f"After length of {c:>27} {lenc:>10} {d:>25} {lend:>10}")
        print(f"After length of {e:>27} {lene:>10} {f:>25} {lenf:>10}")
        if save_bdd:
            with open(path_customer + 'dict_pages', 'wb') as f:
                pickle.dump(dict_pages, f)
            with open(path_customer + 'dict_arts', 'wb') as f:
                pickle.dump(dict_arts, f)
            with open(path_customer + 'list_mdp', 'wb') as f:
                pickle.dump(list_mdp, f)
            with open(path_customer + 'dict_page_array', 'wb') as f:
                pickle.dump(dict_page_array, f)
            with open(path_customer + 'dict_page_array_fast', 'wb') as f:
                pickle.dump(dict_page_array_fast, f)
            with open(path_customer + 'dict_layouts', 'wb') as f:
                pickle.dump(dict_layouts, f)
            COUNTER = 1
        else:
            COUNTER += 1
            print(f"The COUNTER: {COUNTER:->}")
        logger.info('End of DELETE')
        return "{:-^35.2f} sec".format(time.time() - t0)


api.add_resource(GetLayout, '/extract')
api.add_resource(AddPage, '/add')
api.add_resource(DeletePage, '/delete')

if __name__ == '__main__':
    app.run(debug=True, port=param['port'])
