#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 15:57:50 2021

@author: baptistehessel

Script that creates all the files that will be used to compute proposals
of page for a given client. It creates the following objects:
    - dict_pages
    - dict_arts
    - list_mdp
    - dict_page_array
    - dict_page_array_fast
    - dict_layouts

"""
import pickle
import argparse
import sys
import os
import creation_dict_bdd
import creation_dict_layout
import creation_dict_page_array
import creation_dict_page_array_fast
import creation_list_mdp_data
import time


parser = argparse.ArgumentParser(description="Create the customer's database.")
parser.add_argument('rep_data',
                    help=('The directory with the xml files. For example: '
                          '/data/montageia/in/export'),
                    type=str)
parser.add_argument('dir_customer',
                    help='The directory where we will store the objects.',
                    type=str)
parser.add_argument('--no_list_mdp',
                    help=('Whether to create this list which takes time. If '
                          'you add --no_list_mdp to the command line. We do '
                          'not create that file.'),
                    action='store_false')
args = parser.parse_args()


print(f"args.no_list_mdp: {args.no_list_mdp}.")

# Initialisation of the time
t0 = time.time()

# Test about rep_data
if os.path.isdir(args.rep_data):
    print("The argument rep_data seems ok.")
else:
    print(f"rep_data is not a directory \n{args.rep_data}")
    print("Exit()")
    sys.exit()


# Test about dir_customer
if args.dir_customer[-1] == '/':
    dir_customer = args.dir_customer
else:
    dir_customer = args.dir_customer + '/'

# Test wether dir_customer is a directory
# If yes, just a warning. Else we create the directory
if os.path.isdir(dir_customer):
    print("The directory already exists.")
    print("In this directory, there are the following entries:")
    for i, entry in enumerate(os.listdir(dir_customer)):
        print(entry)
        if i == 12:
            print("And maybe more entries...")
            break
else:
    # We make the directory
    print("Creation of the directory.")
    os.mkdir(dir_customer)


# We begin by creating the dict_pages and also the dict_arts
print("Beginning of the creation of the dict_pages.")
creation_dict_bdd.CreationDictPages(args.rep_data, dir_customer)
print("dict_pages. Over.")
print(f"Duration from the beginning: {time.time() - t0:.2f} sec.\n\n")


# Then, the dict_layouts
# Loading of the dictionary with all information on pages
with open(dir_customer + 'dict_pages','rb') as f:
    dict_pages = pickle.load(f)

print("Beginning of the creation of the dict_layouts.")
creation_dict_layout.CreationDictLayoutsSmall(dict_pages, dir_customer)
print("dict_layouts. Over.")
print(f"Duration from the beginning: {time.time() - t0:.2f} sec.\n\n")


# Then, dict_page_array
print("Beginning of the creation of the dict_page_array.")
creation_dict_page_array.CreationDictPageArray(dict_pages, dir_customer)
print("dict_page_array. Over.")
print(f"Duration from the beginning: {time.time() - t0:.2f} sec.\n\n")


# Also, dict_page_array_fast
print("Beginning of the creation of the dict_page_array_fast.")
with open(dir_customer + 'dict_page_array','rb') as f:
    dict_page_array = pickle.load(f)

args_fast = [dict_page_array, dir_customer]
creation_dict_page_array_fast.CreationDictPageArrayFast(*args_fast)
print("dict_page_array_fast. Over.")
print(f"Duration from the beginning: {time.time() - t0:.2f} sec.\n\n")


# Finally, list_mdp_data if we want to wait...
if args.no_list_mdp is False:
    print("We do not create the list_mdp. Script over.")
    sys.exit()
else:
    with open(dir_customer + 'dict_page_array_fast', 'rb') as file:
        dict_page_array_fast = pickle.load(file)
    print("Beginning of the creation of the list_mdp.")
    creation_list_mdp_data.CreationListMdp(dict_page_array_fast, dir_customer)
    print("list_mdp. Over.")

print(f"Total duration: {time.time() - t0:.2f} sec.\n\n")
