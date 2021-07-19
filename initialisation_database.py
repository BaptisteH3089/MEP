#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 15:57:50 2021

@author: baptistehessel

Script that creates all the files that will be used to compute proposals
of page for a given client.

creation_dict_bdd.py:

creation_dict_layout.py:
    - dico_pages

"""
import pickle
import argparse
import creation_dict_bdd
import creation_dict_layout
import creation_dict_page_array
import creation_dict_page_array_fast
import creation_list_mdp_data
import sys
import os

parser = argparse.ArgumentParser(description="Create the customer's database.")
parser.add_argument('rep_data',
                    help='The directory with the xml files.',
                    type=str)
parser.add_argument('dir_customer',
                    help='The directory where we will store the objects.',
                    type=str)
parser.add_argument('--list_mdp',
                    help='Whether to create this list which takes time.',
                    type=bool,
                    default=True)
args = parser.parse_args()

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
    x = input("Continue ? y/n", )
    if x == 'n':
        print("Script interrupted.")
        sys.exit()
else:
    # We make the directory
    os.mkdir(dir_customer)

# We begin by creating the dict_pages
x = str(input("Creation of dict_pages ? y/n "))
if x == 'y':
    creation_dict_bdd.CreationDictPages(args.rep_data, dir_customer)

# Then, the dict_layouts
# Loading of the dictionary with all information on pages
with open(dir_customer + 'dict_pages','rb') as f:
    dict_pages = pickle.load(f)
x = str(input("Creation of dict_layouts ? y/n "))
if x == 'y':
    creation_dict_layout.CreationDictLayoutsSmall(dict_pages, dir_customer)

# Then, dict_page_array
x = str(input("Creation of dict_page_array ? y/n "))
if x == 'y':
    creation_dict_page_array.CreationDictPageArray(dict_pages, dir_customer)

# Also, dict_page_array_fast
with open(dir_customer + 'dict_page_array','rb') as f:
    dict_page_array = pickle.load(f)

x = str(input("Creation of dict_page_array_fast ? y/n "))
if x == 'y':
    args_fast = [dict_page_array, dir_customer]
    creation_dict_page_array_fast.CreationDictPageArrayFast(*args_fast)

# Finally, list_mdp_data if we want to wait...
with open(dir_customer + 'dict_page_array_fast', 'rb') as file:
    dict_page_array_fast = pickle.load(file)
if args.list_mdp:
    creation_list_mdp_data.CreationListMdp(dict_page_array_fast, dir_customer)
