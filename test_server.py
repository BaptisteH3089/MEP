#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 17:12:33 2021

@author: baptistehessel
"""
import zipfile
import argparse
import os
import shutil

class MyException(Exception):
    pass

parser = argparse.ArgumentParser(description='Lancmt server montageIA')
parser.add_argument('path_zip',
                    help="The code of a client to select the good data.",
                    type=str)
parser.add_argument('path_unzip',
                    help="The code of a client to select the good data.",
                    type=str)
args = parser.parse_args()


print(f"args.path_zip: {args.path_zip}")
print(f"os.path.isdir: {os.path.isdir(args.path_zip)}")
print(f"os.path.exists(my_file): {os.path.exists(args.path_zip)}")

#with zipfile.ZipFile(args.path_zip, "r") as z:
    #z.extractall(path=args.path_unzip)




def fixBadZipfile(zipFile):
     f = open(zipFile, 'r+b')
     data = f.read()
     pos = data.find(b'\x50\x4b\x05\x06') # End of central directory signature
     if (pos > 0):
         print("Trancating file at location " + str(pos + 22)+ ".")
         f.seek(pos + 22)   # size of 'ZIP end of central directory record'
         f.truncate()
         f.close()
     else:
         # raise error, file is truncated
         raise MyException('truncated file')


fixBadZipfile(args.path_zip)

shutil.unpack_archive(args.path_zip, extract_dir=args.path_unzip)





