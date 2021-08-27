#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 17:47:28 2021

@author: baptistehessel
"""


import shutil
import zipfile
import os

os.listdir('/Users/baptistehessel/Desktop')

class MyException(Exception):
    pass


path_zip = '/Users/baptistehessel/Documents/DAJ/MEP/montageIA/in/propositionIA2.zip'
path_out = '/Users/baptistehessel/Desktop/TEST2'
# shutil.unpack_archive(path_zip, extract_dir=path_out)


def fixBadZipfile(zipFile):
    f = open(zipFile, 'r+b')
    data = f.read()
    pos = data.find(b'\x50\x4b\x05\x06') # End of central directory signature
    if (pos > 0):
        print("Truncating file at location " + str(pos + 22) + ".")
        f.seek(pos + 22)   # size of 'ZIP end of central directory record'
        f.truncate()
        f.close()
    else:
        # raise error, file is truncated
        raise MyException('truncated file')
    return f


def fixBadZipfile2(zipFile):
    f = open(zipFile, 'r+b')
    content = f.read()
    # reverse find: this str is the end of the zip's central directory.
    pos = content.rfind(b'\x50\x4b\x05\x06')
    if pos>0:
        print(f"pos + 22: {pos+22}")
        f.seek(pos+20)
        f.truncate()
        # Zip file comment length: 0 byte length; tell zip app to stop reading.
        f.write(b'\x00\x00')
        f.seek(0)
    return f


f2 = fixBadZipfile2(path_zip)

f = fixBadZipfile(path_zip)

with zipfile.ZipFile(path_zip, "r") as z:
    z.extractall(path=path_out)


print(zipfile.is_zipfile(path_zip))




#%%

with zipfile.ZipFile(path_zip, 'r') as zipObj:
   # Get a list of all archived file names from the zip
   listOfFileNames = zipObj.namelist()
   # Iterate over the file names
   for fileName in listOfFileNames:
       print(fileName)
