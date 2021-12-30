#!/usr/bin/python
# -*- coding: utf-8 -*-
#====================================================================
# @author: Joe Del Rocco
# @since: 10/25/2016
# @summary: A module with general useful functionality.
#====================================================================
import os
import shutil
import re
from datetime import datetime


'''
Function to log to file and standard out with one call.
'''
def Log(args, msg):
    msg = str(msg)
    print(msg)
    with open(args.logpath, "a") as f:
        f.write(msg + '\n')

'''
Clamp a number to a range.
'''
def clamp(n, minval, maxval):
    return min(max(n, minval), maxval)

'''
Normalize a number between 0-1.
'''
def normalize(n, minval, maxval):
    return float(n-minval)/float(maxval-minval)

'''
Use this for natural (human) sorting. Pass this as a key to a function that takes keys, such as sort.
:param s: The element that will be sorted
:param _nsre: Regular expression to find the digit portion of the element.
:author: https://stackoverflow.com/a/16090640/1002098
'''
RegexDigits = re.compile('([0-9]+)')
def naturalSortKey(s, _nsre=RegexDigits):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(_nsre, s)]

'''
Verify that a string is a valid date or datetime
:param datestr: String that is to be verified.
:param datefmtstr: Format datetime string (e.g. "%Y-%m-%d")
'''
def verifyDateTime(datestr, datefmtstr):
    try:
        datetime.strptime(datestr, datefmtstr)
        return True
    except ValueError:
        return False

'''
Helper function that returns a list of all files, directories, or both, immediate or recursive.
:param mode: 0=both, 1=files, 2=dir
:param recursive: Immediate top-level list or recursive list
:param ext: List of file extensions to filter by
'''
def findFiles(dirpath, mode=0, recursive=False, ext=[]):
    stuff = []
    if len(ext) > 0:
        for i in range(0, len(ext)):
            ext[i] = ext[i].strip().lower()
            if ext[i][0] != ".":
                ext[i] = "." + ext[i]
    # immediate top-level list
    if not recursive:
        for entry in os.listdir(dirpath):
            fullpath = os.path.join(dirpath, entry)
            if mode == 1 or mode == 0:
                base, extension = os.path.splitext(fullpath.strip().lower())
                if os.path.isfile(fullpath):
                    if len(ext) > 0:
                        for e in ext:
                            if extension == e:
                                stuff.append(fullpath)
                    else:
                        stuff.append(fullpath)
            if mode == 2 or mode == 0:
                if os.path.isdir(fullpath):
                    stuff.append(fullpath)
    # recursive list
    else:
        for root, dirs, files in os.walk(dirpath):
            if mode == 1 or mode == 0:
                for file in files:
                    fullpath = os.path.join(root, file)
                    base, extension = os.path.splitext(fullpath.strip().lower())
                    if len(ext) > 0:
                        for e in ext:
                            if extension == e:
                                stuff.append(fullpath)
                    else:
                        stuff.append(fullpath)
            if mode == 2 or mode == 0:
                for dir in dirs:
                    fullpath = os.path.join(root, dir)
                    stuff.append(fullpath)
    return stuff

'''
Helper function to delete all files and folders given a folder.
'''
def cleanFolder(dirpath):
    for filename in os.listdir(dirpath):
        filepath = os.path.join(dirpath, filename)
        try:
            if os.path.isfile(filepath):
                os.unlink(filepath)
            elif os.path.isdir(filepath):
                shutil.rmtree(filepath)
        except Exception as ex:
            print(ex.message)
