#!/usr/bin/python
# -*- coding: utf-8 -*-
# ====================================================================
# @authors: Joe Del Rocco
# @since: 11/06/2018
# @summary: script to merge all results .txt files
# ====================================================================
import os
import csv

Datasets = ['clear-tiff-rgb', 'clear-jpg-rgb']
#Datasets = ['clear-tiff-rgb', 'clear-tiff-hdr', 'clear-jpg-rgb', 'clear-jpg-rgb-exp6', 'clear-jpg-hdr']
#Datasets = ['clear-tiff-rgb', 'clear-tiff-rgb-5nm', 'clear-tiff-rgb-10nm', 'clear-tiff-rgb-15nm', 'clear-tiff-rgb-20nm']
#Datasets = ['clear-tiff-rgb', 'clear-tiff-hsv', 'clear-tiff-hsl', 'clear-tiff-lab']
#Models = ['ETR']
Models = ['ETR', 'RFR', 'KNR', 'LNR']


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

def main():
    # find relevant results .txt files
    files = findFiles('.', mode=1, ext=['txt'])
    files[:] = [f for f in files if '_' in f]  # filter any files without an underscore (e.g. readme.txt)
    for i in range(len(files)-1, -1, -1):      # filter non results files
        with open(files[i]) as f:
            line = f.readline()
        if not line.startswith('*TestSet*'):
            del files[i]

    # reorder files so we process them in a specific order
    filesnew = []
    for dataset in Datasets:
        for model in Models:
            name = model.lower() + '_' + dataset + '.txt'
            for f in files:
                if f.endswith(name):
                    filesnew.append(f)
                    break
    files[:] = filesnew

    # results csv collection file
    outfile = open('_results_.csv', mode='w')
    results = csv.writer(outfile, delimiter=',', lineterminator='\n')
    results.writerow(['date', 'time', 'dataset', 'model', 'r2', 'rmsd'])

    # go through files in results directory
    for file in files:
        name, ext = os.path.splitext(os.path.basename(file))
        model, dataset = name.split('_')
        with open(file) as f:
            content = f.readlines()
        for line in content:
            if not line.startswith('*'): continue
            if '*Capture*' in line:
                toks = line.split()
                date = toks[1].strip()
                time = toks[2].strip()
            elif '*Results*' in line:
                toks = line.split()
                r2 = toks[1].strip()
                rmsd = toks[2].strip()
                # flush to outfile
                results.writerow([date, time, dataset, model.upper(), r2, rmsd])

    outfile.close()


if __name__ == "__main__":
    main()
