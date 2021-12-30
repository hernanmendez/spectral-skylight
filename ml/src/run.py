#!/usr/bin/python
# -*- coding: utf-8 -*-
# ====================================================================
# @authors: Joe Del Rocco, Brandon Patterson
# @since: 07/16/2018
# @summary: Main run file for the work learning radiance curves.
# ====================================================================
import argparse
import os
import sys
import dateutil
import json
# local
import utility
import models
import plot
import eda


def loadConfig(args):
    if not os.path.exists(args.cfgpath):
        return False

    # load config file
    with open(args.cfgpath, 'r') as file:
        args.config = json.load(file)
    if not args.config or len(args.config) <= 0:
        return False

    # cache paths to important files
    ds_name, ds_ext = os.path.splitext(args.dataset)
    args.datasetpath = os.path.join(args.config["DataDir"], args.dataset + ".csv")
    args.modelpath = os.path.join(args.config["ModelDir"], args.model + "_" + ds_name + ".pkl")
    args.scalerpath = os.path.join(args.config["ModelDir"], args.model + "_scaler_" + ds_name + ".pkl")
    args.resultspath = os.path.join(args.config["ResultsDir"], args.model + "_" + ds_name)
    args.allresultspath = os.path.join(args.config["ResultsDir"], "_results_.csv")
    args.visualspath = os.path.join(args.config["VisualsDir"], args.model + "_" + ds_name)
    args.edapath = os.path.join(args.config["VisualsDir"], 'eda_' + ds_name)

    # features and curve
    args.observed = args.config["FeaturesHDR"] if args.hdr else args.config["Features"]
    args.curvebeg = args.config["SpectrumStart"]
    args.curveend = args.config["SpectrumEnd"]
    args.curvelen = args.curveend - args.curvebeg + 1

    # for plotting
    args.XValues = list(range(args.curvelen))
    args.XTicks = list(range(0, args.curvelen+args.config["XTickStep"]-1, args.config["XTickStep"]))
    args.XLabels = [str(i + args.curvebeg) for i in args.XTicks]
    args.YPadding = args.config["YPadding"]

    return True

def inputValidation(args):
    # required parameters
    if not os.path.exists(args.datasetpath):
        print("Error: Missing dataset: " + args.datasetpath)
        return False
    if args.action != "train" and args.action != "test" and args.action != "plot" and args.action != 'eda':
        print("Error: Unknown action: " + args.action)
        return False
    if args.model != "etr" and args.model != "rfr" and args.model != "knr" and args.model != "lnr":
        print("Error: Unknown model: " + args.model)
        return False
    if args.action == "test" and not os.path.exists(args.modelpath):
        print("Error: Missing model: " + args.modelpath)
        return False
    if args.action == "plot" and not os.path.exists(args.datasetpath):
        print("Error: Missing results: " + args.modelpath)
        return False

    # verify captures of interest specified by user
    strfmts = args.captures.split(",")
    args.captures = []
    for s in strfmts:
        if not s: continue
        try:
            dt = dateutil.parser.parse(s)
            dt = dt.replace(second=0, microsecond=0)  # ignore beyond minutes
            args.captures.append(dt)
        except ValueError:
            print("Error: Invalid capture timestamp format: " + s)
            return False

    return True

def main():
    # handle command line args
    parser = argparse.ArgumentParser(description='Framework for machine learning sky radiance sample datasets.', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_help = True
    parser.add_argument('model', help='model to use (etr, rfr, knr, lnr, etc.)')
    parser.add_argument('dataset', help='name of dataset (mix, clear, scattered, etc.)')
    parser.add_argument('action', help='what to do (train, test, plot, eda)')
    parser.add_argument('-s', '--seed', dest='seed', type=int, default=54321, help='prng seed for repeating results (def 54321)')
    parser.add_argument('-c', '--captures', dest='captures', type=str, default='', help='list of capture datetimes for testing (def none)')
    parser.add_argument('-p', '--pca', dest='pca', type=int, default=0, help='PCA components to use on OUTPUT (def 0)')
    parser.add_argument('-m', '--hdr', dest='hdr', action='store_true', help='dataset contains multiple exposures (def false)')
    # training specific
    parser.add_argument('-n', '--numcores', dest='numcores', type=int, default=1, help='number of cpu cores to use (def 1)')
    parser.add_argument('-k', '--kfold', dest='kfold', type=int, default=0, help='k-fold cross validation (def 0)')
    parser.add_argument('-t', '--trees', dest='trees', type=int, default=10, help='number of trees in forest (def 10)')
    parser.add_argument('-d', '--depth', dest='depth', type=int, default=None, help='max depth of tree (def None)')
    parser.add_argument('-l', '--leaf', dest='leaf', type=int, default=1, help='min samples per leaf (def 1)')
    parser.add_argument('-y', '--poly', dest='poly', type=int, default=0, help='polynomial components to use (def 0)')
    parser.add_argument('-!', '--holdout', dest='holdout', action='store_true', help='report hold out score upon train')
    args = parser.parse_args()

    # load config file, and validate input
    args.cfgpath = "../config.json"
    if not loadConfig(args): sys.exit(2)
    if not inputValidation(args): sys.exit(2)

    # start logging this job
    args.logpath = "../log.txt"
    utility.Log(args, "-"*80)
    utility.Log(args, sys.argv)

    # JUST! DO IT! - Shia Labeouf
    if args.action == 'train':
        models.CreateModel(args)
    elif args.action == 'test':
        models.TestModel(args)
    elif args.action == 'plot':
        plot.Plot(args)
    elif args.action == 'eda':
        eda.EDA(args)
    else:
        print("Error: Unknown action: " + args.action)
        sys.exit(2)


if __name__ == "__main__":
    main()
