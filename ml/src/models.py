#!/usr/bin/python
# -*- coding: utf-8 -*-
# ====================================================================
# @authors: Joe Del Rocco, Brandon Patterson, Hassen Dhrif
# @since: 07/22/2018
# @summary: Main file working with scikit-learn ML models.
# ====================================================================
from enum import Enum
import math
import numpy as np
import pandas as pd
import sklearn.metrics as sm
#import sklearn.preprocessing as pp
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.externals import joblib
#local
import utility


Metric = Enum('Metric', 'MSD RMSD MBD MAD R2 Ratio')


def ComputeError(truth, preds, metric, percent=True):
    result = 0
    diff = preds - truth

    if metric == Metric.MSD:
        diff2 = np.square(diff)
        diff2sum = np.sum(diff2)
        result = diff2sum / len(truth)
    elif metric == Metric.RMSD:
        diff2 = np.square(diff)
        diff2sum = np.sum(diff2)
        result = math.sqrt(diff2sum / len(truth))
        # why does Tohsing include this?; for single wavelength??
        # rmsd_tohsing = (rmsd_ours / np.average(y_test)) * 100.0
    elif metric == Metric.MBD:
        diffsum = np.sum(diff)
        result = diffsum / len(truth)
    elif metric == Metric.MAD:
        delta = np.abs(diff)
        deltasum = np.sum(delta)
        result = deltasum / len(truth)
    elif metric == Metric.R2:
        result = sm.r2_score(truth, preds, multioutput = 'variance_weighted')
    elif metric == Metric.Ratio:
        result = truth / preds

    if percent:
        result = result * 100.0

    return result

def ProcessData(args, df, poly=0):
    # shuffle the dataset for better training
    df = df.sample(frac=1, random_state=args.seed)
    df = df.sample(frac=1, random_state=args.seed)
    df = df.sample(frac=1, random_state=args.seed)

    # engineered features (binning of capture timestamp)
    captures = df.Date + ' ' + df.Time
    captures = pd.to_datetime(captures)
    df.insert(0, 'hour', captures.dt.hour)
    df.insert(0, 'dayofyear', captures.dt.dayofyear)
    df.insert(0, 'weekofyear', captures.dt.weekofyear)
    df.insert(0, 'quarter', captures.dt.quarter)
    df.insert(0, 'month', captures.dt.month)
    args.observed = ['month', 'quarter', 'weekofyear', 'dayofyear', 'hour'] + args.observed

    # move captures of interest to the bottom (so they will be included in test set)
    for c in args.captures:
        df = df.reset_index(drop=True)
        captures = df.Date + ' ' + df.Time # needs to be recomputed each iteration
        captures = pd.to_datetime(captures)
        captures = captures.apply(lambda t: t.replace(second=0, microsecond=0, nanosecond=0))
        captures = pd.DataFrame(captures, columns=['timestamp'])
        indices = captures.index[captures['timestamp'] == c]
        a = df.iloc[[i for i in df.index if i not in indices], :]
        b = df.iloc[indices, :]
        df = pd.concat([a, b])
    df = df.reset_index(drop=True)

    # separate into input / output features
    X = df[args.observed]
    y = df.iloc[:, df.columns.get_loc(str(args.curvebeg)):]

    # generate polynomial features
    if poly > 0:
        polyfeatures = PolynomialFeatures(poly)
        polyfeatures = polyfeatures.fit(X)
        X = polyfeatures.transform(X)
        #X = polyfeatures.fit_transform(X.values, y)

    # apply principal component analysis
    pcaobj = None
    if args.pca > 0:
        pcaobj = PCA(n_components=args.pca)
        y = pcaobj.fit_transform(y)

    # make sure X and y are numpy arrays
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.DataFrame):
        y = y.values

    # split X and y into train and test sets
    # doing this manually ensures that df still lines up with X and y
    # ---------------------------------------------------------------
    #                (80%) X_train  y_train
    #  df = capture,       +      , +
    #                (20%) X_test   y_test
    # ---------------------------------------------------------------
    sidx = int(len(X) * 0.8)
    X_train, X_test = np.split(X, [sidx])
    y_train, y_test = np.split(y, [sidx])

    # add timestamp into df for sky capture lookup later (does not effect training)
    captures = df.Date + ' ' + df.Time
    captures = pd.to_datetime(captures)
    captures = captures.apply(lambda t: t.replace(second=0, microsecond=0, nanosecond=0))
    df.insert(0, 'capture', captures)

    return df, X_train, X_test, y_train, y_test, pcaobj

def CreateModel(args):
    # load dataset
    utility.Log(args, 'Loading dataset: ' + args.datasetpath)
    df = pd.read_csv(args.datasetpath)

    # data processing, move captures of interest, split training and testing sets
    df, X_train, X_test, y_train, y_test, pca = ProcessData(args, df, poly=args.poly)

    # create model
    model = None
    if args.model == 'etr':
        model = ExtraTreesRegressor(n_estimators=args.trees, max_depth=args.depth, min_samples_leaf=args.leaf,
                                    random_state=args.seed, n_jobs=args.numcores)
    elif args.model == 'rfr':
        model = RandomForestRegressor(n_estimators=args.trees, max_depth=args.depth, min_samples_leaf=args.leaf,
                                      random_state=args.seed, n_jobs=args.numcores)
    elif args.model == 'lnr':
        model = LinearRegression(n_jobs=args.numcores)
    elif args.model == 'knr':
        model = KNeighborsRegressor(n_jobs=args.numcores)

    # scale data
    scaler = None
    if args.model == 'knr' or args.model == 'lnr':
        scaler = QuantileTransformer()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # train model
    TrainModel(args, model, X_train, X_test, y_train, y_test)

    # save model and scaler(s)
    utility.Log(args, 'Saving model : ' + args.modelpath)
    joblib.dump(model, args.modelpath)
    if args.model == 'knr' or args.model == 'lnr':
        utility.Log(args, 'Saving scaler: ' + args.scalerpath)
        joblib.dump(scaler, args.scalerpath)

def TrainModel(args, model, X_train, X_test, y_train, y_test):
    # cross validation
    if args.kfold > 0:
        utility.Log(args, str(args.kfold) + "-fold cross validation")
        kf = KFold(args.kfold)
        train_r2s, test_r2s = [], []
        train_rmsds, test_rmsds = [], []
        for i, (train_set, test_set) in enumerate(kf.split(X_train)):
            utility.Log(args, 'Training fold: ' + str(i))
            model.fit(X_train[train_set], y_train[train_set])

            # compute k-fold cv train metrics
            #r2_train = model.score(X_train[train_set], y_train[train_set])
            preds = model.predict(X_train[train_set])
            r2 = ComputeError(y_train[train_set], preds, Metric.R2)
            rmsd = ComputeError(y_train[train_set], preds, Metric.RMSD)
            train_r2s.append(r2)
            train_rmsds.append(rmsd)

            # compute k-fold cv test metrics
            #r2_test = model.score(X_train[test_set], y_train[test_set])
            preds = model.predict(X_train[test_set])
            r2 = ComputeError(y_train[test_set], preds, Metric.R2)
            rmsd = ComputeError(y_train[test_set], preds, Metric.RMSD)
            test_r2s.append(r2)
            test_rmsds.append(rmsd)

        # report final cv metrics
        r2 = np.mean(train_r2s)
        rmsd = np.mean(train_rmsds)
        utility.Log(args, 'CV Train: ' + "{0:7.2f}%(R2) {1:7.2f}%(RMSD)".format(r2, rmsd))
        r2 = np.mean(test_r2s)
        rmsd = np.mean(test_rmsds)
        utility.Log(args, 'CV Test : ' + "{0:7.2f}%(R2) {1:7.2f}%(RMSD)".format(r2, rmsd))

    # fit model to all training data
    utility.Log(args, "Fitting model to all training data")
    model.fit(X_train, y_train)

    # test hold out only if user requests it!!
    if args.holdout:
        #r2 = model.score(X_test, y_test)
        preds = model.predict(X_test)
        r2 = ComputeError(y_test, preds, Metric.R2)
        rmsd = ComputeError(y_test, preds, Metric.RMSD)
        utility.Log(args, "Holdout : {0:7.2f}%(R2) {1:7.2f}%(RMSD)".format(r2, rmsd))

def TestModel(args):
    # load dataset
    utility.Log(args, 'Loading dataset: ' + args.datasetpath)
    df = pd.read_csv(args.datasetpath)

    # data processing, move captures of interest, split training and testing sets
    df, X_train, X_test, y_train, y_test, pca = ProcessData(args, df, poly=args.poly)

    # load model and scaler(s)
    utility.Log(args, 'Loading model: ' + args.modelpath)
    model = joblib.load(args.modelpath)
    if args.model == 'knr' or args.model == 'lnr':
        utility.Log(args, 'Loading scale: ' + args.scalerpath)
        scaler = joblib.load(args.scalerpath)
        X_test = scaler.transform(X_test)

    # model is now usable, grab some sample predictions
    utility.Log(args, "Predicting test holdout")
    preds = model.predict(X_test)

    # undo the effect of pca if necessary
    if args.pca > 0:
        preds = pca.inverse_transform(preds)
        y_test = pca.inverse_transform(y_test)

    # report metrics on entire test set
    r2 = ComputeError(y_test, preds, Metric.R2)
    rmsd = ComputeError(y_test, preds, Metric.RMSD)
    fmtstr = "*TestSet* {0:7.2f}%     {1:7.2f}%".format(r2, rmsd)
    utility.Log(args, fmtstr)
    errorfile = open(args.resultspath + ".txt", "w")
    errorfile.write(fmtstr + '\n')

    # report metrics for each capture specified
    for dt in args.captures:
        errorfile.write(("-" * 40) + '\n')
        errorfile.write("*Capture*   {0:s}".format(dt.strftime('%Y-%m-%d %H:%M')) + '\n')

        # actual rows of interest
        idxlbls = df.index[df['capture'] == dt]
        truthrows = df.loc[idxlbls, :]

        # predicted rows of interest
        tidx = np.where(df['capture'] == dt)[0]
        pidx = [i-len(y_train) for i in tidx]
        predsrows = preds[pidx, :]

        # for each sample the capture
        r2s, rmsds = [], []
        for smpidx in range(0, 81):
            # find by sample pattern index
            rowidx = np.where(truthrows['SamplePatternIndex'] == smpidx)[0]
            if len(rowidx) <= 0: continue
            tcurve = truthrows.iloc[rowidx, :]
            tcurve = tcurve.iloc[:, -preds.shape[1]:].values
            pcurve = predsrows[rowidx, :]

            # report metrics per sample
            r2 = ComputeError(tcurve.flatten(), pcurve.flatten(), Metric.R2)
            rmsd = ComputeError(tcurve, pcurve, Metric.RMSD)
            r2s.append(r2)
            rmsds.append(rmsd)
            errorfile.write("SIdx({0:02}): {1:7.2f}%(R2) {2:7.2f}%(RMSD)".format(smpidx, r2, rmsd) + '\n')

        # report metrics for entire capture
        r2 = np.mean(r2s)
        rmsd = np.mean(rmsds)
        utility.Log(args, "*Results* {0:7.2f}%(R2) {1:7.2f}%(RMSD) * {2:s}".format(r2, rmsd, dt.strftime('%Y-%m-%d %H:%M')))
        errorfile.write("*Results* {0:7.2f}%     {1:7.2f}%".format(r2, rmsd) + '\n')
    errorfile.close()

    # overwrite df with predictions
    df.iloc[-preds.shape[0]:, -preds.shape[1]:] = preds

    # save df with predictions to file
    utility.Log(args, 'Saving result: ' + args.resultspath)
    df.iloc[-preds.shape[0]:, :].to_csv(args.resultspath + ".csv", index=False)
