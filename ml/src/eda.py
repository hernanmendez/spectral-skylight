#!/usr/bin/python
# -*- coding: utf-8 -*-
# ====================================================================
# @authors: Brandon Patterson
# @since: 07/23/2018
# @summary: Functions for generating eda
# ====================================================================
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.feature_selection import SelectKBest, f_regression
import os
#local
import models


FontScale = 1.5


def EDA(args):
    """ This function generates various graphs on a data set intended for EDA
    
    Arguments:
        args -- command line arguments
    """

    if not os.path.isdir(args.edapath):
        os.makedirs(args.edapath)

    # make paper vis presentable for paper
    sns.set_context('paper')
    sns.set_style('white')
    sns.set(font='serif', font_scale=FontScale)

    # prepare data
    df = pd.read_csv(args.datasetpath)
    df, X_train, X_test, y_train, y_test, pca = models.ProcessData(args, df)
    X = df[args.observed]
    y = df.iloc[:, -args.curvelen:]
    if not args.hdr:
        xlables = ['m','q','w','d','h','SuAz','SuAl','SmAz','SPA','ClA','ClB','ClC'] # 'SmAl',
    else:
        xlables = ['m','q','w','d','h','SuAz','SuAl','SmAz','SPA','Cl5A','Cl5B','Cl5C','Cl6A','Cl6B','Cl6C','Cl7A','Cl7B','Cl7C','Cl8A','Cl8B','Cl8C'] # 'SmAl',

    # show feature correlation
    fig, ax = plt.subplots()
    #fig.subplots_adjust(left=None, bottom=0.3, right=None, top=None, wspace=None, hspace=None)
    b = sns.heatmap(X.corr(), vmin=0, vmax=1)
    b.set_xticklabels(xlables, rotation=30, ha='right') #b.get_xticklabels()
    #b.set(title='Input Feature Correlation')
    #plt.savefig(os.path.join(args.edapath, 'correlation'), dpi=600, bbox_inches='tight')
    plt.savefig(os.path.join(args.edapath, 'correlation.pdf'), bbox_inches='tight')
    plt.close(fig)

    # show feature collinearity
    fig, ax = plt.subplots()
    b = sns.heatmap(X.corr(), vmin=0.9, vmax=1)
    #b.set(title='Input Feature Collinearity (> 0.9)')
    #fig.subplots_adjust(bottom=0.25, left=0.2)
    b.set_xticklabels(xlables, rotation=30, ha='right')
    #plt.savefig(os.path.join(args.edapath, 'collinearity'), dpi=600, bbox_inches='tight')
    plt.savefig(os.path.join(args.edapath, 'collinearity.pdf'), bbox_inches='tight')
    plt.close(fig)

    # look at outliers
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0.2)
    b = sns.boxplot(data=X, orient='h')
    b.set(title='Input Feature Outlier Detection')
    #plt.savefig(os.path.join(args.edapath, 'outliers'), dpi=600, bbox_inches='tight')
    plt.savefig(os.path.join(args.edapath, 'outliers.pdf'), bbox_inches='tight')
    plt.close(fig)

    # dist of input data
    sns.set(font='serif', font_scale=FontScale-0.5)
    if not args.hdr:
        f, axes = plt.subplots(3, 5, figsize=(16, 8))
    else:
        f, axes = plt.subplots(6, 5, figsize=(16, 16))
    plt.subplots_adjust(wspace=0.35, hspace=0.35)
    sns.distplot(df['month'], ax=axes[0, 0])
    sns.distplot(df['quarter'], ax=axes[0, 1])
    sns.distplot(df['weekofyear'], ax=axes[0, 2])
    sns.distplot(df['dayofyear'], ax=axes[0, 3])
    sns.distplot(df['hour'], ax=axes[0, 4])
    sns.distplot(df['SunAzimuth'], ax=axes[1, 0])
    sns.distplot(df['SunAltitude'], ax=axes[1, 1])
    sns.distplot(df['SampleAzimuth'], ax=axes[1, 2])
    #sns.distplot(df['SampleAltitude'], ax=axes[1, 3])
    sns.distplot(df['SunPointAngle'], ax=axes[1, 4])
    #sns.distplot(df['SkyCover'], ax=axes[2, 0])
    if not args.hdr:
        sns.distplot(df['ColorA'], ax=axes[2, 1])
        sns.distplot(df['ColorB'], ax=axes[2, 2])
        sns.distplot(df['ColorC'], ax=axes[2, 3])
    else:
        sns.distplot(df['ColorA5'], ax=axes[2, 1])
        sns.distplot(df['ColorB5'], ax=axes[2, 2])
        sns.distplot(df['ColorC5'], ax=axes[2, 3])
        sns.distplot(df['ColorA6'], ax=axes[3, 1])
        sns.distplot(df['ColorB6'], ax=axes[3, 2])
        sns.distplot(df['ColorC6'], ax=axes[3, 3])
        sns.distplot(df['ColorA7'], ax=axes[4, 1])
        sns.distplot(df['ColorB7'], ax=axes[4, 2])
        sns.distplot(df['ColorC7'], ax=axes[4, 3])
        sns.distplot(df['ColorA8'], ax=axes[5, 1])
        sns.distplot(df['ColorB8'], ax=axes[5, 2])
        sns.distplot(df['ColorC8'], ax=axes[5, 3])
    #plt.suptitle('Input Feature Distribution')
    #plt.savefig(os.path.join(args.edapath, 'distribution'), dpi=600, bbox_inches='tight')
    plt.savefig(os.path.join(args.edapath, 'distribution.pdf'), bbox_inches='tight')
    plt.close(fig)

    # determine feature importance
    sns.set(font='serif', font_scale=FontScale)
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0.2)
    scores = [0 for i in range(len(X.columns))]
    kselect = SelectKBest(score_func=f_regression)
    for col in y.columns:
        kselect.fit_transform(X, y[col])
        scores = scores + kselect.scores_
    scores = scores / len(y.columns)
    b = sns.barplot(y=X.columns.values, x=scores, orient='h', color='#326982')
    #ax.get_xaxis().get_major_formatter().set_scientific(False)
    b.set(title='Input Feature Importance', xlabel='Mean F-Score', xlim=(0,6000))
    b.axes.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{int(x/1e3)}K'))
    #plt.savefig(os.path.join(args.edapath, 'importance'), dpi=600, bbox_inches='tight')
    plt.savefig(os.path.join(args.edapath, 'importance.pdf'), bbox_inches='tight')
    plt.close(fig)
