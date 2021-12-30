#!/usr/bin/python
# -*- coding: utf-8 -*-
# ====================================================================
# @authors: Joe Del Rocco
# @since: 10/1/2019
# @summary: libRadtran comparison
# ====================================================================
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime


def PlotCurveCompare():
    # inputs
    curvebeg = 350
    curveend = 1780
    curvelen = curveend - curvebeg + 1
    curvelrt = [360, 371.75, 383.5, 395.25, 407, 418.75, 430.5, 442.25, 454, 465.75, 477.5, 489.25, 501, 512.75, 524.5, 536.25, 548, 559.75, 571.5, 583.25, 595, 606.75, 618.5, 630.25, 642, 653.75, 665.5, 677.25, 689, 700.75, 712.5, 724.25, 736, 747.75, 759.5, 771.25, 783, 794.75, 806.5, 818.25]
    XValues = list(range(curvelen))
    XValuesLRT = [round(i)-curvebeg for i in curvelrt]
    # graph options
    XTickStep = 200
    XTicks = list(range(0, curvelen+XTickStep-1, XTickStep))
    XLabels = [str(i + curvebeg) for i in XTicks]
    YTicks = [x / 100.0 for x in range(0, 17, 2)]  # list(range(0, 0.15, YTickStep))
    YLabels = [str(round(i,2)) for i in YTicks]
    YPadding = 0.01

    # load datasets
    df_measured = pd.read_csv("clear-tiff-rgb.csv")
    df_predicted = pd.read_csv("etr_predicted.csv")
    df_libradtran = pd.read_csv("lrt_predicted.csv")

    # # add capture timestamp to dataset truth data (preds already has it)
    # captures = dftruth.Date + ' ' + dftruth.Time
    # captures = pd.to_datetime(captures)
    # captures = captures.apply(lambda t: t.replace(second=0, microsecond=0, nanosecond=0)) # strip beyond minutes
    # dftruth.insert(0, 'capture', captures)
    # # make sure preds capture timestamps are datetime objects
    # dfpreds['capture'] = pd.to_datetime(dfpreds['capture'])

    # # actual curve rows
    # indices = dftruth.index[dftruth['capture'] == capture]
    # trow = dftruth.loc[indices, :]
    # trow = trow.sort_values(by=['SamplePatternIndex'])
    #
    # # predicted curve rows
    # indices = dfpreds.index[dfpreds['capture'] == capture]
    # prow = dfpreds.loc[indices, :]
    # prow = prow.sort_values(by=['SamplePatternIndex'])

    # # make sure they align properly
    # if not np.array_equal(trow['SamplePatternIndex'].values, prow['SamplePatternIndex']):
    #     utility.Log(args, "Error: actual and predicted rows for {0:s} don't align!".format(
    #         capture.strftime('%m/%d/%Y %H:%M')))
    #     return

    # # map to easily find curves by sample pattern index
    # smpmap = dict(zip(prow['SamplePatternIndex'], range(len(trow))))

    # extract curves as arrays
    rad_measured = df_measured.loc[:, str(curvebeg):str(curveend)].values
    rad_predicted = df_predicted.loc[:, str(curvebeg):str(curveend)].values
    rad_libradtran = df_libradtran.loc[:, "360":"818.25"].values

    # # compute max radiance value (for scaling)
    # maxrad = 0
    # maxrad = max(np.amax(np.amax(tcurves)), maxrad)
    # maxrad = max(np.amax(np.amax(pcurves)), maxrad)

    # plot
    for idx, row in df_measured.iterrows():
        fig, axs = plt.subplots()
        axs.set_title('$({0:.2f}\degree, {1:.2f}\degree)$ ({2:s})'.format(df_measured.loc[idx,'SampleAzimuth'], df_measured.loc[idx,'SampleAltitude'], df_measured.loc[idx,'capture']))
        axs.set_ylabel('Radiance ($W/m^2/sr/nm$)')
        #axs.set_ylim(0 - (maxrad * YPadding), maxrad + (maxrad * args.YPadding))
        axs.set_ylim(0-YPadding, YTicks[-1]+YPadding)
        plt.xticks(XTicks, XLabels)
        axs.set_xlabel('Wavelength ($nm$)')
        plt.yticks(YTicks, YLabels)
        axs.plot(XValues, rad_measured[idx], color='dimgray', label='measured', linestyle='--')
        axs.plot(XValues, rad_predicted[idx], color='#ffb000', label='ETR')
        axs.plot(XValuesLRT, rad_libradtran[idx], color='steelblue', label='libRadtran')
        plt.legend()
        capstr = pd.to_datetime(df_measured.loc[idx,'capture']).strftime("%m%d%H%M")        
        crdstr = "{0:d}_{1:d}".format(int(df_measured.loc[idx,'SampleAzimuth']), int(df_measured.loc[idx,'SampleAltitude']))
        #fig.savefig(capstr + "_" + crdstr + ".png", dpi=600, bbox_inches='tight')
        fig.savefig("results_lrt_" + capstr + "_" + crdstr + ".pdf", bbox_inches='tight')
        plt.close()
    return


def main():
    # configure matplotlib
    params = {'mathtext.default': 'regular',
              'legend.fontsize': 'x-large',
              'axes.labelsize': 'x-large',
              'axes.titlesize': 'xx-large',
              'xtick.labelsize': 'large',
              'ytick.labelsize': 'x-large'}
    plt.rcParams.update(params)

    # options
    # plt.style.use('seaborn-pastel')
    # plt.style.use('seaborn-deep')
    # plt.style.use('seaborn-colorblind')
    plt.style.use('seaborn-muted')

    # plot
    PlotCurveCompare()


if __name__ == "__main__":
    main()