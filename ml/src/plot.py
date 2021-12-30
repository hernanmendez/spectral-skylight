#!/usr/bin/python
# -*- coding: utf-8 -*-
# ====================================================================
# @authors: Joe Del Rocco, Joe Kider
# @since: 07/21/2018
# @summary: Functions for plotting radiance curves and errors.
# ====================================================================
import os
import csv
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf
import numpy as np
import pandas as pd
# local
import utility
import models


def Plot(args):
    # configure matplotlib
    # 'figure.figsize': (15, 5),
    params = {'mathtext.default': 'regular',
              'legend.fontsize': 'x-large',
              'axes.labelsize': 'x-large',
              'axes.titlesize': 'xx-large',
              'xtick.labelsize': 'large',
              'ytick.labelsize': 'x-large'}
    plt.rcParams.update(params)
    #plt.rcParams.update(plt.rcParamsDefault)

    # plot bar charts for all results
    PlotFinalResults(args)

    # quick out if no specific captures to plot
    if len(args.captures) <= 0:
        return

    # load dataset and predictions
    dftruth = pd.read_csv(args.datasetpath)
    dfpreds = pd.read_csv(args.resultspath + ".csv")

    # add capture timestamp to dataset truth data (preds already has it)
    captures = dftruth.Date + ' ' + dftruth.Time
    captures = pd.to_datetime(captures)
    captures = captures.apply(lambda t: t.replace(second=0, microsecond=0, nanosecond=0)) # strip beyond minutes
    dftruth.insert(0, 'capture', captures)
    # make sure preds capture timestamps are datetime objects
    dfpreds['capture'] = pd.to_datetime(dfpreds['capture'])

    # plot for each capture specified
    for dt in args.captures:
        # all visuals for this capture will go into this directory
        vizdir = args.visualspath + "_" + dt.strftime('%Y%m%d%H%M')
        if os.path.isdir(vizdir):
            utility.cleanFolder(vizdir)
        else:
            os.makedirs(vizdir)

        # plot all dis stuff
        PlotAllCurves(args, dt, vizdir, dftruth, dfpreds)
        PlotAllCurvesAvg(args, dt, vizdir, dftruth, dfpreds)
        PlotAllCurvesRatio(args, dt, vizdir, dftruth, dfpreds)
        PlotWholeSkyError(args, dt, vizdir, dftruth, dfpreds)
        PlotSelectSamples(args, dt, vizdir, dftruth, dfpreds)
        PlotEachSampleCurve(args, dt, vizdir, dftruth, dfpreds)


def PlotAllCurves(args, capture, vizdir, dftruth, dfpreds):
    # actual curves
    indices = dftruth.index[dftruth['capture'] == capture]
    tcurves = dftruth.loc[indices, str(args.curvebeg):str(args.curveend)]

    # predicted curves
    indices = dfpreds.index[dfpreds['capture'] == capture]
    pcurves = dfpreds.loc[indices, str(args.curvebeg):str(args.curveend)]

    # compute max radiance value (for scaling)
    maxrad = 0
    maxrad = max(np.amax(np.amax(tcurves)), maxrad)
    maxrad = max(np.amax(np.amax(pcurves)), maxrad)

    # plot actual curves
    fig, axs = plt.subplots()
    axs.set_title('Measured (' + capture.strftime('%m/%d/%Y %H:%M') + ')')
    axs.set_xlabel('Wavelength ($nm$)')
    axs.set_ylabel('Radiance ($W/m^2/sr/nm$)')
    axs.set_ylim(0-(maxrad*args.YPadding), maxrad+(maxrad*args.YPadding))
    plt.xticks(args.XTicks, args.XLabels)
    for idx in range(len(tcurves)):
        curve = tcurves.iloc[idx, :]
        axs.plot(args.XValues, curve, color='dimgray', alpha=0.25)
    #fig.savefig(os.path.join(vizdir, "all_measured.png"), dpi=600, bbox_inches='tight')
    fig.savefig(os.path.join(vizdir, "all_measured.pdf"), bbox_inches='tight')
    plt.close()

    # plot predicted curves
    fig, axs = plt.subplots()
    axs.set_title('{0:s} Predicted ({1:s})'.format(args.model.upper(), capture.strftime('%m/%d/%Y %H:%M')))
    axs.set_xlabel('Wavelength ($nm$)')
    axs.set_ylabel('Radiance ($W/m^2/sr/nm$)')
    axs.set_ylim(0-(maxrad*args.YPadding), maxrad+(maxrad*args.YPadding))
    plt.xticks(args.XTicks, args.XLabels)
    for idx in range(len(pcurves)):
        curve = pcurves.iloc[idx, :]
        axs.plot(args.XValues, curve, color='orange', alpha=0.25)
    #fig.savefig(os.path.join(vizdir, "all_predicted.png"), dpi=600, bbox_inches='tight')
    fig.savefig(os.path.join(vizdir, "all_predicted.pdf"), bbox_inches='tight')
    plt.close()


def PlotAllCurvesAvg(args, capture, vizdir, dftruth, dfpreds):
    # actual curve rows
    indices = dftruth.index[dftruth['capture'] == capture]
    trow = dftruth.loc[indices, :]
    trow = trow.sort_values(by=['SamplePatternIndex'])

    # predicted curve rows
    indices = dfpreds.index[dfpreds['capture'] == capture]
    prow = dfpreds.loc[indices, :]
    prow = prow.sort_values(by=['SamplePatternIndex'])

    # make sure they align properly
    if not np.array_equal(trow['SamplePatternIndex'].values, prow['SamplePatternIndex']):
        utility.Log(args, "Error: actual and predicted rows for {0:s} don't align!".format(
            capture.strftime('%m/%d/%Y %H:%M')))
        return

    # map to easily find curves by sample pattern index
    smpmap = dict(zip(prow['SamplePatternIndex'], range(len(trow))))

    # extract curves as arrays
    tcurves = trow.loc[:, str(args.curvebeg):str(args.curveend)].values
    pcurves = prow.loc[:, str(args.curvebeg):str(args.curveend)].values

    # compute max radiance value (for scaling)
    maxrad = 0
    maxrad = max(np.amax(np.amax(tcurves)), maxrad)
    maxrad = max(np.amax(np.amax(pcurves)), maxrad)

    # averaged curves
    tcurvesavg = np.average(tcurves, axis=0)
    pcurvesavg = np.average(pcurves, axis=0)
    #tcurvesavg2 = np.mean(tcurves, axis=0) # should be same as above
    #pcurvesavg2 = np.mean(pcurves, axis=0)
    # variances
    tcurvesvar = np.average((tcurves - tcurvesavg) ** 2, axis=0)
    pcurvesvar = np.average((pcurves - pcurvesavg) ** 2, axis=0)
    # standard deviations
    tcurvesstd = np.sqrt(tcurvesvar)
    pcurvesstd = np.sqrt(pcurvesvar)
    #tcurvesstd2 = np.std(tcurves, axis=0) # should be same as above
    #pcurvesstd2 = np.std(pcurves, axis=0)
    # ranges
    tupper = tcurvesavg + tcurvesstd
    tlower = tcurvesavg - tcurvesstd
    pupper = pcurvesavg + pcurvesstd
    plower = pcurvesavg - pcurvesstd

    # # # deltas and weights
    # # deltas = np.abs(tcurves - pcurves)
    # # deltasums = np.sum(deltas, axis=0)
    # # w = deltas / deltasums
    # # w = 1.0 - w
    #
    # # deltas and weights
    # tdeltas = np.abs(tcurves - tcurvesavg)
    # tdeltasums = np.sum(tdeltas, axis=0)
    # tw = tdeltas / tdeltasums
    # tw = 1.0 - tw
    # pdeltas = np.abs(pcurves - pcurvesavg)
    # pdeltasums = np.sum(pdeltas, axis=0)
    # pw = pdeltas / pdeltasums
    # pw = 1.0 - pw
    #
    # # weighted averaged curves
    # tcurvesavg_w = np.average(tcurves, axis=0, weights=tw)
    # pcurvesavg_w = np.average(pcurves, axis=0, weights=pw)
    # # weighted variances
    # tcurvesvar_w = np.average((tcurves - tcurvesavg_w) ** 2, axis=0, weights=tw)
    # pcurvesvar_w = np.average((pcurves - pcurvesavg_w) ** 2, axis=0, weights=pw)
    # # weighted standard deviations
    # tcurvesstd_w = np.sqrt(tcurvesvar_w)
    # pcurvesstd_w = np.sqrt(pcurvesvar_w)
    # # weighted ranges
    # tupper_w = tcurvesavg_w + tcurvesstd_w
    # tlower_w = tcurvesavg_w - tcurvesstd_w
    # pupper_w = pcurvesavg_w + pcurvesstd_w
    # plower_w = pcurvesavg_w - pcurvesstd_w

    # plot
    fig, axs = plt.subplots()
    axs.clear()
    axs.set_title('{0:s} Mean Radiance ({1:s})'.format(args.model.upper(), capture.strftime('%m/%d/%Y %H:%M')))
    axs.set_ylabel('Radiance ($W/m^2/sr/nm$)')
    axs.set_ylim(0 - (maxrad * args.YPadding), maxrad + (maxrad * args.YPadding))
    axs.set_xlabel('Wavelength ($nm$)')
    plt.xticks(args.XTicks, args.XLabels)
    for idx in range(len(tcurves)):
        curve = tcurves[idx, :]
        axs.plot(args.XValues, curve, color='dimgray', alpha=0.0075)
    for idx in range(len(pcurves)):
        curve = pcurves[idx, :]
        axs.plot(args.XValues, curve, color='orange', alpha=0.0075)
    axs.plot(args.XValues, tcurvesavg, color='dimgray', label='$\overline{measured}$')
    axs.plot(args.XValues, pcurvesavg, color='orange', label='$\overline{predicted}$')
    plt.legend()
    #fig.savefig(os.path.join(vizdir, "all_avg.png"), dpi=600, bbox_inches='tight')
    fig.savefig(os.path.join(vizdir, "all_avg.pdf"), bbox_inches='tight')
    plt.close()

    # plot
    fig, axs = plt.subplots()
    axs.clear()
    axs.set_title('{0:s} SD from Mean ({1:s})'.format(args.model.upper(), capture.strftime('%m/%d/%Y %H:%M')))
    axs.set_ylabel('Radiance ($W/m^2/sr/nm$)')
    axs.set_ylim(0 - (maxrad * args.YPadding), maxrad + (maxrad * args.YPadding))
    axs.set_xlabel('Wavelength ($nm$)')
    plt.xticks(args.XTicks, args.XLabels)
    axs.fill_between(args.XValues, tupper, tlower, facecolor='darkgray', alpha=0.5, label='$measured\quad\sigma$')
    axs.fill_between(args.XValues, pupper, plower, facecolor='orange', alpha=0.5, label='$predicted\quad\sigma$')
    axs.plot(args.XValues, tupper, color='darkgray', lw=0.5, alpha=0.75)
    axs.plot(args.XValues, tlower, color='darkgray', lw=0.5, alpha=0.75)
    axs.plot(args.XValues, pupper, color='orange', lw=0.5, alpha=0.75)
    axs.plot(args.XValues, plower, color='orange', lw=0.5, alpha=0.75)
    plt.legend()
    #fig.savefig(os.path.join(vizdir, "all_sd.png"), dpi=600, bbox_inches='tight')
    fig.savefig(os.path.join(vizdir, "all_sd.pdf"), bbox_inches='tight')
    plt.close()

    # plot
    fig, axs = plt.subplots()
    axs.clear()
    axs.set_title('{0:s} Measured vs Predicted ({1:s})'.format(args.model.upper(), capture.strftime('%m/%d/%Y %H:%M')))
    axs.set_ylabel('Radiance ($W/m^2/sr/nm$)')
    axs.set_ylim(0 - (maxrad * args.YPadding), maxrad + (maxrad * args.YPadding))
    axs.set_xlabel('Wavelength ($nm$)')
    plt.xticks(args.XTicks, args.XLabels)
    for idx in range(len(tcurves)):
        curve = tcurves[idx, :]
        axs.plot(args.XValues, curve, color='dimgray', alpha=0.0075)
    for idx in range(len(pcurves)):
        curve = pcurves[idx, :]
        axs.plot(args.XValues, curve, color='orange', alpha=0.0075)
    axs.fill_between(args.XValues, tupper, tlower, facecolor='darkgray', alpha=0.5, label='$measured\quad\sigma$')
    axs.fill_between(args.XValues, pupper, plower, facecolor='orange', alpha=0.5, label='$predicted\quad\sigma$')
    axs.plot(args.XValues, tupper, color='darkgray', lw=0.5, alpha=0.75)
    axs.plot(args.XValues, tlower, color='darkgray', lw=0.5, alpha=0.75)
    axs.plot(args.XValues, pupper, color='orange', lw=0.5, alpha=0.75)
    axs.plot(args.XValues, plower, color='orange', lw=0.5, alpha=0.75)
    axs.plot(args.XValues, tcurvesavg, color='dimgray', label='$\overline{measured}$', ls='--')
    axs.plot(args.XValues, pcurvesavg, color='darkorange', label='$\overline{predicted}$', ls='--')
    plt.legend()
    #fig.savefig(os.path.join(vizdir, "all_combined.png"), dpi=600, bbox_inches='tight')
    fig.savefig(os.path.join(vizdir, "all_combined.pdf"), bbox_inches='tight')
    plt.close()


def PlotAllCurvesRatio(args, capture, vizdir, dftruth, dfpreds):
    # actual curve rows
    indices = dftruth.index[dftruth['capture'] == capture]
    trow = dftruth.loc[indices, :]
    trow = trow.sort_values(by=['SamplePatternIndex'])

    # predicted curve rows
    indices = dfpreds.index[dfpreds['capture'] == capture]
    prow = dfpreds.loc[indices, :]
    prow = prow.sort_values(by=['SamplePatternIndex'])

    # make sure they align properly
    if not np.array_equal(trow['SamplePatternIndex'].values, prow['SamplePatternIndex']):
        utility.Log(args, "Error: actual and predicted rows for {0:s} don't align!".format(
            capture.strftime('%m/%d/%Y %H:%M')))
        return

    # map to easily find curves by sample pattern index
    smpmap = dict(zip(prow['SamplePatternIndex'], range(len(trow))))

    # extract curves as arrays
    tcurves = trow.loc[:, str(args.curvebeg):str(args.curveend)].values
    pcurves = prow.loc[:, str(args.curvebeg):str(args.curveend)].values

    # averaged curves
    tcurvesavg = np.average(tcurves, axis=0)
    pcurvesavg = np.average(pcurves, axis=0)

    # ratio
    curveratio = models.ComputeError(tcurvesavg, pcurvesavg, models.Metric.Ratio, percent=False)

    # # deltas and weights
    # tdeltas = np.abs(tcurves - tcurvesavg)
    # tdeltasums = np.sum(tdeltas, axis=0)
    # tw = tdeltas / tdeltasums
    # tw = 1.0 - tw
    # pdeltas = np.abs(pcurves - pcurvesavg)
    # pdeltasums = np.sum(pdeltas, axis=0)
    # pw = pdeltas / pdeltasums
    # pw = 1.0 - pw
    #
    # # weighted averaged curves
    # tcurvesavg_w = np.average(tcurves, axis=0, weights=tw)
    # pcurvesavg_w = np.average(pcurves, axis=0, weights=pw)
    #
    # # weighted ratio
    # curveratio_w = models.ComputeError(tcurvesavg_w, pcurvesavg_w, models.Metric.Ratio, percent=False)

    # plot
    fig, axs = plt.subplots()
    axs.clear()
    axs.set_title('{0:s} Pred. / Meas. ({1:s})'.format(args.model.upper(), capture.strftime('%m/%d/%Y %H:%M')))
    axs.set_ylabel('Radiance ($W/m^2/sr/nm$)')
    axs.set_ylim(0.0, 2.0)
    axs.set_xlabel('Wavelength ($nm$)')
    plt.xticks(args.XTicks, args.XLabels)
    axs.plot(args.XValues, curveratio, color='black', label='ratio')
    #axs.plot(args.XValues, curveratio_w, color='orange', label='ratio')
    plt.legend()
    #fig.savefig(os.path.join(vizdir, "all_avg_ratio.png"), dpi=600, bbox_inches='tight')
    fig.savefig(os.path.join(vizdir, "all_avg_ratio.pdf"), bbox_inches='tight')
    plt.close()


def PlotSelectSamples(args, capture, vizdir, dftruth, dfpreds):
    # nothing to plot?
    if len(args.config["Samples2Plot"]["Samples"])<=0:
        return

    # actual curve rows
    indices = dftruth.index[dftruth['capture'] == capture]
    trow = dftruth.loc[indices, :]
    trow = trow.sort_values(by=['SamplePatternIndex'])

    # predicted curve rows
    indices = dfpreds.index[dfpreds['capture'] == capture]
    prow = dfpreds.loc[indices, :]
    prow = prow.sort_values(by=['SamplePatternIndex'])

    # make sure they align properly
    if not np.array_equal(trow['SamplePatternIndex'].values, prow['SamplePatternIndex']):
        utility.Log(args, "Error: actual and predicted rows for {0:s} don't align!".format(
            capture.strftime('%m/%d/%Y %H:%M')))
        return

    # map to easily find curves by sample pattern index
    smpmap = dict(zip(prow['SamplePatternIndex'], range(len(trow))))

    # extract curves as arrays
    tcurves = trow.loc[:, str(args.curvebeg):str(args.curveend)].values
    pcurves = prow.loc[:, str(args.curvebeg):str(args.curveend)].values

    # compute max radiance value (for scaling)
    maxrad = 0
    maxrad = max(np.amax(np.amax(tcurves)), maxrad)
    maxrad = max(np.amax(np.amax(pcurves)), maxrad)

    # plot each sample (actual and predicted) to the same file
    fig, axs = plt.subplots()
    axs.set_title('Select Samples ({0:s})'.format(capture.strftime('%m/%d/%Y %H:%M')))
    axs.set_ylabel('Radiance ($W/m^2/sr/nm$)')
    axs.set_ylim(0 - (maxrad * args.YPadding), maxrad + (maxrad * args.YPadding))
    axs.set_xlabel('Wavelength ($nm$)')
    plt.xticks(args.XTicks, args.XLabels)
    for i in range(len(args.config["Samples2Plot"]["Samples"])):
        smpidx = args.config["Samples2Plot"]["Samples"][i]
        tcurve = tcurves[smpmap[smpidx], :]
        pcurve = pcurves[smpmap[smpidx], :]
        axs.plot(args.XValues, tcurve, color=args.config["Samples2Plot"]["ColorTruth"][i], label='# '+str(smpidx))
        axs.plot(args.XValues, pcurve, color=args.config["Samples2Plot"]["ColorPredicted"][i], label='# '+str(smpidx)+' Predicted')
    plt.legend()
    fig.savefig(os.path.join(vizdir, "few_samples.png"), dpi=600, bbox_inches='tight')
    plt.close()


def PlotEachSampleCurve(args, capture, vizdir, dftruth, dfpreds):
    # actual curve rows
    indices = dftruth.index[dftruth['capture'] == capture]
    trow = dftruth.loc[indices, :]
    trow = trow.sort_values(by=['SamplePatternIndex'])

    # predicted curve rows
    indices = dfpreds.index[dfpreds['capture'] == capture]
    prow = dfpreds.loc[indices, :]
    prow = prow.sort_values(by=['SamplePatternIndex'])

    # make sure they align properly
    if not np.array_equal(trow['SamplePatternIndex'].values, prow['SamplePatternIndex']):
        utility.Log(args, "Error: actual and predicted rows for {0:s} don't align!".format(
            capture.strftime('%m/%d/%Y %H:%M')))
        return

    # map to easily find curves by sample pattern index
    smpmap = dict(zip(prow['SamplePatternIndex'], range(len(trow))))

    # extract curves as arrays
    tcurves = trow.loc[:, str(args.curvebeg):str(args.curveend)].values
    pcurves = prow.loc[:, str(args.curvebeg):str(args.curveend)].values

    # compute max radiance value (for scaling)
    maxrad = 0
    maxrad = max(np.amax(np.amax(tcurves)), maxrad)
    maxrad = max(np.amax(np.amax(pcurves)), maxrad)

    # plot each sample (actual and predicted) to its own file
    fig, axs = plt.subplots()
    for smpidx,idx in smpmap.items():
        tcurve = tcurves[idx, :]
        pcurve = pcurves[idx, :]

        # plot
        axs.clear()
        axs.set_title('Sample {0:02d} $({1:.2f}\degree, {2:.2f}\degree)$ ({3:s})'.format(smpidx, args.config["SamplingPattern"][smpidx][0], args.config["SamplingPattern"][smpidx][1], capture.strftime('%m/%d/%Y %H:%M')))
        axs.set_ylabel('Radiance ($W/m^2/sr/nm$)')
        axs.set_ylim(0 - (maxrad * args.YPadding), maxrad + (maxrad * args.YPadding))
        axs.set_xlabel('Wavelength ($nm$)')
        plt.xticks(args.XTicks, args.XLabels)
        axs.plot(args.XValues, tcurve, color='dimgray', label='measured')
        axs.plot(args.XValues, pcurve, color='orange', label='predicted')
        plt.legend()
        #fig.savefig(os.path.join(vizdir, "sample_{0:02d}.png".format(smpidx)), dpi=600, bbox_inches='tight')
        fig.savefig(os.path.join(vizdir, "sample_{0:02d}.pdf".format(smpidx)), dpi=600, bbox_inches='tight')
    plt.close()


def PlotWholeSkyError(args, capture, vizdir, dftruth, dfpreds):
    # actual curve rows
    indices = dftruth.index[dftruth['capture'] == capture]
    trow = dftruth.loc[indices, :]
    trow = trow.sort_values(by=['SamplePatternIndex'])

    # predicted curve rows
    indices = dfpreds.index[dfpreds['capture'] == capture]
    prow = dfpreds.loc[indices, :]
    prow = prow.sort_values(by=['SamplePatternIndex'])

    # make sure they align properly
    if not np.array_equal(trow['SamplePatternIndex'].values, prow['SamplePatternIndex']):
        utility.Log(args, "Error: actual and predicted rows for {0:s} don't align!".format(
            capture.strftime('%m/%d/%Y %H:%M')))
        return

    # map to easily find curves by sample pattern index
    smpmap = dict(zip(prow['SamplePatternIndex'], range(len(trow))))

    # extract curves as arrays
    tcurves = trow.loc[:, str(args.curvebeg):str(args.curveend)].values
    pcurves = prow.loc[:, str(args.curvebeg):str(args.curveend)].values

    # numpy arrays of: azimuth, altitude, error
    azicoords = np.empty(0, dtype=float)
    altcoords = np.empty(0, dtype=float)
    errvalues = np.empty(0, dtype=float)
    for smpidx,idx in smpmap.items():
        tcurve = tcurves[idx, :].reshape(1,-1)
        pcurve = pcurves[idx, :].reshape(1,-1)
        azicoords = np.append(azicoords, args.config["SamplingPattern"][smpidx][0])
        altcoords = np.append(altcoords, args.config["SamplingPattern"][smpidx][1])
        errvalues = np.append(errvalues, models.ComputeError(tcurve, pcurve, models.Metric.RMSD))

    # adjustments to coordinates
    rmax = 90
    radii = (90.0 - altcoords)     # flip altitude back to zenith because polar requires a radius
    thetas = np.radians(azicoords)
    xs = radii * np.cos(thetas)
    ys = radii * np.sin(thetas)
    zs = errvalues
    ri, ti = np.mgrid[0:rmax:100j, 0:2 * np.pi:100j]
    rbf = Rbf(xs, ys, zs, function='linear')
    zi = rbf(ri * np.cos(ti), ri * np.sin(ti))

    # plot RMSD
    #cc = mcolors.ColorConverter().to_rgb
    #wbr = makeColormap([cc('light blue'), cc('dark blue'), 0.5, cc('dark blue'), cc('red')])
    fig = plt.figure()
    ax = plt.subplot(111, polar=True)
    ax.set_theta_zero_location("S")
    ax.set_theta_direction(1)
    ax.set_rmax(rmax)
    #ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.yaxis.set_major_formatter(plt.NullFormatter())
    ax.grid(color='w', alpha=0.25, linewidth=0.25)
    cax = ax.contour(ti, ri, zi, 10, linewidths=0.5, colors='k')
    cax = ax.contourf(ti, ri, zi, 10, cmap=plt.cm.RdBu_r)
    # # plot azimuth labels, with a North label.
    # gridX, gridY = 10.0, 45.0
    # ax = plt.gca()
    # ax.text(0.5, 1.025, 'N', transform=ax.transAxes, horizontalalignment='center', verticalalignment='bottom', size=25)
    # for para in np.arange(gridY, 360, gridY):
    #     x = (1.1 * 0.5 * np.sin(np.deg2rad(para))) + 0.5
    #     y = (1.1 * 0.5 * np.cos(np.deg2rad(para))) + 0.5
    #     ax.text(x, y, u'%i\N{DEGREE SIGN}' % para, transform=ax.transAxes, horizontalalignment='center', verticalalignment='center')
    # plot sun
    # ax.plot(angles[:,0],90-angles[:,1] , 'ro', markersize=8,  c='yellow')
    # ax.plot(219.86-45,(90-19.25) , 'ro', markersize=8,  c='yellow')
    cb = fig.colorbar(cax, pad=0.1)
    cb.set_label("RMSD%")
    fig.savefig(os.path.join(vizdir, "rmsd_wholesky.pdf"), bbox_inches='tight')
    plt.close(fig)

    # normalize RMSD
    zi = (100 * (zi - np.min(zi)) / np.ptp(zi))

    # plot normalized RMSD
    fig = plt.figure()
    ax = plt.subplot(111, polar=True)
    ax.set_theta_zero_location("S")
    ax.set_theta_direction(1)
    ax.set_rmax(rmax)
    ax.yaxis.set_major_formatter(plt.NullFormatter())
    ax.grid(color='w', alpha=0.25, linewidth=0.25)
    cax = ax.contour(ti, ri, zi, 10, linewidths=0.5, colors='k')
    cax = ax.contourf(ti, ri, zi, 10, cmap=plt.cm.RdBu_r)
    cb = fig.colorbar(cax, pad=0.1)
    cb.set_label("RMSD% Normalized")
    fig.savefig(os.path.join(vizdir, "rmsd_wholesky_normalized.pdf"), bbox_inches='tight')
    plt.close(fig)


def PlotFinalResults(args):
    # nothing to plot?
    if not os.path.exists(args.allresultspath):
        return

    # options
    #plt.style.use('fivethirtyeight')
    #plt.style.use('seaborn-pastel')
    plt.style.use('seaborn-muted')
    #plt.style.use('seaborn-deep')
    #plt.style.use('seaborn-colorblind')
    colors = ['#ffb000', 'springgreen', 'xkcd:light salmon', 'steelblue'] #'xkcd:lightish green'
    clear = mpatches.Patch(color='whitesmoke', label='clear')
    scattered = mpatches.Patch(color='lightgray', label='scattered')
    overcast = mpatches.Patch(color='darkgray', label='overcast')
    rot = 10  # rotation of x label dates
    width = 0.2  # the width of the bars

    curr_dataset = None
    curr_model = None
    dt_done = False
    models = []
    captures = []
    model_scores = []

    # read through final results file
    with open(args.allresultspath, 'r') as infile:
        reader = csv.reader(infile, delimiter=',')
        next(reader, None)  # skip header
        for row in reader:

            # found another dataset (plot previous and move on)
            if row[2] != curr_dataset:
                # plot previous dataset first
                if curr_dataset:
                    index = np.arange(len(captures))  # the x locations for the bins
                    fig, axs = plt.subplots(1, figsize=(6,4.5))
                    if curr_dataset == 'mix':
                        fig.set_figwidth(12)
                        axs.axvspan(-0.2, 3.79, color='whitesmoke')
                        axs.axvspan(3.8, 7.8, color='lightgray', alpha=0.8)
                        axs.axvspan(7.81, 10.8, color='darkgray', alpha=0.75)
                    scores_per_capture = [ axs.bar(index + width * i, scores, width, color=colors[i], edgecolor='#444444', linewidth=1)[0] for i, scores in enumerate(model_scores) ]
                    if curr_dataset == 'mix':
                        scores_per_capture.extend([clear, scattered, overcast])
                        models.extend(['clear', 'scattered', 'overcast'])
                    axs.set_ylabel('RMSD (%)')
                    axs.set_xticks(index + width)
                    axs.set_xticklabels(captures, rotation=rot)
                    axs.set_ylim(0, 100)
                    if curr_dataset == 'mix':
                        axs.set_xlim(-0.25, 10.85)
                    axs.legend(tuple(scores_per_capture), tuple(models))
                    axs.set_title(curr_dataset)
                    plt.tight_layout()
                    #fig.savefig(os.path.join(args.visualspath, '..', curr_dataset + ".png"), dpi=600, bbox_inches='tight')
                    fig.savefig(os.path.join(args.visualspath, '..', curr_dataset + ".pdf"), bbox_inches='tight')
                    plt.close()
                # setup next dataset
                curr_dataset = row[2]
                curr_model = None
                models = []
                captures = []
                model_scores = []
                dt_done = False

            # found another model
            if row[3] != curr_model:
                # stop collecting capture dates
                if curr_model:
                    dt_done = True
                # collect new model
                curr_model = row[3]
                models.append(curr_model)
                model_scores.append([])

            # collect captures (just once per dataset)
            if not dt_done:
                dt = row[0] + " " + row[1]
                captures.append(dt)

            # always collect result scores
            model_scores[-1].append(float(row[5].strip('%')))

    # plot final dataset
    index = np.arange(len(captures))  # the x locations for the bins
    fig, axs = plt.subplots(1, figsize=(6, 4.5))
    scores_per_capture = [axs.bar(index + width * i, scores, width, color=colors[i], edgecolor='#444444', linewidth=1)[0] for i, scores in enumerate(model_scores)]
    axs.set_ylabel('RMSD (%)')
    axs.set_xticks(index + width)
    axs.set_xticklabels(captures, rotation=rot)
    axs.set_ylim(0, 100)
    axs.legend(tuple(scores_per_capture), tuple(models))
    axs.set_title(curr_dataset)
    plt.tight_layout()
    #fig.savefig(os.path.join(args.visualspath, '..', curr_dataset + ".png"), dpi=600, bbox_inches='tight')
    fig.savefig(os.path.join(args.visualspath, '..', curr_dataset + ".pdf"), bbox_inches='tight')
    plt.close()