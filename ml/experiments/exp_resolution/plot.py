#!/usr/bin/python
# -*- coding: utf-8 -*-
# ====================================================================
# @authors: Joe Del Rocco
# @since: 09/10/2019
# @summary: Plotting spectrum resolution results
# ====================================================================
import matplotlib.pyplot as plt


def PlotResults():
    width = 0.85  # the width of the bars

    # data
    rez = [1, 5, 10, 15, 20]
    ttrain = [34.91032934, 11.08991918, 8.84946592, 8.20906384, 7.86744054]
    ttest = [14.36979382, 6.51071808, 5.17482766, 4.9317502, 4.74232798]
    #szdata = [99060736, 20192256, 10332160, 7025664, 5373952]
    szmodel = [3648302080, 747172864, 384746496, 262155264, 202105856]
    rmsd = [7.39, 3.58, 2.42, 2.08, 1.87]

    # unit conversion
    #szdata = [i / 1024 / 1024 / 1024 for i in szdata]
    szmodel = [i / 1024 / 1024 / 1024 for i in szmodel]

    # plot
    fig, ax1 = plt.subplots(1, figsize=(7, 4.5))
    # rmsd
    ax1.set_title("ETR 05/26/2013 15:15")
    ax1.set_xticks(rez)
    ax1.set_xlabel('spectrum resolution (nm)')
    ax1.set_ylabel('RMSD (%)')
    ax1.set_ylim(0, 100)
    ax1.bar(rez, rmsd, width, label="RMSD (%)", color='#ffb000', edgecolor='black', zorder=1) # color='None'
    # storage
    ax2 = ax1.twinx()
    ax2.set_ylabel('size (GB)')
    ax2.set_ylim(0, 4)
    ax2.plot(rez, szmodel, label="storage", color='tab:red', marker='s', markersize=11, fillstyle='full')
    #ax2.plot(rez, szdata, label="dataset", color='tab:red', marker='s', markersize=11, fillstyle='none')
    # timings
    ax3 = ax1.twinx()
    ax3.set_ylabel('time (s)')
    ax3.set_ylim(0, 60)
    ax3.plot(rez, ttrain, label="training", color='tab:green', marker='o', markersize=11, fillstyle='none')
    ax3.plot(rez, ttest, label="testing", color='tab:green', marker='o', markersize=11, fillstyle='full')

    # y-axes config
    #ax1.yaxis.label.set_color('tab:blue')
    #ax1.tick_params(axis='y', colors='tab:blue')
    ax2.yaxis.label.set_color('tab:red')
    ax2.tick_params(axis='y', colors='tab:red')
    ax2.spines["right"].set_position(("axes", 1.2))
    ax3.yaxis.label.set_color('forestgreen')
    ax3.tick_params(axis='y', colors='forestgreen')

    # legend
    data1, labels1 = ax1.get_legend_handles_labels()
    data2, labels2 = ax2.get_legend_handles_labels()
    data3, labels3 = ax3.get_legend_handles_labels()
    ax3.legend(data1 + data2 + data3, labels1 + labels2 + labels3) # , loc=0

    # save
    plt.tight_layout()
    #fig.savefig("resolution.png", dpi=600, bbox_inches='tight')
    fig.savefig("resolution.pdf", bbox_inches='tight')
    plt.close()


def main():
    # configure matplotlib
    params = {'legend.fontsize': 'x-large',
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

    PlotResults()


if __name__ == "__main__":
    main()