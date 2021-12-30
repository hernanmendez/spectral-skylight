#!/usr/bin/python
# -*- coding: utf-8 -*-
# ====================================================================
# @authors: Joe Del Rocco
# @since: 09/09/2019
# @summary: Plotting color model results
# ====================================================================
import os
import matplotlib.pyplot as plt
import numpy as np

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
width = 0.2  # the width of the bars

captures = ["05/26/2013 15:15", "05/27/2013 10:15", "07/26/2013 13:15", "09/24/2013 15:39"]
renders = ["jpg-rgb", "jpg-rgb-6", "tiff-rgb", "jpg-hdr", "tiff-hdr"]
models = ["ETR"] #, "RFR", "KNR", "LNR"]
scores = [
    [[7.77, 8.72, 7.39, 9.11, 8.93]],
    [[5.33, 5.53, 4.02, 7.63, 7.55]],
    [[3.25, 3.82, 4.06, 4.01, 4.13]],
    [[4.20, 4.67, 4.21, 5.47, 4.61]]
]
# scores = [
#     [[7.39, 8.42, 7.75, 8.10], [8.52, 8.15, 8.12, 7.88], [10.62, 11.64, 11.27, 10.90], [23.64, 24.55, 24.43, 23.92]],
#     [[4.02, 5.22, 5.67, 6.11], [8.76, 9.31, 8.43, 9.37], [8.63, 8.84, 8.25, 9.28], [33.18, 33.51, 33.65, 33.05]],
#     [[4.06, 3.56, 3.53, 4.84], [4.62, 6.51, 5.96, 7.08], [3.55, 3.74, 3.64, 5.16], [14.22, 15.70, 15.44, 14.01]],
#     [[4.21, 4.58, 4.16, 5.49], [5.55, 5.11, 5.24, 5.39], [6.92, 7.80, 7.77, 6.61], [17.59, 17.65, 17.30, 17.99]],
# ]
index = np.arange(len(renders))


for idx, capture in enumerate(captures):
    fig, axs = plt.subplots(1, figsize=(6, 4.5))
    scores_per_model = [axs.bar(index, s, width)[0] for i, s in enumerate(scores[idx])]
    for i, s in enumerate(scores[idx][0]):
         #axs.text(index + width * i, s, str(s))
         axs.text(index[i], s + 1.0, str(s))
    axs.set_ylabel('RMSD (%)')
    axs.set_xticks(index)
    axs.set_xticklabels(renders)
    axs.set_ylim(0, 100)
    axs.legend(tuple(scores_per_model), tuple(models))
    axs.set_title(capture)
    capture = capture.replace('/', '').replace(':', '').replace(' ', '')
    plt.tight_layout()
    fig.savefig(capture + ".png", dpi=600, bbox_inches='tight')
    plt.close()