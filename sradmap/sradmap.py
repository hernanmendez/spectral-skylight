#!/usr/bin/python
# -*- coding: utf-8 -*-
# ====================================================================
# @authors: Joe Del Rocco
# @since: 10/11/2018
# @summary: Script to predict spectral radiance curves for every pixel
# in a fisheye sky photo using a machine learned regression model.
# ====================================================================
# %%
import argparse
import os
import sys
import math
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import PolynomialFeatures
from colormath.color_objects import SpectralColor, sRGBColor
from colormath.color_conversions import convert_color
# local
import spa


'''
Function to log to file and standard out with one call.
'''
def log(args, msg):
    if not args.log: return
    msg = str(msg)
    print(msg)
    with open(args.logpath, "a") as f:
        f.write(msg + '\n')

'''
Convert a sky coordinate (azimuth, altitude) to fisheye UV coordinate (0-1, 0-1).
Note that images in this application were taken with North facing downward, so we must account for this in UV.
Note sampling pattern coordinates in this application were measured in altitude, but calculation below requires zenith.
Note altering of zenith to account for warp of lens used:
http://paulbourke.net/dome/fisheyecorrect/
'''
def SkyCoord2FisheyeUV(args, azimuth, altitude):
    # rotate azimuth so that position of North is pointing directly down
    azimuth = 360 - ((azimuth + 270) % 360)

    # convert altitude to zenith
    zenith = (90 - altitude)

    # convert from angles to radians
    azimuth = azimuth * math.pi / 180.0
    zenith = zenith * math.pi / 180.0

    # compute radius
    # account for non-linearity/warp of actual lens
    radius = np.polyval(args.lenswarp, zenith)

    # compute UVs
    u = radius * math.cos(azimuth)
    v = radius * math.sin(azimuth)

    # adjust and limit to [0, 1] range
    u = 0.5 * u + 0.5
    v = 0.5 * v + 0.5

    return u, v

'''
Convert a fisheye UV coordinate (0-1, 0-1) to a sky coordinate (azimuth, altitude).
'''
def FisheyeUV2SkyCoord(args, u, v):
    # adjust to [-1, 1] range
    u = (u - 0.5) * 2
    v = (v - 0.5) * 2

    radius = math.sqrt((u * u) + (v * v))

    # compute azimuth
    azimuth = math.atan2(u, v)
    # rotate azimuth so that position of North is pointing directly down
    azimuth = (azimuth + 2*math.pi) % (2*math.pi)

    # compute zenith
    # account for non-linearity/warp of actual lens
    zenith = np.polyval(args.lensinv, radius)
    # convert zenith to altitude
    altitude = (math.pi / 2) - zenith

    # convert from radians to angles
    azimuth = azimuth * 180.0 / math.pi
    altitude = altitude * 180.0 / math.pi

    return azimuth, altitude

'''
Take in a pair of (azimuth, altitude) sky coordintes (in radians) and returns the central angle between them.
:note: Assumes angles are in radians (for efficiency).
https://en.wikipedia.org/wiki/Great-circle_distance#Formulas
'''
def CentralAngle(a, b):
    return math.acos( math.sin(a[1]) * math.sin(b[1]) + math.cos(a[1]) * math.cos(b[1]) * math.cos( abs(a[0]-b[0]) ) )

'''
Function to compute the (azimuth, altitude) position of the sun using NREL SPA.
:param spadata: spa_data object with site info and date
:note: NREL SPA can be found at https://midcdmz.nrel.gov/spa/
:return: A single (azimuth, altitude) tuple of solar position.
'''
# def computeSunPosition(spadata):
#     spa.spa_calculate(spadata)
#     altitude = 90 - spadata.zenith  # this application uses altitude (90 - zenith)
#     return (spadata.azimuth, altitude)

'''
Function to fill a spa_data object from NREL SPA with specified date and time.
:param spadata: spa_data object
:param dt: datetime object
:note: NREL SPA can be found at https://midcdmz.nrel.gov/spa/
'''
def fillSPADateTime(spadata, dt):
    if spadata is None or dt is None:
        return
    spadata.year = dt.year
    spadata.month = dt.month
    spadata.day = dt.day
    spadata.hour = dt.hour
    spadata.minute = dt.minute
    spadata.second = dt.second

'''
Function to load configuration file. Parameters not passed into application will be loaded from config file.
'''
def loadConfig(args):
    if not os.path.exists(args.cfgpath):
        return False

    # load config file
    with open(args.cfgpath, 'r') as file:
        args.config = json.load(file)
    if not args.config or len(args.config) <= 0:
        return False

    # the following config options are required

    # collect ml features of interest
    args.features = args.config["Features"]
    args.featidxmap = {args.features[i]: i for i in range(0, len(args.features))}
    args.curvebeg = args.config["SpectrumStart"]
    args.curveend = args.config["SpectrumEnd"]
    args.curvelen = args.curveend - args.curvebeg + 1

    # collect lens warp/linearity data
    args.lens = "(ideal)"
    args.lenswarp = tuple([0,0,0,1/(math.pi/2.0),0])
    args.lensinv = tuple([0,0,0,(math.pi/2.0),0])
    if "Lens" in args.config:
        if "Name" in args.config["Lens"]:
            args.lens = tuple(args.config["Lens"]["Name"])
        if "Linearity" in args.config["Lens"]:
            args.lenswarp = tuple(args.config["Lens"]["Linearity"])
        if "Inverse" in args.config["Lens"]:
            args.lensinv = tuple(args.config["Lens"]["Inverse"])

    # collect SPA data (init values from their example)
    args.spa = spa.spa_data()
    args.spa.year = 2003
    args.spa.month = 10
    args.spa.day = 17
    args.spa.hour = 12
    args.spa.minute = 30
    args.spa.second = 30
    args.spa.time_zone = -7.0
    args.spa.delta_ut1 = 0
    args.spa.delta_t = 67
    args.spa.longitude = -105.1786
    args.spa.latitude = 39.742476
    args.spa.elevation = 1830.14
    args.spa.pressure = 820
    args.spa.temperature = 11
    args.spa.slope = 30
    args.spa.azm_rotation = -10
    args.spa.atmos_refract = 0.5667
    args.spa.function = spa.SPA_ZA
    # collect SPA values from config
    if "SPA" in args.config:
        args.spa.time_zone = float(args.config["SPA"]["time_zone"])
        args.spa.delta_ut1 = float(args.config["SPA"]["delta_ut1"])
        args.spa.delta_t = float(args.config["SPA"]["delta_t"])
        args.spa.longitude = float(args.config["SPA"]["longitude"])
        args.spa.latitude = float(args.config["SPA"]["latitude"])
        args.spa.elevation = float(args.config["SPA"]["elevation"])
        args.spa.pressure = float(args.config["SPA"]["pressure"])
        args.spa.temperature = float(args.config["SPA"]["temperature"])
        args.spa.slope = float(args.config["SPA"]["slope"])
        args.spa.azm_rotation = float(args.config["SPA"]["azm_rotation"])
        args.spa.atmos_refract = float(args.config["SPA"]["atmos_refract"])

    # the following config options are backups in case not specified by cmd line
    if not args.photo and "Photo" in args.config:
        args.photo = args.config["Photo"]
    if not args.timestamp and "Timestamp" in args.config:
        args.timestamp = args.config["Timestamp"]
    if not args.cover and "SkyCover" in args.config:
        args.cover = args.config["SkyCover"]
    if not args.model and "Model" in args.config:
        args.model = args.config["Model"]
    if not args.scaler and "Scaler" in args.config:
        args.scaler = args.config["Scaler"]
    if not args.polynomial and "Polynomial" in args.config:
        args.polynomial = args.config["Polynomial"]
    if not args.colorbar and "ColorBar" in args.config:
        args.colorbar = args.config["ColorBar"]

    # the following options are completely optional from config, not cmd line
    if "AltitudeConstraint" in args.config:
        args.altitude = args.config["AltitudeConstraint"]
    if "CoordSpace" in args.config:
        args.coordspace = args.config["CoordSpace"]

    # drop-through defaults
    if not args.polynomial:
        args.polynomial = 0
    if not args.altitude:
        args.altitude = 0  # essentially no constraint
    if not args.coordspace:
        args.coordspace = 1  # Polar

    return True

'''
Function to vet all program input whether from config or command line.
'''
def inputValidation(args):
    # error check the following required arguments
    if not args.photo:
        log(args, "Error: Photo not specified.")
        return False
    if not os.path.exists(args.photo):
        log(args, "Error: Could not find photo: " + args.photo)
        return False
    if not args.model:
        log(args, "Error: Model not specified.")
        return False
    if not os.path.exists(args.model):
        log(args, "Error: Could not find model: " + args.model)
        return False
    if not args.spa:
        log(args, "Error: Sun Positioning Algorithm (SPA) data not specified.")
        return False
    if not args.timestamp:
        log(args, "Error: Capture timestamp not specified.")
        return False
    try:
        ts = datetime.strptime(args.timestamp, "%Y/%m/%d %H:%M")  # make sure conversion succeeds
        ts = ts.replace(second=0, microsecond=0)                  # clear anything beyond minutes
        args.capture = ts                                         # cache datetime object
        args.timestamp = ts.strftime("%Y/%m/%d %H:%M")            # cache final datetime string
    except ValueError:
        log(args, "Error: Invalid capture timestamp: " + args.timestamp)
        return False
    if not args.cover:
        log(args, "Error: Sky cover not specified.")
        return False

    # error check the following optional arguments
    if args.scaler and len(args.scaler) > 0:
        if not os.path.exists(args.scaler):
            log(args, "Error: Could not find model scaler: " + args.scaler)
            return False

    # error check ranges (clamp accordingly)
    args.polynomial = min(max(0, args.polynomial), 10)
    args.cover = min(max(1, args.cover), 4)  # valid sky covers: 1=UNK, 2=CLR, 3=SCT, 4=OVC
    args.coordspace = min(max(1, args.coordspace), 3)  # valid coordinate spaces: 1=Polar, 2=PolarNorm, 3=UV
    args.altitude = min(max(0, args.altitude), 90)  # valid altitude constraint 0-90
    args.circumsolar = min(max(0, args.circumsolar), 45)  # valid circumsolar region 0-45 (radius)

    # init after input validation
    # name of photo used as filename for all output files
    args.filename = os.path.splitext(args.photo)[0]

    # warnings
    if args.lens == "(ideal)":
        log(args, "Warning: No lens warp/linearity specified; ideal lens assumed.")

    return True

'''
Entry point
'''
def main():
    # handle command line args
    parser = argparse.ArgumentParser(description='Framework for machine learning sky radiance sample datasets.', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_help = True
    # required parameters (either as args or config file)
    parser.add_argument('-p', '--photo', dest='photo', type=str, help='path to fisheye sky photo')
    parser.add_argument('-m', '--model', dest='model', type=str, help='path to ml model')
    parser.add_argument('-t', '--timestamp', dest='timestamp', type=str, help='datetime of capture (00/00/0000 00:00)')
    parser.add_argument('-c', '--cover', dest='cover', type=int, help='sky cover (1=UNK, 2=CLR, 3=SCT, 4=OVC)')
    # optional parameters
    parser.add_argument('-s', '--scaler', dest='scaler', type=str, help='path to ml model scaler')
    parser.add_argument('-y', '--polynomial', dest='polynomial', type=int, help='polynomial feature expansion')
    parser.add_argument('-cs', '--circumsolar', dest='circumsolar', type=float, default=0, help='consider circumsolar region (radius)')
    parser.add_argument('-g', '--grayscale', dest='grayscale',action='store_true', help='generate grayscale map')
    parser.add_argument('-v', '--visible', dest='visible', action='store_true', help='generate visible spectrum map')
    parser.add_argument('-b', '--colorbar', dest='colorbar', action='store_true', help='export colorbar as well')
    parser.add_argument('-x', '--export', dest='export', action='store_true', help='export predictions to file')
    parser.add_argument('-l', '--log', dest='log', action='store_true', help='log progress to stdout and file')
    args = parser.parse_args()

    # start logging this job
    args.logpath = "log.txt"
    log(args, "-" * 80)
    log(args, ' '.join(sys.argv))

    # load config file, and validate input
    args.cfgpath = "config.json"
    if not loadConfig(args): sys.exit(2)
    if not inputValidation(args): sys.exit(2)

    # configure matplotlib
    params = {'legend.fontsize': 'x-large',
              'axes.labelsize': 'x-large',
              'axes.titlesize': 'xx-large',
              'xtick.labelsize': 'large',
              'ytick.labelsize': 'x-large'}
    plt.rcParams.update(params)

    # compute timestamp binned features
    ts = pd.Series(pd.to_datetime(args.timestamp))
    args.capture_hour = ts.dt.hour[0]
    args.capture_dayofyear = ts.dt.dayofyear[0]
    args.capture_weekofyear = ts.dt.weekofyear[0]
    args.capture_quarter = ts.dt.quarter[0]
    args.capture_month = ts.dt.month[0]

    # load sky photo, scrape pixels and various info
    with Image.open(args.photo) as image:
        pixels = image.load()
        dimensions = image.size
        center = (int(dimensions[0] / 2), int(dimensions[1] / 2))
        diameter = dimensions[1]  # diameter of fisheye area is height of photo
        radius = int(diameter / 2)
        topleft = (center[0] - radius, center[1] - radius)
        bottomright = (center[0] + radius, center[1] + radius)
        #diagonal = math.sqrt(((bottomright[0] - topleft[0]) * (bottomright[0] - topleft[0]) +
        #                      (bottomright[1] - topleft[1]) * (bottomright[1] - topleft[1])))

    # compute radius w/ altitude constraint
    u, v = SkyCoord2FisheyeUV(args, 90, args.altitude)
    x = topleft[0] + int(u * diameter)
    y = topleft[1] + int(v * diameter)
    altradius2 = ((x - center[0]) * (x - center[0])) + ((y - center[1]) * (y - center[1]))
    altradius = int(math.sqrt(altradius2))
    sqrarea = diameter * diameter                   # number of pixels in square area around center
    fsharea = int(math.pi * radius * radius)        # number of pixels in fisheye circle
    altarea = int(math.pi * altradius * altradius)  # number of pixels in fisheye circle w/ altitude constraint

    # compute sun position for all samples of this photo/timestamp
    fillSPADateTime(args.spa, args.capture)
    sunpos = computeSunPosition(args.spa)
    sunposrad = (math.radians(sunpos[0]), math.radians(sunpos[1]))
    sunposxy = (SkyCoord2FisheyeUV(args, sunpos[0], sunpos[1]))
    sunposxy = (int(round(sunposxy[0] * diameter)), int(round(sunposxy[1] * diameter)))
    if args.coordspace == 1:  # Polar space
        pass  # already in polar
    # if args.coordspace == 2:  # PolarNorm space
    #     sunpos = (sunpos[0]/360.0, sunpos[1]/90.0)
    # elif args.coordspace == 3:  # UV space
    #     sunpos = (SkyCoord2FisheyeUV(args, sunpos[0], sunpos[1]))
    sunpos = (round(sunpos[0], 4), round(sunpos[1], 4))

    # report metrics
    log(args, "time:\t{0} (quarter: {1}, month: {2}, week: {3}, day: {4}, hour: {5})".format(args.timestamp, args.capture_quarter, args.capture_month, args.capture_weekofyear, args.capture_dayofyear, args.capture_hour))
    log(args, "sun:\t({0:.04f}, {1:.04f})".format(sunpos[0], sunpos[1]))
    log(args, "sunxy:\t({0}, {1})".format(sunposxy[0], sunposxy[1]))
    log(args, "alt:\t{0}deg altitude constraint".format(args.altitude))
    log(args, "photo:\t{0}".format(args.photo))
    log(args, "size:\t{0}px x {1}px".format(dimensions[0], dimensions[1]))
    log(args, "radius:\t{0}px / {1}px (with and without altitude constraint)".format(radius, altradius))
    log(args, "area:\t{0}px / {1}px / {2}px (square, fisheye, constrained)".format(sqrarea, fsharea, altarea))
    log(args, "model:\t{0}".format(args.model))

    # process each pixel of interest of the photo and extract features for ml model
    # consider all pixels in (height x height) area about center of photo
    log(args, "Extracting features from pixels...")
    samples = []
    samplestps = []
    samplesuvs = []
    samplespas = []
    timestart = time.time()
    idx = 0
    for r in range(topleft[1], bottomright[1]+1):
        for c in range(topleft[0], bottomright[0]+1):
            # ignore pixels outside of fisheye radius w/ altitude constraints
            dist2 = ((c - center[0]) * (c - center[0])) + ((r - center[1]) * (r - center[1]))
            if dist2 > altradius2: continue

            # these features should be the same for every pixel in the photo
            row = []
            row.append(args.capture_month)                      # month
            row.append(args.capture_quarter)                    # quarter
            row.append(args.capture_weekofyear)                 # weekofyear
            row.append(args.capture_dayofyear)                  # dayofyear
            row.append(args.capture_hour)                       # hour
            row.extend(sunpos)                                  # SunAzimuth, SunAltitude
            #row.append(args.cover)                              # SkyCover

            # compute sample UV and skycoord position
            u = (c - topleft[0]) / diameter
            v = r / diameter
            azi, alt = FisheyeUV2SkyCoord(args, u, v)
            samplestps.append([azi, alt])
            samplesuvs.append([u, v])
            # tp = (round(azi,1), round(alt,1))
            # sampletpmap[tp] = idx
            # d = math.sqrt((azi-float(60))**2 + (alt-float(30))**2)
            # if d < 0.5:
            #     print("(" + str(round(azi,1)) + ", " + str(round(alt,1)) + ") ", end='')

            # add sample azimuth and altitude
            row.append(0)  # NOTE: hardcoding SampleAzimuth to 0 right now!!
            #row.extend([0, 0])  # hardcoding no sample position right now
            #row.append(round(azi, 4))  # hardcoding adding SampleAzimuth in polar right now
            # if args.coordspace == 1:    # Polar
            #     row.extend([round(azi, 4), round(alt, 4)])      # SampleAzimuth, SampleAltitude
            # elif args.coordspace == 2:  # PolarNorm
            #     row.extend([round(azi/360.0, 4), round(alt/90.0, 4)])
            # elif args.coordspace == 3:  # UV
            #     row.extend([round(u, 4), round(v, 4)])          # SampleAzimuth, SampleAltitude

            # add sun-point-angle
            point = (math.radians(azi), math.radians(alt))
            angle = CentralAngle(sunposrad, point)
            spang = round(math.degrees(angle), 3)
            samplespas.append(spang)
            row.append(spang)           # SunPointAngle

            # add pixel color
            row.extend(pixels[c, r])                            # ColorA, ColorB, ColorC

            # append to collection
            samples.append(row)
            idx += 1

            # DEBUG: fill pixels w/ default color
            #u, v = SkyCoord2FisheyeUV(args, azi, alt)
            #x = int(round(u * diameter + topleft[0]))
            #y = int(round(v * diameter + topleft[1]))
            #sradmappix[c, r] = (0, 255, 0, 255)  # paint it green for c,r coords
            #sradmappix[x, y] = (0, 0, 255, 255)  # paint it blue for x,y coords

    log(args, "Extracted features from {0} pixels in {1}s".format(len(samples), int(time.time() - timestart)))

    # convert to numpy 2d array
    numpix = len(samples)
    X = np.array(samples)
    Xt = X

    # generate polynomial features, if requested
    if args.polynomial > 0:
        polyfeatures = PolynomialFeatures(args.polynomial)
        polyfeatures = polyfeatures.fit(Xt)
        Xt = polyfeatures.transform(Xt)

    # load model and scaler(s)
    log(args, "Loading model...")
    timestart = time.time()
    model = joblib.load(args.model)
    if args.scaler:
        scaler = joblib.load(args.scaler)
        Xt = scaler.transform(Xt)
    log(args, "Loaded model in {0}s".format(int(time.time() - timestart)))

    # predict!
    log(args, "Predicting...")
    timestart = time.time()
    preds = model.predict(Xt)
    log(args, "Predictions computed in {0}s".format(int(time.time() - timestart)))

    # collect, compute metrics
    rads = np.sum(preds, axis=1, keepdims=True)
    minrad = np.min(rads)
    maxrad = np.max(rads)
    sumrad = np.sum(rads)
    xrads = np.reshape(rads, (1, numpix)).flatten()
    nrads = (xrads - minrad) / (maxrad - minrad)
    # nrads = (nrads - np.min(nrads)) / np.ptp(nrads)
    # nrads = np.rint(200 * ((nrads - np.min(nrads)) / np.ptp(nrads))).astype(np.uint8)
    radsUV = preds[:, 350-args.curvebeg:(400-args.curvebeg)+1]
    radsVIS = preds[:, 380-args.curvebeg:(780-args.curvebeg)+1]
    rads600 = preds[:, 600-args.curvebeg:(600-args.curvebeg)+1]
    radsIR = preds[:, 780-args.curvebeg:(1780-args.curvebeg)+1]
    radsVNIR = preds[:, 350-args.curvebeg:(1780-args.curvebeg)+1]
    sumradUV = np.sum(radsUV)
    sumradVIS = np.sum(radsVIS)
    sumrad600 = np.sum(rads600)
    sumradIR = np.sum(radsIR)
    sumradVNIR = np.sum(radsVNIR)

    # log metrics
    log(args, "Sky RadMin:   {0:.4f}".format(minrad))
    log(args, "Sky RadMax:   {0:.4f}".format(maxrad))
    log(args, "Sky Total:    {0:.4f}".format(sumrad))
    log(args, "Sky Adjust:   {0:.4f}".format(sumrad * 0.000538))
    log(args, "Sky 73%:      {0:.4f}".format(sumrad * 0.000538 * 0.73))
    log(args, "Sky UV 73%:   {0:.4f}".format(sumradUV * 0.000538 * 0.73))
    log(args, "Sky VIS 73%:  {0:.4f}".format(sumradVIS * 0.000538 * 0.73))
    log(args, "Sky 600 73%:  {0:.4f}".format(sumrad600 * 0.000538 * 0.73))
    log(args, "Sky IR 73%:   {0:.4f}".format(sumradIR * 0.000538 * 0.73))
    log(args, "Sky VNIR 73%: {0:.4f}".format(sumradVNIR * 0.000538 * 0.73))

    # save prediction results to csv file, if requested
    if args.export:
        # merge features and predicted outputs
        results = np.concatenate((X, rads, preds), axis=1)
        log(args, "Exporting results...")
        timestart = time.time()
        resolution = 1 # int(args.curvelen / model.n_outputs_)
        header = ''.join(str(f)+',' for f in args.features) + 'Radiance,' + ''.join(str(i)+',' for i in range(args.curvebeg, args.curveend+1, resolution))
        header = header[:-1]  # trim trailing comma
        formats = '%d, %d, %d, %d, %d, %.4f, %.4f, %.4f, %.4f, %d, %d, %d, ' + '%.4f, ' + ''.join('%f, ' for i in range(args.curvebeg, args.curveend+1, resolution))  # %.4f,
        formats = formats[:-2]  # trim trailing comma and space
        np.savetxt(args.filename + ".csv", results, delimiter=',', comments='', header=header, fmt=formats)
        log(args, "Exported results in {0}s".format(int(time.time() - timestart)))

    # colormaps for colors for false color renders
    #cmap = plt.get_cmap("viridis")
    #cmap = plt.cm.cool
    #cmap = plt.get_cmap("Blues_r")
    #cmap = mpl.colors.LinearSegmentedColormap.from_list("", ["blue", "white", "red"])
    #cmap = ReNormColormapAdaptor(mpl.cm.jet, mpl.colors.LogNorm(0, maxrad))
    cmap = mpl.colors.LinearSegmentedColormap.from_list("", ["mediumblue", "white", "xkcd:fire engine red"])
    if args.grayscale:
        gmap = plt.get_cmap("gray")

    # save colorbar to file, if requested
    if args.colorbar:
        # rads colored
        fig = plt.figure(figsize=(1, 3))
        ax1 = fig.add_axes([0, 0, 0.25, 1.0])
        norm = mpl.colors.Normalize(vmin=0, vmax=maxrad)
        cb = mpl.colorbar.ColorbarBase(ax1, cmap=cmap, norm=norm, orientation='vertical')
        cb.set_label('Radiance ($W/m^2$) per pixel')
        fig.savefig(args.filename + "_cb.png", dpi=600, bbox_inches='tight', pad_inches=0)
        # rads grayscale
        if args.grayscale:
            fig = plt.figure(figsize=(1, 3))
            ax1 = fig.add_axes([0, 0, 0.25, 1.0])
            # #norm = mpl.colors.Normalize(vmin=0, vmax=maxrad)
            # #cb = mpl.colorbar.ColorbarBase(ax1, cmap=cmap, norm=mpl.colors.NoNorm, orientation='vertical')
            # #cb = plt.colorbar(ax1, cmap=cmap, orientation='vertical')
            # #cb.set_label('Radiance ($W/m^2/sr/nm$)')
            # img = ax1.imshow(data, interpolation='nearest', vmin=0.5, vmax=0.99)
            # fig.colorbar(img)
            norm = mpl.colors.Normalize(vmin=0, vmax=maxrad)
            cb = mpl.colorbar.ColorbarBase(ax1, cmap=gmap, norm=norm, orientation='vertical')
            cb.set_label('Radiance ($W/m^2$) per pixel')
            fig.savefig(args.filename + "_cb_gray.png", dpi=600, bbox_inches='tight', pad_inches=0)

    # generate full spectral radiance map (false-colored)
    log(args, "Generating false-colored spectral radiance map...")
    timestart = time.time()
    sradmap_rgba = Image.new("RGBA", (diameter, diameter), (255, 0, 255, 0))  # square image of fisheye portion
    sradmappix_rgba = sradmap_rgba.load()
    for i in range(0, numpix):
        x = int(round(samplesuvs[i][0] * diameter))
        y = int(round(samplesuvs[i][1] * diameter))
        rgba = cmap(nrads[i])
        sradmappix_rgba[x, y] = (int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255), 255)
    sradmap_rgba.save(args.filename + "_rgba.png", "PNG")
    log(args, "Generated in {0}s".format(int(time.time() - timestart)))

    # generate grayscale map
    if args.grayscale:
        log(args, "Generating grayscale spectral radiance map...")
        timestart = time.time()
        sradmap_gray = Image.new("RGBA", (diameter, diameter), (255, 0, 255, 0))  # square image of fisheye portion
        sradmappix_gray = sradmap_gray.load()
        for i in range(0, numpix):
            x = int(round(samplesuvs[i][0] * diameter))
            y = int(round(samplesuvs[i][1] * diameter))
            gray = gmap(nrads[i])
            sradmappix_gray[x, y] = (int(gray[0] * 255), int(gray[1] * 255), int(gray[2] * 255), 255)
        sradmap_gray.save(args.filename + "_gray.png", "PNG")
        log(args, "Generated in {0}s".format(int(time.time() - timestart)))

    # generate visible spectrum map (spectral render)
    if args.visible:
        log(args, "Generating visible spectrum radiance map...")
        visible = preds[:, 350 - args.curvebeg:830 - args.curvebeg:10]  # visible spectrum (350-830nm, 10nm resolution)
        timestart = time.time()
        sradmap_spec = Image.new("RGBA", (diameter, diameter), (255, 0, 255, 0))  # square image of fisheye portion
        sradmappix_spec = sradmap_spec.load()
        for i in range(0, numpix):
            x = int(round(samplesuvs[i][0] * diameter))
            y = int(round(samplesuvs[i][1] * diameter))
            vis = visible[i,:].tolist()
            vis.insert(0, 0) # 340nm
            spec = SpectralColor(*vis)
            cobj = convert_color(spec, sRGBColor)
            nrgb = cobj.get_value_tuple()
            sradmappix_spec[x, y] = (int(nrgb[0] * 255), int(nrgb[1] * 255), int(nrgb[2] * 255), 255)
        sradmap_spec.save(args.filename + "_spec.png", "PNG")
        log(args, "Generated in {0}s".format(int(time.time() - timestart)))

    # generate full spectral radiance map (false-colored) with circumsolar region highlighted
    if args.circumsolar > 0:
        log(args, "Generating circumsolar map...")
        timestart = time.time()
        sradmap_rgba = Image.new("RGBA", (diameter, diameter), (255, 0, 255, 0))  # square image of fisheye portion
        sradmappix_rgba = sradmap_rgba.load()
        csrads = []
        for i in range(0, numpix):
            x = int(round(samplesuvs[i][0] * diameter))
            y = int(round(samplesuvs[i][1] * diameter))
            rgba = cmap(nrads[i])
            sradmappix_rgba[x, y] = (int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255), 255)
            if samplespas[i] < args.circumsolar:
                sradmappix_rgba[x, y] = (255, 255, 0, 255)
                csrads.append(list(preds[i]))
            elif (samplespas[i] - args.circumsolar) <= 0.5:
                sradmappix_rgba[x, y] = (0, 0, 0, 255)
            # # To check point (x,y) inside projected circumsolar ellipse...
            # # First compute vector v=(x,y)−(sx,sy)
            # v = (x-sunposxy[0], y-sunposxy[1])
            # # Then project it onto the principal axes of the ellipse using dot products as follows:
            # av = (sunposaf[0]-sunposxy[0], sunposaf[1]-sunposxy[1])
            # ad = math.sqrt((av[0]*av[0]) + (av[1]*av[1]))
            # av = (av[0]/ad, av[1]/ad)
            # k = v[0]*av[0] + v[1]*av[1]
            # bv = (sunposbf[0]-sunposxy[0], sunposbf[1]-sunposxy[1])
            # bd = math.sqrt((bv[0]*bv[0]) + (bv[1]*bv[1]))
            # bv = (bv[0]/bd, bv[1]/bd)
            # l = v[0]*bv[0] + v[1]*bv[1]
            # # Then compute scaled sum of squares and compare with the cutoff:
            # test = ((k*k) / (ad*ad)) + ((l*l) / (bd*bd))
            # if test < 1:
            #     sradmappix_rgba[x, y] = (255, 255, 0, 255)
            #     solarpreds.append(list(preds[i]))
        sradmap_rgba.save(args.filename + "_solar.png", "PNG")
        log(args, "Generated in {0}s".format(int(time.time() - timestart)))
        # save circumsolar samples to csv file, if requested
        # if args.export:
        log(args, "Exporting circumsolar results...")
        timestart = time.time()
        csrads = np.array(csrads)
        # compute and log metrics
        solarrads = np.sum(csrads, axis=1, keepdims=True)
        csminrad = np.min(solarrads)
        csmaxrad = np.max(solarrads)
        cssumrad = np.sum(solarrads)
        csradsUV = csrads[:, 350 - args.curvebeg:(400 - args.curvebeg)+1]
        csradsVIS = csrads[:, 380 - args.curvebeg:(780 - args.curvebeg)+1]
        csrads600 = csrads[:, 600 - args.curvebeg:(600 - args.curvebeg)+1]
        csradsIR = csrads[:, 780 - args.curvebeg:(1780 - args.curvebeg)+1]
        csradsVNIR = csrads[:, 350 - args.curvebeg:(1780 - args.curvebeg)+1]
        cssumradUV = np.sum(csradsUV)
        cssumradVIS = np.sum(csradsVIS)
        cssumrad600 = np.sum(csrads600)
        cssumradIR = np.sum(csradsIR)
        cssumradVNIR = np.sum(csradsVNIR)
        log(args, "CS Samples: {0}".format(len(solarrads)))
        log(args, "CS RadMin:  {0:.4f}".format(csminrad))
        log(args, "CS RadMax:  {0:.4f}".format(csmaxrad))
        log(args, "CS Total:   {0:.4f}".format(cssumrad))
        log(args, "CS Adjust:  {0:.4f}".format(cssumrad * 0.000538))
        log(args, "CS UV:      {0:.4f}".format(cssumradUV * 0.000538))
        log(args, "CS VIS:     {0:.4f}".format(cssumradVIS * 0.000538))
        log(args, "CS 600:     {0:.4f}".format(cssumrad600 * 0.000538))
        log(args, "CS IR:      {0:.4f}".format(cssumradIR * 0.000538))
        log(args, "CS VNIR:    {0:.4f}".format(cssumradVNIR * 0.000538))

        # export
        solarresults = np.concatenate((solarrads, csrads), axis=1)
        resolution = 1 # int(args.curvelen / model.n_outputs_)
        header = 'Radiance,' + ''.join(str(i)+',' for i in range(args.curvebeg, args.curveend+1, resolution))
        header = header[:-1]  # trim trailing comma
        formats = '%.4f, ' + ''.join('%f, ' for i in range(args.curvebeg, args.curveend+1, resolution))  # %.4f,
        formats = formats[:-2]  # trim trailing comma and space
        np.savetxt(args.filename + "_solar.csv", solarresults, delimiter=',', comments='', header=header, fmt=formats)
        log(args, "Exported circumsolar results in {0}s".format(int(time.time() - timestart)))

# %%
if __name__ == "__main__":
    main()

# %%
