=======================================================================================
Details about data capture process can be found in the following publication
=======================================================================================
Kider Jr, J. T., Knowlton, D., Newlin, J., Li, Y. K., & Greenberg, D. P. (2014). A framework for the experimental comparison of solar and skydome illumination. ACM Transactions on Graphics (TOG), 33(6), 180.

=======================================================================================
Camera lens used
=======================================================================================
Sigma 8mm f/3.5 EX DG
Lens fisheye correction linearity polynomial (real to fisheye):
-0.0004325x4 -0.0499x3 +0.0252x2 +0.7230x +0
Lens fisheye correction linearity inverse polynomial (fisheye to real):
0.4171x4 -0.3803x3 0.1755x2 1.3525x +0
See http://paulbourke.net/dome/fisheyecorrect

=======================================================================================
Site data used for NREL SPA sun path/position computations
Site: Frank Rhodes Hall, Cornell, Ithaca, NY (2012-2013)
=======================================================================================
time_zone	-5
delta_ut1	0.153
delta_t	66.9069
longitude	-76.481638
latitude	42.443441
elevation	325
pressure	1032.844454
temperature	-2.9
slope	0
azm_rotation	0
atmos_refract	0.5667

=======================================================================================
For HDR photos, the EXIF DateTimeOriginal values DID NOT MATCH
the folder and file names that were created when the data was captured!
=======================================================================================
OFF BY ?h:
2012-08-31

OFF BY 3h:
2012-10-22

OFF BY 1h:
2013-04-13
2013-04-14
2013-04-15
2013-04-26
2013-05-01
2013-05-02
2013-05-03
2013-05-07
2013-05-11
2013-05-12
2013-05-26
2013-05-27
2013-05-30
2013-05-31
2013-06-03
2013-06-15
2013-06-17
2013-07-02
2013-07-26
2013-09-24

=======================================================================================
For HDR photos, these capture photos were rotated to align sun with SPA algorithm.
Positive degrees are counterclockwise, negative clockwise.
=======================================================================================

This has the following negative effects:
 - RGB pixels WERE NOT PRESERVED 100% due to lossy JPEG format
 - CR2 raw image WERE NOT rotated due to lack of support of PIL API 
 - later generated TIFF images were rotated

2012-11-15 by 2°
2013-04-14 by 2°
2013-04-15 by 3°
2013-05-26 by -2°
2013-05-30 by 2°
2013-06-15 by -1°
2013-09-24 by -2°
2013-09-26 by 2°

=======================================================================================
For ASD files, convert with ViewSpecPro software like this.
=======================================================================================
Process -> Radiometric Calculation  (.rad)
Process -> ASCII Export (.txt)

So .asd to .asd.rad to .asd.rad.txt.
Which is oddly different then if you go straight from .asd to .txt

=======================================================================================
For ASD captures, some of the captures have less than the expected 81 samples.
=======================================================================================
These capture dates have roughly only 35-41 samples for every capture interval:
measurements\2012-08-31
measurements\2012-10-22
measurements\2012-10-25
measurements\2012-11-08
measurements\2012-11-09

For various other folders, there may not be exactly 81 samples per capture time (e.g. missing 1-5 samples).
In those cases, it is not exactly clear which sample is missing, and because of that we must not use this data at all,
because regardless of the name of the file, the data may not map to the sample indicated.
It is safer to avoid this data until we can find which sample is missing (log?)

=======================================================================================
Fun data capture errors
=======================================================================================
// dropped data
measurements\2013-05-31\ASD\09.00.00\74_225.00_71.9187_.asd
measurements\2013-05-31\ASD\09.15.00\45_150.00_33.7490_.asd
measurements\2013-05-31\ASD\09.45.00\02_022.50_12.1151_.asd
measurements\2013-07-26\ASD\12.00.00\46_135.00_33.7490_.asd
// bugs (literally)
measurements\2013-05-02\HDR\16.51.01
measurements\2013-09-24\HDR\12.39.30
// polarizing lens slip
measurements\2013-05-11_ruined\HDR
// glare
measurements\2012-11-15\HDR\11.43.25 (glare)

=======================================================================================
Sampling Pattern
=======================================================================================

Altitudes (Zenith)

12.1151 (77.8849)
33.749  (56.251)
53.3665 (36.6335)
71.9187 (18.0813)
90      (0)

Original pattern of pre-determined 81 points of measurement

0, 12.1151
11.25, 12.1151
22.5, 12.1151
33.75, 12.1151
45, 12.1151
56.25, 12.1151
67.5, 12.1151
78.75, 12.1151
90, 12.1151
101.25, 12.1151
112.5, 12.1151
123.75, 12.1151
135, 12.1151
146.25, 12.1151
157.5, 12.1151
168.75, 12.1151
180, 12.1151
191.25, 12.1151
202.5, 12.1151
213.75, 12.1151
225, 12.1151
236.25, 12.1151
247.5, 12.1151
258.75, 12.1151
270, 12.1151
281.25, 12.1151
292.5, 12.1151
303.75, 12.1151
315, 12.1151
326.25, 12.1151
337.5, 12.1151
348.75, 12.1151
345, 33.749
330, 33.749
315, 33.749
300, 33.749
285, 33.749
270, 33.749
255, 33.749
240, 33.749
225, 33.749
210, 33.749
195, 33.749
180, 33.749
165, 33.749
150, 33.749
135, 33.749
120, 33.749
105, 33.749
90, 33.749
75, 33.749
60, 33.749
45, 33.749
30, 33.749
15, 33.749
0, 33.749
0, 53.3665
22.5, 53.3665
45, 53.3665
67.5, 53.3665
90, 53.3665
112.5, 53.3665
135, 53.3665
157.5, 53.3665
180, 53.3665
202.5, 53.3665
225, 53.3665
247.5, 53.3665
270, 53.3665
292.5, 53.3665
315, 53.3665
337.5, 53.3665
315, 71.9187
270, 71.9187
225, 71.9187
180, 71.9187
135, 71.9187
90, 71.9187
45, 71.9187
0, 71.9187
0, 90

Pattern with azimuth transposed (360-azimuth) because measurements were captured going West from North (and polar coordinate system goes East from North)

0.0, 12.1151
348.75, 12.1151
337.5, 12.1151
326.25, 12.1151
315.0, 12.1151
303.75, 12.1151
292.5, 12.1151
281.25, 12.1151
270.0, 12.1151
258.75, 12.1151
247.5, 12.1151
236.25, 12.1151
225.0, 12.1151
213.75, 12.1151
202.5, 12.1151
191.25, 12.1151
180.0, 12.1151
168.75, 12.1151
157.5, 12.1151
146.25, 12.1151
135.0, 12.1151
123.75, 12.1151
112.5, 12.1151
101.25, 12.1151
90.0, 12.1151
78.75, 12.1151
67.5, 12.1151
56.25, 12.1151
45.0, 12.1151
33.75, 12.1151
22.5, 12.1151
11.25, 12.1151
15.0, 33.749
30.0, 33.749
45.0, 33.749
60.0, 33.749
75.0, 33.749
90.0, 33.749
105.0, 33.749
120.0, 33.749
135.0, 33.749
150.0, 33.749
165.0, 33.749
180.0, 33.749
195.0, 33.749
210.0, 33.749
225.0, 33.749
240.0, 33.749
255.0, 33.749
270.0, 33.749
285.0, 33.749
300.0, 33.749
315.0, 33.749
330.0, 33.749
345.0, 33.749
0.0, 33.749
0.0, 53.3665
337.5, 53.3665
315.0, 53.3665
292.5, 53.3665
270.0, 53.3665
247.5, 53.3665
225.0, 53.3665
202.5, 53.3665
180.0, 53.3665
157.5, 53.3665
135.0, 53.3665
112.5, 53.3665
90.0, 53.3665
67.5, 53.3665
45.0, 53.3665
22.5, 53.3665
45.0, 71.9187
90.0, 71.9187
135.0, 71.9187
180.0, 71.9187
225.0, 71.9187
270.0, 71.9187
315.0, 71.9187
0.0, 71.9187
0.0, 90.0
