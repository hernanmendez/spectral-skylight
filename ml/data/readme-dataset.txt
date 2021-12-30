=======================================================================================
Dataset columns
=======================================================================================

Date,Time,Space,SunAzimuth,SunAltitude,SkyCover,SamplePatternIndex,SampleAzimuth,SampleAltitude,PixelRegion,PixelWeighting,ColorModel,Exposure,ColorA,ColorB,ColorC,(wavelengths)

=======================================================================================
Dataset column descriptions
=======================================================================================

DATE, TIME
Sample's captured timestamp

SPACE
System/space for coordinates
1 Polar [0-360, 0-90] (degrees East from North)
2 UV    [0-1, 0-1]    (normalized distance between top-left of square fisheye portion of photo)

SUNAZIMUTH
Azimuth of sun
0-360 (Polar)
0-1   (UV)

SUNALTITUDE
Altitude of sun
0-90 (Polar)
0-1  (UV)
(if ever negative, capture was taken before dawn or after dusk)

SKYCOVER
Cloud cover at time of capture
1 UNK/unknown
2 CLR/clear
3 SCT/scattered
4 OVC/overcast

SAMPLEPATTERNINDEX
Index of sample coordinate in sampling pattern (not index in dataset)
0-80

SAMPLEAZIMUTH
Azimuth of sample
0-360 (Polar)
0-1   (UV)

SAMPLEALTITUDE
Altitude of sample
0-90 (Polar)
0-1  (UV)

SUNPOINTANGLE
Central angle (degrees) between sun and sample
0-180

PIXELREGION
n for an [n x n] matrix of pixels sampled
1
3
...
97
99

Values of PIXELWEIGHTING include:
1 Mean
2 Median
3 Gaussian

COLORMODEL
Model that colors are stored in
1 RGB
2 HSV
3 LAB

EXPOSURE
Exposure of sky photo capture (in seconds)
0.000125
0.001000
0.008000
0.066000
0.033000
0.250000
1.000000
2.000000
4.000000

COLORA, COLORB, COLORC:
Three components of color based on COLORMODEL
RGB -> 0-255, 0-255, 0-255
HSV -> 0-360, 0-100, 0-100
LAB -> 0-100, -128-128, -128-128

(wavelengths)
A column for each wavelength of the electromagnetic spectrum that was measured and exported (e.g. 350, 351, ..., 2499, 2500)
0-??
(if ever slightly negative, that means the sensor was noisy - theoretically should never be negative)
(if ever all or many 0s in a row, that means data is missing)

=======================================================================================
Dataset notes
=======================================================================================

Multiple EXPOSURE and resulting COLORABC values are possible, for an HDR dataset.
Example: Exposure1, Exposure2, ColorA1, ColorB1, ColorC1, ColorA2, ColorB2, ColorC2
