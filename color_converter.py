import sys
import numpy as np

"""
=============== Program Architecture ===============

+-------+                      +-------+
|  CMY  |                      |  HSV  |
+---+---+                      +---+---+
    ^                              ^
    |                              |
    +---------+           +--------+
              |           |
              v           v
           +--+-----------+--+
           |       RGB       |
           +--------+--------+
                    ^
                    |
                    v
                +---+---+
                |  XYZ  |
                +---+---+
                    ^
                    |
                    v
                +---+----+
                | CIELAB |
                +--------+

We make use of 'chained' color conversions. For example,
to convert from cmy to CIELAB, we convert from cmy to RGB,
from RGB to XYZ and from XYZ to CIELAB, in this order.

"""

# RGB to HSV
# https://www.rapidtables.com/convert/color/rgb-to-hsv.html
# HSV to RGB
# https://www.rapidtables.com/convert/color/hsv-to-rgb.html

# RGB to cmy
# https://www.rapidtables.com/convert/color/rgb-to-cmyk.html
# cmy to RGB
# https://www.rapidtables.com/convert/color/cmyk-to-rgb.html

# RGB to XYZ and XYZ to RGB:
# https://www.image-engineering.de/library/technotes/958-how-to-convert-between-srgb-and-ciexyz

# XYZ to LAB and LAB to XYZ:
# http://www.easyrgb.com/en/math.php


def rgb2hsv(color):
    rgb = color / 255

    rr = rgb[0]
    gg = rgb[1]
    bb = rgb[2]

    cmax = np.max(rgb)
    cmin = np.min(rgb)

    delta = cmax - cmin

    # hue
    h = 0
    if(cmax == rr):
        h = 60 * (((gg - bb) / delta) % 6)
    elif(cmax == gg):
        h = 60 * (((bb - rr) / delta) + 2)
    elif(cmax == bb):
        h = 60 * (((rr - gg) / delta) + 4)

    # saturation
    s = 0
    if(cmax != 0):
        s = delta / cmax

    # value
    v = cmax

    return np.round(np.array([h, s, v]), decimals=2)


def hsv2rgb(color):
    h = color[0]
    s = color[1]
    v = color[2]

    c = s * v

    x = c * (1 - abs(((h / 60) % 2) - 1))

    rgb = [0, 0, 0]
    if((h >= 0) & (h < 60)):
        rgb = [c, x, 0]
    elif((h >= 60) & (h < 120)):
        rgb = [x, c, 0]
    elif((h >= 120) & (h < 180)):
        rgb = [0, c, x]
    elif((h >= 180) & (h < 240)):
        rgb = [0, x, c]
    elif((h >= 240) & (h < 300)):
        rgb = [x, 0, c]
    elif((h >= 300) & (h < 360)):
        rgb = [c, 0, x]

    m = v - c
    return np.round((np.array(rgb) + m) * 255)  # TODO: limit to range [0, 255]


# TODO: limit to range [0, 100]
def rgb2cmy(color):
    return np.round(np.array((-1 * (color / 255)) + 1), decimals=2)


def cmy2rgb(color):
    return np.round(np.array(1 - color) * 255)


def rgb2xyz(color):
    rgb = color / 255.0
    rgb = [(v / 12.92)
           if v <= 0.04045
           else pow((v + 0.055) / 1.055, 2.4)
           for v in rgb]
    rgb = np.array(rgb)

    # matrix specific to d50
    m = np.array([[0.4360747, 0.3850649, 0.1430804],
                  [0.2225045, 0.7168786, 0.0606169],
                  [0.0139322, 0.0971045, 0.7141733]])

    return np.round(np.matmul(m, rgb), decimals=4)


def xyz2rgb(color):
    # matrix specific to d50
    m = np.array([[3.1338561, -1.6168667, -0.4906146],
                  [-0.9787684, 1.9161415, 0.0334540],
                  [0.0719453, -0.2289914, 1.4052427]])

    rgb = np.matmul(m, np.array(color))

    rgb = [(v * 12.92)
           if v <= 0.0031308
           else ((1.055 * pow(v, 1/2.4)) - 0.055)
           for v in rgb]

    rgb = [0 if v < 0
           else 1 if v > 1
           else round(v, 2)
           for v in rgb]  # clamp values in [0, 1]

    return np.round(np.array(rgb) * 255)


def xyz2lab(color):
    d50 = [0.9642, 1, 0.8251]  # reference illuminant

    xyz = [0, 0, 0]
    for i in range(3):
        v = color[i] / d50[i]
        xyz[i] = pow(v, 1/3) if v > 0.008856 else ((v * 7.787) + (16 / 116))

    lab = [(116 * xyz[1]) - 16, 500 *
           (xyz[0] - xyz[1]), 200 * (xyz[1] - xyz[2])]

    return np.round(np.array(lab), decimals=2)


def lab2xyz(color):
    d50 = [0.9642, 1, 0.8251]  # reference illuminant

    y = (color[0] + 16) / 116
    x = (color[1] / 500) + y
    z = y - (color[2] / 200)
    xyz = [x, y, z]

    xyz = [pow(v, 3)
           if pow(v, 3) > 0.008856
           else ((v - (16 / 116)) / 7.787)
           for v in xyz]

    return np.round(np.multiply(xyz, d50), decimals=4)


def printError():
    print("One or more given color models aren't correct.")
    print("Accepted color models are: rgb, hsv, cmy, xyz, cielab.")
    exit()


model1 = sys.argv[1]
model2 = sys.argv[2]
color = np.array(sys.argv[3:]).astype(np.float)

if(model1 == 'rgb'):
    if(model2 == 'hsv'):
        color = rgb2hsv(color)
    elif(model2 == 'cmy'):
        color = rgb2cmy(color)
    elif(model2 == 'xyz'):
        color = rgb2xyz(color)
    elif(model2 == 'cielab'):
        color = rgb2xyz(color)
        color = xyz2lab(color)
    else:
        printError()
elif(model1 == 'hsv'):
    if(model2 == 'rgb'):
        color = hsv2rgb(color)
    elif(model2 == 'cmy'):
        color = hsv2rgb(color)
        color = rgb2cmy(color)
    elif(model2 == 'xyz'):
        color = hsv2rgb(color)
        color = rgb2xyz(color)
    elif(model2 == 'cielab'):
        color = hsv2rgb(color)
        color = rgb2xyz(color)
        color = xyz2lab(color)
    else:
        printError()
elif(model1 == 'cmy'):
    if(model2 == 'rgb'):
        color = cmy2rgb(color)
    elif(model2 == 'hsv'):
        color = cmy2rgb(color)
        color = rgb2hsv(color)
    elif(model2 == 'xyz'):
        color = cmy2rgb(color)
        color = rgb2xyz(color)
    elif(model2 == 'cielab'):
        color = cmy2rgb(color)
        color = rgb2xyz(color)
        color = xyz2lab(color)
    else:
        printError()
elif(model1 == 'xyz'):
    if(model2 == 'rgb'):
        color = xyz2rgb(color)
    elif(model2 == 'hsv'):
        color = xyz2rgb(color)
        color = rgb2hsv(color)
    elif(model2 == 'cmy'):
        color = xyz2rgb(color)
        color = rgb2cmy(color)
    elif(model2 == 'cielab'):
        color = xyz2lab(color)
    else:
        printError()
elif(model1 == 'cielab'):
    if(model2 == 'rgb'):
        color = lab2xyz(color)
        color = xyz2rgb(color)
    elif(model2 == 'hsv'):
        color = lab2xyz(color)
        color = xyz2rgb(color)
        color = rgb2hsv(color)
    elif(model2 == 'cmy'):
        color = lab2xyz(color)
        color = xyz2rgb(color)
        color = rgb2cmy(color)
    elif(model2 == 'xyz'):
        color = lab2xyz(color)
    else:
        printError()
else:
    printError()

print(model1 + ' to ' + model2 + ' ' +
      str(color[0]) + ' ' + str(color[1]) + ' ' + str(color[2]))
