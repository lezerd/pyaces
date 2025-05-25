import numpy as np
import math

M_PI = 3.14159265358979323846


# Transformations from RGB to other color representations
def rgb_2_hue(rgb: np.array) -> float:
    # Returns a geometric hue angle in degrees (0-360) based on RGB values.
    # For neutral colors, hue is undefined and the function will return a quiet NaN value.
    hue = 0

    if rgb[0] == rgb[1] and rgb[1] == rgb[2]:
        hue = math.nan  # RGB triplets where RGB are equal have an undefined hue
    else:
        hue = (180.0 / M_PI) * math.atan2(
            math.sqrt(3) * (rgb[1] - rgb[2]), 2 * rgb[0] - rgb[1] - rgb[2]
        )

    if hue < 0.0:
        hue = hue + 360

    return hue


def rgb_2_yc(rgb: np.array, ycRadiusWeight: float = 1.75) -> float:
    # Converts RGB to a luminance proxy, here called YC
    # YC is ~ Y + K * Chroma
    # Constant YC is a cone-shaped surface in RGB space, with the tip on the
    # neutral axis, towards white.
    # YC is normalized: RGB 1 1 1 maps to YC = 1
    #
    # ycRadiusWeight defaults to 1.75, although can be overridden in function
    # call to rgb_2_yc
    # ycRadiusWeight = 1 -> YC for pure cyan, magenta, yellow == YC for neutral
    # of same value
    # ycRadiusWeight = 2 -> YC for pure red, green, blue  == YC for  neutral of
    # same value.

    r = rgb[0]
    g = rgb[1]
    b = rgb[2]

    chroma = math.sqrt(b * (b - g) + g * (g - r) + r * (r - b))
    return (b + g + r + ycRadiusWeight * chroma) / 3


def calc_sat_adjust_matrix(sat: float, rgb2Y: np.array) -> np.array:
    #
    # This function determines the terms for a 3x3 saturation matrix that is
    # based on the luminance of the input.
    #
    M = np.zeros((3, 3))
    M[0][0] = (1.0 - sat) * rgb2Y[0] + sat
    M[1][0] = (1.0 - sat) * rgb2Y[0]
    M[2][0] = (1.0 - sat) * rgb2Y[0]

    M[0][1] = (1.0 - sat) * rgb2Y[1]
    M[1][1] = (1.0 - sat) * rgb2Y[1] + sat
    M[2][1] = (1.0 - sat) * rgb2Y[1]

    M[0][2] = (1.0 - sat) * rgb2Y[2]
    M[1][2] = (1.0 - sat) * rgb2Y[2]
    M[2][2] = (1.0 - sat) * rgb2Y[2] + sat

    return M


# Transformations between CIE XYZ tristimulus values and CIE x,y
# chromaticity coordinates
def XYZ_2_xyY(XYZ: np.array) -> np.array:
    xyY = np.empty((3))
    divisor = XYZ[0] + XYZ[1] + XYZ[2]
    if divisor == 0.0:
        divisor = 1e-10
    xyY[0] = XYZ[0] / divisor
    xyY[1] = XYZ[1] / divisor
    xyY[2] = XYZ[1]
    return xyY


def xyY_2_XYZ(xyY: np.array) -> np.array:
    XYZ = np.empty((3))
    XYZ[0] = xyY[0] * xyY[2] / max(xyY[1], 1e-10)
    XYZ[1] = xyY[2]
    XYZ[2] = (1.0 - xyY[0] - xyY[1]) * xyY[2] / max(xyY[1], 1e-10)

    return XYZ
