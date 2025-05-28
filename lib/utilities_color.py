import numpy as np
import math
from dataclasses import dataclass

M_PI = 3.14159265358979323846


@dataclass
class Chromaticities:
    red: np.ndarray  # (x, y)
    green: np.ndarray
    blue: np.ndarray
    white: np.ndarray


# ACES AP0
AP0 = Chromaticities(
    red=np.array([0.73470, 0.26530]),
    green=np.array([0.00000, 1.00000]),
    blue=np.array([0.00010, -0.07700]),
    white=np.array([0.32168, 0.33767]),  # D60
)

# ACES AP1
AP1 = Chromaticities(
    red=np.array([0.71300, 0.29300]),
    green=np.array([0.16500, 0.83000]),
    blue=np.array([0.12800, 0.04400]),
    white=np.array([0.32168, 0.33767]),  # D60
)

# Rec.709
REC709_PRI = Chromaticities(
    red=np.array([0.64000, 0.33000]),
    green=np.array([0.30000, 0.60000]),
    blue=np.array([0.15000, 0.06000]),
    white=np.array([0.31270, 0.32900]),  # D65
)


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

    chroma = math.sqrt((b * (b - g) + g * (g - r) + r * (r - b)))
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


CONE_RESP_MAT_BRADFORD = np.array(
    [
        [0.89510, -0.75020, 0.03890],
        [0.26640, 1.71350, -0.06850],
        [-0.16140, 0.03670, 1.02960],
    ]
).T


def calculate_cat_matrix(
    src_xy: np.ndarray,  # x,y chromaticity of source white
    des_xy: np.ndarray,  # x,y chromaticity of destination white
    coneRespMat: np.ndarray = CONE_RESP_MAT_BRADFORD,
):
    # //
    # // Calculates and returns a 3x3 Von Kries chromatic adaptation transform
    # // from src_xy to des_xy using the cone response primaries defined
    # // by coneRespMat. By default, coneRespMat is set to CONE_RESP_MAT_BRADFORD.
    # // The default coneRespMat can be overridden at runtime.
    # //

    src_xyY = np.array([src_xy[0], src_xy[1], 1.0])
    des_xyY = np.array([des_xy[0], des_xy[1], 1.0])

    src_XYZ = xyY_2_XYZ(src_xyY)
    des_XYZ = xyY_2_XYZ(des_xyY)

    src_coneResp = coneRespMat @ src_XYZ
    des_coneResp = coneRespMat @ des_XYZ

    vkMat = np.array(
        [
            [des_coneResp[0] / src_coneResp[0], 0.0, 0.0],
            [0.0, des_coneResp[1] / src_coneResp[1], 0.0],
            [0.0, 0.0, des_coneResp[2] / src_coneResp[2]],
        ]
    )

    cat_matrix = np.linalg.inv(coneRespMat) @ vkMat @ coneRespMat

    return cat_matrix


def bt1886_f(V: float, gamma: float, Lw: float, Lb: float) -> float:
    # The reference EOTF specified in Rec. ITU-R BT.1886
    # L = a(max[(V+b),0])^g
    a = pow(pow(Lw, 1.0 / gamma) - pow(Lb, 1.0 / gamma), gamma)
    b = pow(Lb, 1.0 / gamma) / (pow(Lw, 1.0 / gamma) - pow(Lb, 1.0 / gamma))
    L = a * pow(max(V + b, 0.0), gamma)
    return L


def bt1886_r(L: float, gamma: float, Lw: float, Lb: float) -> float:
    # The reference EOTF specified in Rec. ITU-R BT.1886
    # L = a(max[(V+b),0])^g
    a = pow(pow(Lw, 1.0 / gamma) - pow(Lb, 1.0 / gamma), gamma)
    b = pow(Lb, 1.0 / gamma) / (pow(Lw, 1.0 / gamma) - pow(Lb, 1.0 / gamma))
    V = pow(max(L / a, 0.0), 1.0 / gamma) - b
    return V


def fullRange_to_smpteRange(input: float) -> float:
    REFBLACK = 64.0 / 1023.0
    REFWHITE = 940.0 / 1023.0
    return input * (REFWHITE - REFBLACK) + REFBLACK


def fullRange_to_smpteRange_f3(rgbIn: float):
    rgbOut = np.empty((3))
    rgbOut[0] = fullRange_to_smpteRange(rgbIn[0])
    rgbOut[1] = fullRange_to_smpteRange(rgbIn[1])
    rgbOut[2] = fullRange_to_smpteRange(rgbIn[2])
    return rgbOut
