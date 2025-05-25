import math
import numpy as np
from .transform_common import AP1_2_XYZ_MAT, XYZ_2_AP1_MAT
from .utilities_color import XYZ_2_xyY, xyY_2_XYZ

# Target white and black points for cinema system tonescale
CINEMA_WHITE = 48.0
CINEMA_BLACK = math.pow(10, math.log10(0.02))  # CINEMA_WHITE / 2400.
# CINEMA_BLACK is defined in this roundabout manner in order to be exactly equal to
# the result returned by the cinema 48-nit ODT tonescale.
# Though the min point of the tonescale is designed to return 0.02, the tonescale is
# applied in log-log space, which loses precision on the antilog. The tonescale
# return value is passed into Y_2_linCV, where CINEMA_BLACK is subtracted. If
# CINEMA_BLACK is defined as simply 0.02, then the return value of this subfunction
# is very, very small but not equal to 0, and attaining a CV of 0 is then impossible.
# For all intents and purposes, CINEMA_BLACK=0.02.

DIM_SURROUND_GAMMA = 0.9811


def Y_2_linCV(Y: float, Ymax: float, Ymin: float) -> float:
    return (Y - Ymin) / (Ymax - Ymin)


def darkSurround_to_dimSurround(linearCV: np.array) -> np.array:
    XYZ = AP1_2_XYZ_MAT @ linearCV
    xyY = XYZ_2_xyY(XYZ)
    xyY[2] = np.clip(xyY[2], 0.0, np.finfo(np.float16).max)
    xyY[2] = pow(xyY[2], DIM_SURROUND_GAMMA)
    XYZ = xyY_2_XYZ(xyY)

    return XYZ_2_AP1_MAT @ XYZ
