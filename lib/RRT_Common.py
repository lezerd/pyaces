import math
import numpy as np
import copy
from .transform_common import AP1_RGB2Y
from .utilities_color import calc_sat_adjust_matrix
from .dtype import f32

# "Glow" module constants
RRT_GLOW_GAIN = f32(0.05)
RRT_GLOW_MID = f32(0.08)

# Red modifier constants
RRT_RED_SCALE = f32(0.82)
RRT_RED_PIVOT = f32(0.03)
RRT_RED_HUE = f32(0.0)
RRT_RED_WIDTH = f32(135.0)

RRT_SAT_FACTOR = 0.96
RRT_SAT_MAT = calc_sat_adjust_matrix(RRT_SAT_FACTOR, AP1_RGB2Y)


def sigmoid_shaper(x: float) -> float:
    # Sigmoid function in the range 0 to 1 spanning -2 to +2.

    t = np.maximum(f32(1.0) - np.abs(x / f32(2.0)), f32(0.0))
    y = f32(1.0) + np.sign(x) * (f32(1.0) - t * t)

    return f32(y / f32(2))


# ------- Glow module functions
def glow_fwd(ycIn: float, glowGainIn: float, glowMid: float) -> float:
    glowGainOut = f32(0)

    if ycIn <= f32(2.0) / f32(3.0) * glowMid:
        glowGainOut = glowGainIn
    elif ycIn >= f32(2.0) * glowMid:
        glowGainOut = f32(0.0)
    else:
        glowGainOut = glowGainIn * (glowMid / ycIn - f32(1.0) / f32(2.0))

    return f32(glowGainOut)


# // ------- Red modifier functions
def cubic_basis_shaper(
    x: float, w: float
) -> float:  # full base width of the shaper function (in degrees)
    M = np.array(
        [
            [
                f32(-1.0 / 6),
                f32(3.0 / 6),
                f32(-3.0 / 6),
                f32(1.0 / 6),
            ],
            [
                f32(3.0 / 6),
                f32(-6.0 / 6),
                f32(3.0 / 6),
                f32(0.0 / 6),
            ],
            [
                f32(-3.0 / 6),
                f32(0.0 / 6),
                f32(3.0 / 6),
                f32(0.0 / 6),
            ],
            [
                f32(1.0 / 6),
                f32(4.0 / 6),
                f32(1.0 / 6),
                f32(0.0 / 6),
            ],
        ]
    )

    knots = [-w / f32(2.0), -w / f32(4.0), f32(0.0), w / f32(4.0), w / f32(2.0)]

    y = f32(0.0)
    if (x > knots[0]) and (x < knots[4]):
        knot_coord = (x - knots[0]) * f32(4.0) / w
        j = int(knot_coord)
        t = knot_coord - j

        monomials = [t * t * t, t * t, t, f32(1.0)]

        # // (if/else structure required for compatibility with CTL < v1.5.)
        if j == 3:
            y = (
                monomials[0] * M[0][0]
                + monomials[1] * M[1][0]
                + monomials[2] * M[2][0]
                + monomials[3] * M[3][0]
            )
        elif j == 2:
            y = (
                monomials[0] * M[0][1]
                + monomials[1] * M[1][1]
                + monomials[2] * M[2][1]
                + monomials[3] * M[3][1]
            )
        elif j == 1:
            y = (
                monomials[0] * M[0][2]
                + monomials[1] * M[1][2]
                + monomials[2] * M[2][2]
                + monomials[3] * M[3][2]
            )
        elif j == 0:
            y = (
                monomials[0] * M[0][3]
                + monomials[1] * M[1][3]
                + monomials[2] * M[2][3]
                + monomials[3] * M[3][3]
            )
        else:
            y = 0.0

    return f32(y * f32(3) / f32(2.0))


# Return the hue in a scale centred around centerH in range (-180, 180)
def center_hue(hue: float, centerH: float) -> float:
    hueCentered = hue - centerH
    if hueCentered < f32(-180.0):
        hueCentered = hueCentered + f32(360)
    elif hueCentered > f32(180.0):
        hueCentered = hueCentered - f32(360)
    return f32(hueCentered)
