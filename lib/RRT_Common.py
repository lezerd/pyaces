import math
import numpy as np
import copy

# "Glow" module constants
RRT_GLOW_GAIN = 0.05
RRT_GLOW_MID = 0.08

# Red modifier constants
RRT_RED_SCALE = 0.82
RRT_RED_PIVOT = 0.03
RRT_RED_HUE = 0.0
RRT_RED_WIDTH = 135.0


def sigmoid_shaper(x: float) -> float:
    # Sigmoid function in the range 0 to 1 spanning -2 to +2.

    t = max(1.0 - abs(x / 2.0), 0.0)
    y = 1.0 + np.sign(x) * (1.0 - t * t)

    return y / 2


# ------- Glow module functions
def glow_fwd(ycIn: float, glowGainIn: float, glowMid: float) -> float:
    glowGainOut = 0

    if ycIn <= 2.0 / 3.0 * glowMid:
        glowGainOut = glowGainIn
    elif ycIn >= 2.0 * glowMid:
        glowGainOut = 0.0
    else:
        glowGainOut = glowGainIn * (glowMid / ycIn - 1.0 / 2.0)

    return glowGainOut


# // ------- Red modifier functions
def cubic_basis_shaper(
    x: float, w: float
) -> float:  # full base width of the shaper function (in degrees)
    M = np.array(
        [
            [-1.0 / 6, 3.0 / 6, -3.0 / 6, 1.0 / 6],
            [3.0 / 6, -6.0 / 6, 3.0 / 6, 0.0 / 6],
            [-3.0 / 6, 0.0 / 6, 3.0 / 6, 0.0 / 6],
            [1.0 / 6, 4.0 / 6, 1.0 / 6, 0.0 / 6],
        ]
    )

    knots = [-w / 2.0, -w / 4.0, 0.0, w / 4.0, w / 2.0]

    y = 0.0
    if (x > knots[0]) and (x < knots[4]):
        knot_coord = (x - knots[0]) * 4.0 / w
        j = int(knot_coord)
        t = knot_coord - j

        monomials = [t * t * t, t * t, t, 1.0]

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

    return y * 3 / 2.0


# Return the hue in a scale centred around centerH in range (-180, 180)
def center_hue(hue: float, centerH: float) -> float:
    hueCentered = hue - centerH
    if hueCentered < -180.0:
        hueCentered = hueCentered + 360
    elif hueCentered > 180.0:
        hueCentered = hueCentered - 360
    return hueCentered
