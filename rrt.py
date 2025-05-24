# Python implementation of ACES 1.3 RRT using numpy
# Made by Antoine Mayet
import numpy as np

# <ACEStransformID>urn:ampas:aces:transformId:v1.5:RRT.a1.0.3</ACEStransformID>
# <ACESuserName>ACES 1.0 - RRT</ACESuserName>

#
# Reference Rendering Transform (RRT)
#
#   Input is ACES
#   Output is OCES
#

from lib.transform_common import rgb_2_saturation
from lib.utilities_color import rgb_2_yc, rgb_2_hue
from lib.RRT_Common import (
    sigmoid_shaper,
    glow_fwd,
    center_hue,
    cubic_basis_shaper,
    RRT_GLOW_GAIN,
    RRT_GLOW_MID,
    RRT_RED_HUE,
    RRT_RED_PIVOT,
    RRT_RED_SCALE,
    RRT_RED_WIDTH,
)


def rrt_main(aces: np.array) -> np.array:
    # // --- Glow module --- //
    saturation = rgb_2_saturation(aces)
    ycIn = rgb_2_yc(aces)
    s = sigmoid_shaper((saturation - 0.4) / 0.2)
    addedGlow = 1.0 + glow_fwd(ycIn, RRT_GLOW_GAIN * s, RRT_GLOW_MID)

    aces = addedGlow * aces

    # // --- Red modifier --- //
    hue = rgb_2_hue(aces)
    centeredHue = center_hue(hue, RRT_RED_HUE)
    hueWeight = cubic_basis_shaper(centeredHue, RRT_RED_WIDTH)

    aces[0] = aces[0] + hueWeight * saturation * (RRT_RED_PIVOT - aces[0]) * (
        1.0 - RRT_RED_SCALE
    )

    return aces
