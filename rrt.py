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

from lib.transform_common import rgb_2_saturation, AP0_to_AP1_MATRIX, AP1_to_AP0_MATRIX
from lib.utilities_color import rgb_2_yc, rgb_2_hue
from lib.Tonescales import segmented_spline_c5_fwd
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
    RRT_SAT_MAT,
)
from lib.dtype import f32


def rrt_main(
    aces: np.array,
    glow=True,
    red_modifier=True,
    desaturation=True,
    tonescale=True,
    clip=True,
) -> np.array:
    aces = f32(aces)

    if glow:
        # // --- Glow module --- //
        saturation = rgb_2_saturation(aces)
        ycIn = rgb_2_yc(aces)
        s = sigmoid_shaper((saturation - f32(0.4)) / f32(0.2))
        addedGlow = f32(f32(1.0) + glow_fwd(ycIn, RRT_GLOW_GAIN * s, RRT_GLOW_MID))

        aces = f32(addedGlow * aces)

    if red_modifier:
        # // --- Red modifier --- //
        hue = rgb_2_hue(aces)
        centeredHue = center_hue(hue, RRT_RED_HUE)
        hueWeight = cubic_basis_shaper(centeredHue, RRT_RED_WIDTH)

        aces[0] = f32(
            aces[0]
            + hueWeight
            * saturation
            * (RRT_RED_PIVOT - aces[0])
            * (f32(1.0) - RRT_RED_SCALE)
        )

    # // --- ACES to RGB rendering space --- //
    aces = f32(
        np.clip(aces, f32(0.0), float("inf"))
    )  # avoids saturated negative colors from becoming positive in the matrix

    rgbPre = f32(AP0_to_AP1_MATRIX @ aces)  # convert to AP1

    if clip:
        rgbPre = np.clip(
            rgbPre, 0.0, f32(65504.0)
        )  # CLip to max AP1 value 65504 (max float16 value)

    if desaturation:
        # // --- Global desaturation --- //
        rgbPre = f32(RRT_SAT_MAT @ rgbPre)

    if tonescale:
        # // --- Apply the tonescale independently in rendering-space RGB --- //
        rgbPost = f32(np.zeros((3)))
        rgbPost[0] = f32(segmented_spline_c5_fwd(rgbPre[0]))
        rgbPost[1] = f32(segmented_spline_c5_fwd(rgbPre[1]))
        rgbPost[2] = f32(segmented_spline_c5_fwd(rgbPre[2]))
    else:
        rgbPost = f32(np.copy(rgbPre))

    # // --- RGB rendering space to OCES --- //
    rgbOces = f32(AP1_to_AP0_MATRIX @ rgbPost)

    return rgbOces
