# Python implementation of ACES 1.3 REC709 ODT using numpy
# Made by Antoine Mayet
import numpy as np

# <ACEStransformID>urn:ampas:aces:transformId:v1.5:ODT.Academy.Rec709_100nits_dim.a1.0.3</ACEStransformID>
# <ACESuserName>ACES 1.0 Output - Rec.709</ACESuserName>

#
# Output Device Transform - Rec709
#

#
# Summary :
#  This transform is intended for mapping OCES onto a Rec.709 broadcast monitor
#  that is calibrated to a D65 white point at 100 cd/m^2. The assumed observer
#  adapted white is D65, and the viewing environment is a dim surround.
#
#  A possible use case for this transform would be HDTV/video mastering.
#
# Device Primaries :
#  Primaries are those specified in Rec. ITU-R BT.709
#  CIE 1931 chromaticities:  x         y         Y
#              Red:          0.64      0.33
#              Green:        0.3       0.6
#              Blue:         0.15      0.06
#              White:        0.3127    0.329     100 cd/m^2
#
# Display EOTF :
#  The reference electro-optical transfer function specified in
#  Rec. ITU-R BT.1886.
#
# Signal Range:
#    By default, this transform outputs full range code values. If instead a
#    SMPTE "legal" signal is desired, there is a runtime flag to output
#    SMPTE legal signal. In ctlrender, this can be achieved by appending
#    '-param1 legalRange 1' after the '-ctl odt.ctl' string.
#
# Assumed observer adapted white point:
#         CIE 1931 chromaticities:    x            y
#                                     0.3127       0.329
#
# Viewing Environment:
#   This ODT has a compensation for viewing environment variables more typical
#   of those associated with video mastering.
#
from lib.transform_common import AP0_to_AP1_MATRIX, AP1_to_AP0_MATRIX, AP1_2_XYZ_MAT
from lib.Tonescales import segmented_spline_c9_fwd
from lib.utilities_color import bt1886_r, fullRange_to_smpteRange_f3
from lib.ODT_Common import (
    Y_2_linCV,
    darkSurround_to_dimSurround,
    CINEMA_BLACK,
    CINEMA_WHITE,
    ODT_SAT_MAT,
    D60_2_D65_CAT,
)

# /* --- ODT Parameters --- */
XYZ_2_DISPLAY_PRI_MAT = np.array(
    [
        [3.24096994, -1.53738318, -0.49861076],
        [-0.96924364, 1.87596750, 0.04155506],
        [0.05563008, -0.20397696, 1.05697151],
    ]
)

DISPGAMMA = 2.4
L_W = 1.0
L_B = 0.0


def main_odt_rec709D65(
    oces: np.array,
    legalRange=False,
    tone_scale=True,
    surround_adaptation=True,
    desaturation=True,
    CAT=True,
    gamma=True,
    scale=True,
):
    # OCES to RGB rendering space
    rgbPre = AP0_to_AP1_MATRIX @ oces

    if tone_scale:
        # Apply the tonescale independently in rendering-space RGB
        rgbPost = np.empty((3))
        rgbPost[0] = segmented_spline_c9_fwd(rgbPre[0])
        rgbPost[1] = segmented_spline_c9_fwd(rgbPre[1])
        rgbPost[2] = segmented_spline_c9_fwd(rgbPre[2])
    else:
        rgbPost = np.copy(rgbPre)

    if scale:
        # // Scale luminance to linear code value
        linearCV = np.empty((3))
        linearCV[0] = Y_2_linCV(rgbPost[0], CINEMA_WHITE, CINEMA_BLACK)
        linearCV[1] = Y_2_linCV(rgbPost[1], CINEMA_WHITE, CINEMA_BLACK)
        linearCV[2] = Y_2_linCV(rgbPost[2], CINEMA_WHITE, CINEMA_BLACK)
    else:
        linearCV = np.copy(rgbPost)

    if surround_adaptation:
        # // Apply gamma adjustment to compensate for dim surround
        linearCV = darkSurround_to_dimSurround(linearCV)

    if desaturation:
        # // Apply desaturation to compensate for luminance difference
        linearCV = ODT_SAT_MAT @ linearCV

    if CAT:
        # // Convert to display primary encoding
        # // Rendering space RGB to XYZ
        XYZ = AP1_2_XYZ_MAT @ linearCV

        # // Apply CAT from ACES white point to assumed observer adapted white point
        XYZ = D60_2_D65_CAT @ XYZ

        # // CIE XYZ to display primaries
        linearCV = XYZ_2_DISPLAY_PRI_MAT @ XYZ

    # // Handle out-of-gamut values
    # // Clip values < 0 or > 1 (i.e. projecting outside the display primaries)
    linearCV = np.clip(linearCV, 0.0, 1.0)

    if gamma:
        # Encode linear code values with transfer function
        outputCV = np.empty((3))
        outputCV[0] = bt1886_r(linearCV[0], DISPGAMMA, L_W, L_B)
        outputCV[1] = bt1886_r(linearCV[1], DISPGAMMA, L_W, L_B)
        outputCV[2] = bt1886_r(linearCV[2], DISPGAMMA, L_W, L_B)
    else:
        outputCV = np.copy(linearCV)

    if legalRange:
        outputCV = fullRange_to_smpteRange_f3(outputCV)

    return outputCV
