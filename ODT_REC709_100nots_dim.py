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
from lib.transform_common import AP0_to_AP1_MATRIX, AP1_to_AP0_MATRIX
from lib.Tonescales import segmented_spline_c9_fwd
from lib.ODT_Common import Y_2_linCV, CINEMA_BLACK, CINEMA_WHITE

# /* --- ODT Parameters --- */
XYZ_2_DISPLAY_PRI_MAT = np.array(
    [[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]]
)

DISPGAMMA = 2.4
L_W = 1.0
L_B = 0.0


def main_odt_rec709D65(oces: np.array, legalRange=False):
    # OCES to RGB rendering space
    rgbPre = AP0_to_AP1_MATRIX @ oces

    # Apply the tonescale independently in rendering-space RGB
    rgbPost = np.empty((3))
    rgbPost[0] = segmented_spline_c9_fwd(rgbPre[0])
    rgbPost[1] = segmented_spline_c9_fwd(rgbPre[1])
    rgbPost[2] = segmented_spline_c9_fwd(rgbPre[2])

    # // Scale luminance to linear code value
    linearCV = np.empty((3))
    linearCV[0] = Y_2_linCV(rgbPost[0], CINEMA_WHITE, CINEMA_BLACK)
    linearCV[1] = Y_2_linCV(rgbPost[1], CINEMA_WHITE, CINEMA_BLACK)
    linearCV[2] = Y_2_linCV(rgbPost[2], CINEMA_WHITE, CINEMA_BLACK)
