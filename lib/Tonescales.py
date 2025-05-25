from dataclasses import dataclass
import math
import numpy as np

M = np.array([[0.5, -1.0, 0.5], [-1.0, 1.0, 0.5], [0.5, 0.0, 0.0]])


@dataclass
class SplineMapPoint:
    x: float
    y: float


@dataclass
class SegmentedSplineParams_c5:
    coefsLow: list[
        float
    ]  # coefs for B-spline between minPoint and midPoint (units of log luminance)
    coefsHigh: list[
        float
    ]  # coefs for B-spline between midPoint and maxPoint (units of log luminance)
    minPoint: SplineMapPoint  # {luminance, luminance} linear extension below this
    midPoint: SplineMapPoint  # {luminance, luminance}
    maxPoint: SplineMapPoint  # {luminance, luminance} linear extension above this
    slopeLow: float  # log-log slope of low linear extension
    slopeHigh: float  # log-log slope of high linear extension


@dataclass
class SegmentedSplineParams_c9:
    coefsLow: list[
        float
    ]  # coefs for B-spline between minPoint and midPoint (units of log luminance)
    coefsHigh: list[
        float
    ]  # coefs for B-spline between midPoint and maxPoint (units of log luminance)
    minPoint: SplineMapPoint  # {luminance, luminance} linear extension below this
    midPoint: SplineMapPoint  # {luminance, luminance}
    maxPoint: SplineMapPoint  # {luminance, luminance} linear extension above this
    slopeLow: float  # log-log slope of low linear extension
    slopeHigh: float  # log-log slope of high linear extension


RRT_PARAMS = SegmentedSplineParams_c5(
    # coefsLow[6]
    [
        -4.0000000000,
        -4.0000000000,
        -3.1573765773,
        -0.4852499958,
        1.8477324706,
        1.8477324706,
    ],
    # coefsHigh[6]
    [
        -0.7185482425,
        2.0810307172,
        3.6681241237,
        4.0000000000,
        4.0000000000,
        4.0000000000,
    ],
    SplineMapPoint(0.18 * pow(2.0, -15), 0.0001),  # minPoint
    SplineMapPoint(0.18, 4.8),  # midPoint
    SplineMapPoint(0.18 * pow(2.0, 18), 10000.0),  # maxPoint
    0.0,  # slopeLow
    0.0,  # slopeHigh
)


# map 5.5e-6 - 47000 to 0 - 10000 and 0.18 to 4.8
def segmented_spline_c5_fwd(x, C: SegmentedSplineParams_c5 = RRT_PARAMS):
    N_KNOTS_LOW: int = 4
    N_KNOTS_HIGH: int = 4

    # Check for negatives or zero before taking the log. If negative or zero,
    # set to HALF_MIN.
    logx = math.log10(max(x, np.finfo(np.float16).tiny))
    logy: float = 0

    if logx <= math.log10(C.minPoint.x):
        logy = logx * C.slopeLow + (
            math.log10(C.minPoint.y) - C.slopeLow * math.log10(C.minPoint.x)
        )

    elif (logx > math.log10(C.minPoint.x)) and (logx < math.log10(C.midPoint.x)):
        knot_coord = (
            (N_KNOTS_LOW - 1)
            * (logx - math.log10(C.minPoint.x))
            / (math.log10(C.midPoint.x) - math.log10(C.minPoint.x))
        )
        j = int(knot_coord)
        t = knot_coord - j

        cf = np.array([C.coefsLow[j], C.coefsLow[j + 1], C.coefsLow[j + 2]])

        monomials = np.array([t * t, t, 1.0])
        logy = np.dot(monomials, (cf @ M))

    elif (logx >= math.log10(C.midPoint.x)) and (logx < math.log10(C.maxPoint.x)):
        knot_coord = (
            (N_KNOTS_HIGH - 1)
            * (logx - math.log10(C.midPoint.x))
            / (math.log10(C.maxPoint.x) - math.log10(C.midPoint.x))
        )
        j = int(knot_coord)
        t = knot_coord - j

        cf = np.array([C.coefsHigh[j], C.coefsHigh[j + 1], C.coefsHigh[j + 2]])

        monomials = np.array([t * t, t, 1.0])
        logy = np.dot(monomials, (cf @ M))
    else:  # if ( logIn >= log10(C.maxPoint.x) ) {
        logy = logx * C.slopeHigh + (
            math.log10(C.maxPoint.y) - C.slopeHigh * math.log10(C.maxPoint.x)
        )

    return math.pow(10, logy)


ODT_48nits = SegmentedSplineParams_c9(
    # coefsLow[10]
    [
        -1.6989700043,
        -1.6989700043,
        -1.4779000000,
        -1.2291000000,
        -0.8648000000,
        -0.4480000000,
        0.0051800000,
        0.4511080334,
        0.9113744414,
        0.9113744414,
    ],
    # coefsHigh[10]
    [
        0.5154386965,
        0.8470437783,
        1.1358000000,
        1.3802000000,
        1.5197000000,
        1.5985000000,
        1.6467000000,
        1.6746091357,
        1.6878733390,
        1.6878733390,
    ],
    SplineMapPoint(segmented_spline_c5_fwd(0.18 * pow(2.0, -6.5)), 0.02),  # minPoint
    SplineMapPoint(segmented_spline_c5_fwd(0.18), 4.8),  # midPoint
    SplineMapPoint(segmented_spline_c5_fwd(0.18 * pow(2.0, 6.5)), 48.0),  # maxPoint
    0.0,  # slopeLow
    0.04,  # slopeHigh
)


def segmented_spline_c9_fwd(x: float, C: SegmentedSplineParams_c9 = ODT_48nits):
    N_KNOTS_LOW = 8
    N_KNOTS_HIGH = 8

    # Check for negatives or zero before taking the log. If negative or zero,
    # set to HALF_MIN.
    logx = math.log10(max(x, np.finfo(np.float16).tiny))

    logy: float

    if logx <= math.log10(C.minPoint.x):
        logy = logx * C.slopeLow + (
            math.log10(C.minPoint.y) - C.slopeLow * math.log10(C.minPoint.x)
        )

    elif (logx > math.log10(C.minPoint.x)) and (logx < math.log10(C.midPoint.x)):
        knot_coord = (
            (N_KNOTS_LOW - 1)
            * (logx - math.log10(C.minPoint.x))
            / (math.log10(C.midPoint.x) - math.log10(C.minPoint.x))
        )
        j = int(knot_coord)
        t = knot_coord - j

        cf = np.array([C.coefsLow[j], C.coefsLow[j + 1], C.coefsLow[j + 2]])

        monomials = np.array([t * t, t, 1.0])
        logy = np.dot(monomials, (cf @ M))

    elif (logx >= math.log10(C.midPoint.x)) and (logx < math.log10(C.maxPoint.x)):
        knot_coord = (
            (N_KNOTS_HIGH - 1)
            * (logx - math.log10(C.midPoint.x))
            / (math.log10(C.maxPoint.x) - math.log10(C.midPoint.x))
        )
        j = int(knot_coord)
        t = knot_coord - j

        cf = np.array([C.coefsHigh[j], C.coefsHigh[j + 1], C.coefsHigh[j + 2]])
        monomials = np.array([t * t, t, 1.0])
        logy = np.dot(monomials, (cf @ M))
    else:  # if ( logIn >= log10(C.maxPoint.x) ) {
        logy = logx * C.slopeHigh + (
            math.log10(C.maxPoint.y) - C.slopeHigh * math.log10(C.maxPoint.x)
        )

    return math.pow(10, logy)
