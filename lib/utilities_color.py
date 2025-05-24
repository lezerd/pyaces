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
