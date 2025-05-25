import numpy as np


# Function for genrating matrix to convert relative (0.0 to 1.0) RGB datas to XYZ
# Arguments :
#               Wx, Wy : White point cie 1931 chromaticity
#               Rx, Ry : Red point cie 1931 chromaticity
#               Gx, Gy : Green point cie 1931 chromaticity
#               Bx, By : Blue point cie 1931 chromaticity


def generateRGBtoXYZmatrix(Wx, Wy, Rx, Ry, Gx, Gy, Bx, By):
    # Generate white point missing XYZ coordinates
    Xw = Wx * (1 / Wy)
    Zw = (1 - Wx - Wy) * (1 / Wy)
    Yw = 1
    W = np.array([Xw, Yw, Zw])

    # Calculate z for each primaries
    Rz = 1 - Rx - Ry
    Gz = 1 - Gx - Gy
    Bz = 1 - Bx - By

    # Primary chromaticities as columns in XYZ space
    primaries = np.array([[Rx, Gx, Bx], [Ry, Gy, By], [Rz, Gz, Bz]])

    # Solve for scale factors
    S = np.linalg.inv(primaries) @ W

    # Apply scale factors to each primary column
    return primaries @ np.diag(S)


# Function for genrating matrix to convert normalizedimport coulour (Ymax = 1.0) XYZ datas to RGB
# Arguments :
#               Wx, Wy : White point cie 1931 chromaticity
#               Rx, Ry : Red point cie 1931 chromaticity
#               Gx, Gy : Green point cie 1931 chromaticity
#               Bx, By : Blue point cie 1931 chromaticity


def generateXYZtoRGBmatrix(Wx, Wy, Rx, Ry, Gx, Gy, Bx, By):
    # Inverse RGB to XYZ matrix
    return np.linalg.inv(generateRGBtoXYZmatrix(Wx, Wy, Rx, Ry, Gx, Gy, Bx, By))


def xytoXYZ(xy, Y=1):
    S = Y / xy[1]
    X = xy[0] * S
    Z = (1 - xy[0] - xy[1]) * S

    return np.array([X, Y, Z])


def XYZtoxy(XYZ):
    S = sum(XYZ)
    return np.array([XYZ[0] / S, XYZ[1] / S])
