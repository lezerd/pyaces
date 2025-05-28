import sys

sys.path.append("/Users/antoinemayet/Documents/GitHub/cubergb")
from rrt import rrt_main
from ODT_REC709_100nits_dim import main_odt_rec709D65
from cubergb import CubeRGB
import numpy as np
import OpenEXR
import Imath
import numpy as np

# Open the EXR file
infile = OpenEXR.InputFile("AP0.exr")

# Get image size
dw = infile.header()["dataWindow"]
width = dw.max.x - dw.min.x + 1
height = dw.max.y - dw.min.y + 1

# Define pixel type (typically HALF or FLOAT)
pt = Imath.PixelType(Imath.PixelType.FLOAT)

# Read each channel
r = np.frombuffer(infile.channel("R", pt), dtype=np.float32).reshape((height, width))
g = np.frombuffer(infile.channel("G", pt), dtype=np.float32).reshape((height, width))
b = np.frombuffer(infile.channel("B", pt), dtype=np.float32).reshape((height, width))

# Combine into a single RGB image
data = np.dstack((r, g, b))

data.clip(0, np.finfo(np.float16).max)
print(data[500, 500, :])
for i in range(0, data.shape[0]):
    for j in range(0, data.shape[1]):
        data[i, j, :] = main_odt_rec709D65(rrt_main(data[i, j, :]))
print(data[500, 500, :])
header = {
    "compression": OpenEXR.ZIP_COMPRESSION,
    "type": OpenEXR.scanlineimage,
}
channels = {"RGB": data}
with OpenEXR.File(header, channels) as outfile:
    outfile.write("OUTtest.exr")
