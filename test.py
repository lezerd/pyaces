import sys

sys.path.append("/Users/antoinemayet/Documents/GitHub/cubergb")
from rrt import rrt_main
from ODT_REC709_100nits_dim import main_odt_rec709D65
from cubergb import CubeRGB
import numpy as np
import OpenEXR

with OpenEXR.File("AP0.exr") as infile:
    data = infile.channels()["RGB"].pixels

data.clip(0, np.finfo(np.float16).max)

for i in range(0, data.shape[0]):
    for j in range(0, data.shape[1]):
        data[i, j, :] = main_odt_rec709D65(rrt_main(data[i, j, :]))

header = {
    "compression": OpenEXR.ZIP_COMPRESSION,
    "type": OpenEXR.scanlineimage,
}
channels = {"RGB": data}
with OpenEXR.File(header, channels) as outfile:
    outfile.write("REC709.exr")
