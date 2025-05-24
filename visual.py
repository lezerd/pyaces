#
import sys

sys.path.append("/Users/antoinemayet/Documents/GitHub/cubergb")
from cubergb import CubeRGB
from lib.transform_common import rgb_2_saturation
from lib.utilities_color import rgb_2_yc
from lib.RRT_Common import sigmoid_shaper, RRT_GLOW_GAIN, RRT_GLOW_MID, glow_fwd
import numpy as np
import matplotlib.pyplot as plt
from rrt import rrt_main
import math


def lin_to_rle(lin):
    MIDDLE_GRAY_LE = -2.473931188332412
    return math.log2(max(lin, 2e-16)) - MIDDLE_GRAY_LE


def sRGBOETF(lin):
    return 1.055 * lin ** (1 / 2.4) - 0.055 if lin > 0.0031308 else 12.92 * lin


rle = np.vectorize(lin_to_rle)

# fig, ax = plt.subplots()
# plt.style.use("seaborn-v0_8-whitegrid")
# ax.spines["bottom"].set_linewidth(2)
# ax.spines["left"].set_linewidth(2)
# ax.spines["top"].set_visible(False)
# ax.spines["right"].set_visible(False)
# ax.tick_params(width=2)
# ax.grid(True)
# ax.set_xlim(0, 1)

### Visualise rgb_2_satuartion
##sat = CubeRGB.id(33)
##
##for i in range(0, sat.values.shape[0]):
##    for j in range(0, sat.values.shape[1]):
##        point_sat = rgb_2_saturation(sat.values[i, j, :])
##        if point_sat >= 0.5:
##            sat.alpha[i, j] = 0
##        sat.points_original_color[i, j, :] = np.array([point_sat, 0, 0])
##
##sat.show()

### Visualise rgb_2_yc
## vis = CubeRGB.id(66)
##
##for i in range(0, vis.values.shape[0]):
##   for j in range(0, vis.values.shape[1]):
##       point_yc = rgb_2_yc(vis.values[i, j, :])
##       if point_yc >= 0.5:
##           vis.alpha[i, j] = 0
##       vis.points_original_color[i, j, :] = np.array([np.clip(point_yc, 0, 1), 0, 0])
##
##vis.show()

# Visualise s

# x_axis = np.linspace(0, 1, 256)
# y_axis = []
# for x in x_axis:
#    y_axis.append(sigmoid_shaper((x - 0.4) / 0.2))
# y_axis = np.array(y_axis)
# plt.plot(x_axis, y_axis)
# plt.show()
#
# sat = CubeRGB.id(70)
#
# for i in range(0, sat.values.shape[0]):
#    for j in range(0, sat.values.shape[1]):
#        point_sat = rgb_2_saturation(sat.values[i, j, :])
#        point_sat = sigmoid_shaper((point_sat - 0.4) / 0.2)
#        if point_sat >= 0.5:
#            sat.alpha[i, j] = 0
#        sat.points_original_color[i, j, :] = np.array([point_sat, 0, 0])
#
# sat.show()

# Visualise addedglow
# vis = CubeRGB.id(70)
#
# for i in range(0, vis.values.shape[0]):
#    for j in range(0, vis.values.shape[1]):
#        point_sat = rgb_2_saturation(vis.values[i, j, :])
#        point_yc = rgb_2_yc(vis.values[i, j, :])
#        point_s = sigmoid_shaper((point_sat - 0.4) / 0.2)
#        glow = 1.0 + glow_fwd(point_yc, RRT_GLOW_GAIN * point_s, RRT_GLOW_MID)
#        if glow == 1:
#            vis.alpha[i, j] = 0
#        vis.points_original_color[i, j, :] = np.array([np.clip((glow - 1), 0, 1), 0, 0])
#
# vis.show()

# visualise glow module on red axis
# fig, ax = plt.subplots()
# plt.style.use("seaborn-v0_8-whitegrid")
# ax.spines["bottom"].set_linewidth(2)
# ax.spines["left"].set_linewidth(2)
# ax.spines["top"].set_visible(False)
# ax.spines["right"].set_visible(False)
# ax.tick_params(width=2)
# ax.grid(True)
# ax.set_xlim(0, 0.18)
# ax.set_ylim(1, 1.06)
##
##
# sat_axis = np.linspace(0, 0.18, 10)
# for gb in sat_axis:
#     x_axis = np.linspace(0, 0.18, 100)
#     y_axis = []
#     for x in x_axis:
#         rgb = np.array([x, gb, gb])
#         glowout = rrt_main(rgb)
#         saturation = rgb_2_saturation(rgb)
#         ycIn = rgb_2_yc(rgb)
#         s = sigmoid_shaper((saturation - 0.4) / 0.2)
#         glow_factor = 1 + glow_fwd(ycIn, RRT_GLOW_GAIN * s, RRT_GLOW_MID)
#         print(glow_factor)
#         y_axis.append(glow_factor)
#     y_axis = np.array(y_axis)
#     plt.plot(x_axis, y_axis, color=(sRGBOETF(0.18), sRGBOETF(gb), sRGBOETF(gb)))
# plt.show()

# Visualise glow module
# vis = CubeRGB.id(60)
# for i in range(0, vis.values.shape[0]):
#    for j in range(0, vis.values.shape[1]):
#        point_sat = rgb_2_saturation(vis.values[i, j, :])
#        point_yc = rgb_2_yc(vis.values[i, j, :])
#        point_s = sigmoid_shaper((point_sat - 0.4) / 0.2)
#        glow = 1.0 + glow_fwd(point_yc, RRT_GLOW_GAIN * point_s, RRT_GLOW_MID)
#        rgb = vis.values[i, j, :] * glow
#        if glow == 1:
#            vis.alpha[i, j] = 0
#        else:
#            print(f"In : {vis.values[i, j, :]}, Out: {rgb}")
#        vis.values[i, j, :] = rgb
# vis.show()

# visualise hue
##from lib.utilities_color import rgb_2_hue
##
##vis = CubeRGB.id(70)
##for i in range(0, vis.values.shape[0]):
##    for j in range(0, vis.values.shape[1]):
##        point_hue = rgb_2_hue(vis.values[i, j])
##        if not (point_hue > 169 and point_hue < 171):
##            vis.alpha[i, j] = 0
##vis.show()

### visualise cubic_basis_shaper
##from lib.RRT_Common import cubic_basis_shaper
##
##fig, ax = plt.subplots()
##plt.style.use("seaborn-v0_8-whitegrid")
##ax.spines["bottom"].set_linewidth(2)
##ax.spines["left"].set_linewidth(2)
##ax.spines["top"].set_visible(False)
##ax.spines["right"].set_visible(False)
##ax.tick_params(width=2)
##ax.grid(True)
##ax.set_xlim(-180, 180)
##ax.set_ylim(0, 1)
##
##x_axis = np.linspace(-180, 180, 720)
##y_axis = []
##w = 135
##for x in x_axis:
##    hue_weight = cubic_basis_shaper(x, w)
##    y_axis.append(hue_weight)
##print(cubic_basis_shaper(0, 135))
##y_axis = np.array(y_axis)
##plt.plot(x_axis, y_axis)
##plt.show()
##

# Visualise red modifier module
from lib.RRT_Common import (  # noqa: E402
    cubic_basis_shaper,
    center_hue,
    RRT_RED_HUE,
    RRT_RED_PIVOT,
    RRT_RED_SCALE,
    RRT_RED_WIDTH,
)
from lib.utilities_color import rgb_2_hue

vis = CubeRGB.id(60)
for i in range(0, vis.values.shape[0]):
    for j in range(0, vis.values.shape[1]):
        aces = np.copy(vis.values[i, j, :])
        saturation = rgb_2_saturation(aces)
        hue = rgb_2_hue(aces)
        centeredHue = center_hue(hue, RRT_RED_HUE)
        hueWeight = cubic_basis_shaper(centeredHue, RRT_RED_WIDTH)

        aces[0] = aces[0] + hueWeight * saturation * (RRT_RED_PIVOT - aces[0]) * (
            1.0 - RRT_RED_SCALE
        )
        if not hue == 0:
            vis.alpha[i, j] = 0
        vis.values[i, j, :] = aces
        # vis.points_original_color[i, j, :] = np.array([hueWeight, 0, 0])
vis.show()
