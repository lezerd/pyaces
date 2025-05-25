import numpy as np

TINY = 1e-10

AP0_to_AP1_MATRIX = np.array(
    [
        [1.4514393161, -0.2365107469, -0.2149285693],
        [-0.0765537734, 1.1762296998, -0.0996759264],
        [0.0083161484, -0.0060324498, 0.9977163014],
    ]
)

AP1_to_AP0_MATRIX = np.linalg.inv(AP0_to_AP1_MATRIX)

AP1_2_XYZ_MAT = np.array(
    [
        [0.6624541811, 0.1340042065, 0.1561876870],
        [0.2722287168, 0.6740817658, 0.0536895174],
        [0.0055746495, 0.0040607335, 0.9949223160],
    ]
)

XYZ_2_AP1_MAT = np.linalg.inv(AP1_2_XYZ_MAT)

AP1_RGB2Y = np.array([AP1_2_XYZ_MAT[0][1], AP1_2_XYZ_MAT[1][1], AP1_2_XYZ_MAT[2][1]])


def rgb_2_saturation(rgb: np.ndarray) -> float:
    max_rgb = max(np.max(rgb), TINY)
    min_rgb = max(np.min(rgb), TINY)
    denom = max(max_rgb, 1e-2)
    return (max_rgb - min_rgb) / denom
