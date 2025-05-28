import numpy as np
from .dtype import f32

TINY = f32(1e-10)


# ACES SMPTE STANDARD COMPLIANT DEFINITION OF MATRICES
ACES_AP0_TO_XYZ = f32(
    np.array(
        [
            [0.9525523959, 0.0000000000, 0.0000936786],
            [0.3439664498, 0.7281660966, -0.0721325464],
            [0.0000000000, 0.0000000000, 1.0088251844],
        ]
    )
)

ACES_XYZ_TO_AP0 = f32(np.linalg.inv(ACES_AP0_TO_XYZ))

AP0_to_AP1_MATRIX = f32(
    np.array(
        [
            [1.4514393161, -0.2365107469, -0.2149285693],
            [-0.0765537734, 1.1762296998, -0.0996759264],
            [0.0083161484, -0.0060324498, 0.9977163014],
        ]
    )
)

AP1_to_AP0_MATRIX = f32(np.linalg.inv(AP0_to_AP1_MATRIX))

AP1_2_XYZ_MAT = f32(ACES_AP0_TO_XYZ @ AP1_to_AP0_MATRIX)

XYZ_2_AP1_MAT = f32(np.linalg.inv(AP1_2_XYZ_MAT))

#
AP1_RGB2Y = f32(
    np.array([AP1_2_XYZ_MAT[1][0], AP1_2_XYZ_MAT[1][1], AP1_2_XYZ_MAT[1][2]])
)


def rgb_2_saturation(rgb: np.ndarray) -> float:
    max_rgb = np.maximum(np.max(rgb), TINY)
    min_rgb = np.maximum(np.min(rgb), TINY)
    denom = np.maximum(max_rgb, f32(1e-2))
    return f32((max_rgb - min_rgb) / denom)
