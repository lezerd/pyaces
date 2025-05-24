import numpy as np

TINY = 1e-10


def rgb_2_saturation(rgb: np.ndarray) -> float:
    max_rgb = max(np.max(rgb), TINY)
    min_rgb = max(np.min(rgb), TINY)
    denom = max(max_rgb, 1e-2)
    return (max_rgb - min_rgb) / denom
