from typing import Tuple

import numpy as np


def get_simcc_maximum(
    simcc_x: np.ndarray,
    simcc_y: np.ndarray,
    simcc_z: np.ndarray,
    apply_softmax: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Get maximum response location and value from 3D SimCC representations."""
    assert simcc_x.ndim in (2, 3)
    assert simcc_x.ndim == simcc_y.ndim == simcc_z.ndim

    if simcc_x.ndim == 3:
        n, k, _ = simcc_x.shape
        simcc_x = simcc_x.reshape(n * k, -1)
        simcc_y = simcc_y.reshape(n * k, -1)
        simcc_z = simcc_z.reshape(n * k, -1)
    else:
        n = None

    if apply_softmax:
        simcc_x = simcc_x - np.max(simcc_x, axis=1, keepdims=True)
        simcc_y = simcc_y - np.max(simcc_y, axis=1, keepdims=True)
        simcc_z = simcc_z - np.max(simcc_z, axis=1, keepdims=True)
        ex, ey, ez = np.exp(simcc_x), np.exp(simcc_y), np.exp(simcc_z)
        simcc_x = ex / np.sum(ex, axis=1, keepdims=True)
        simcc_y = ey / np.sum(ey, axis=1, keepdims=True)
        simcc_z = ez / np.sum(ez, axis=1, keepdims=True)

    x_locs = np.argmax(simcc_x, axis=1)
    y_locs = np.argmax(simcc_y, axis=1)
    z_locs = np.argmax(simcc_z, axis=1)
    locs = np.stack((x_locs, y_locs, z_locs), axis=-1).astype(np.float32)

    max_val_x = np.amax(simcc_x, axis=1)
    max_val_y = np.amax(simcc_y, axis=1)
    mask = max_val_x > max_val_y
    max_val_x[mask] = max_val_y[mask]
    vals = max_val_x
    locs[vals <= 0.0] = -1

    if n is not None:
        locs = locs.reshape(n, k, 3)
        vals = vals.reshape(n, k)

    return locs, vals
