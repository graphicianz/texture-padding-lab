# ============================================================
# uv_pad_nearest_with_distance_blur.py
# ============================================================

import cv2
import math
import numpy as np


# ============================================================
# UTILS
# ============================================================

def to_min_odd(n):
    """
    Convert number (int or float) to nearest odd integer
    not greater than n. Minimum allowed value is 3.
    Python 2 / 3 compatible.
    """
    if n < 3:
        return 3

    v = int(math.floor(n))
    if v % 2 == 1:
        return v
    return v - 1


def gaussian_kernel(radius, sigma=None):
    """
    Create a 2D Gaussian kernel.

    radius : int
        Kernel radius (kernel size = 2*radius + 1)
    sigma : float or None
        Standard deviation (default = radius / 2)

    return : np.ndarray (float32)
        Normalized kernel (sum = 1)
    """
    if radius <= 0:
        raise ValueError("radius must be > 0")

    if sigma is None:
        sigma = radius / 2.0

    size = radius * 2 + 1
    kernel = np.zeros((size, size), dtype=np.float32)

    for y in range(size):
        for x in range(size):
            dy = y - radius
            dx = x - radius
            kernel[y, x] = math.exp(
                -(dx * dx + dy * dy) / (2.0 * sigma * sigma)
            )

    kernel /= kernel.sum()
    return kernel


# ============================================================
# CORE : UV PAD (VORONOI)
# ============================================================

def uv_pad_nearest(img_rgba, pad_radius):
    """
    UV padding by copying color from nearest island pixel
    (Voronoi-style, no blur)
    """
    rgb = img_rgba[:, :, :3]
    alpha = img_rgba[:, :, 3]

    # island mask: 1 = island, 0 = background
    mask = (alpha > 0).astype(np.uint8)

    # distance to nearest island pixel
    dist, labels = cv2.distanceTransformWithLabels(
        1 - mask,
        cv2.DIST_L2,
        5,
        labelType=cv2.DIST_LABEL_PIXEL,
    )

    # map: label -> representative island pixel
    island_coords = np.column_stack(np.where(mask == 1))
    label_to_coord = {}

    for y, x in island_coords:
        label = labels[y, x]
        if label not in label_to_coord:
            label_to_coord[label] = (y, x)

    out = rgb.copy()

    ys, xs = np.where((alpha == 0) & (dist <= pad_radius))
    for y, x in zip(ys, xs):
        label = labels[y, x]
        coord = label_to_coord.get(label)
        if coord is None:
            continue

        iy, ix = coord
        out[y, x] = rgb[iy, ix]

    alpha_out = np.full_like(alpha, 255)
    return np.dstack([out, alpha_out]), mask, dist


# ============================================================
# CORE : DISTANCE-AWARE BLUR
# ============================================================

def blur_pixel(y, x, img, radius, kernel):
    """
    Blur a single pixel using given kernel
    """
    h, w, _ = img.shape

    acc = np.zeros(3, dtype=np.float32)
    weight_sum = 0.0

    for ky in range(-radius, radius + 1):
        iy = y + ky
        if iy < 0 or iy >= h:
            continue

        for kx in range(-radius, radius + 1):
            ix = x + kx
            if ix < 0 or ix >= w:
                continue

            wgt = kernel[ky + radius, kx + radius]
            acc += img[iy, ix, :3] * wgt
            weight_sum += wgt

    if weight_sum > 0:
        acc /= weight_sum

    return acc.astype(np.uint8)


def blur_with_distance(img_padding, mask, dist, max_radius=500):
    """
    Apply distance-aware blur to padding pixels only
    """
    h, w, _ = img_padding.shape
    out = img_padding.copy()

    kernel_cache = {}

    for y in range(h):
        for x in range(w):
            d = dist[y, x]

            # island pixel → skip
            if d <= 0:
                continue

            # limit blur strength
            r = int(min(d, max_radius))
            r = to_min_odd(r)

            if r <= 0:
                continue

            if r not in kernel_cache:
                kernel_cache[r] = gaussian_kernel(r)

            kernel = kernel_cache[r]

            out[y, x, :3] = blur_pixel(
                y, x,
                img_padding,
                r,
                kernel,
            )

    return out


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    INPUT_PATH = r"N:\VersionControl\Github_sourcetree2\texture-padding-lab\data\input\input3.png"
    OUTPUT_PATH = r"N:\VersionControl\Github_sourcetree2\texture-padding-lab\data\output\output5.png"

    img = cv2.imread(INPUT_PATH, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError("Failed to load image")

    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    padded, mask, dist = uv_pad_nearest(
        img_rgba=img,
        pad_radius=64,
    )

    result = blur_with_distance(
        img_padding=padded,
        mask=mask,
        dist=dist,
        max_radius=15,   # ปรับความแรงได้
    )

    cv2.imwrite(OUTPUT_PATH, result)
