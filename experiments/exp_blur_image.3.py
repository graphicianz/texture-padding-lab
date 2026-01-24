# ============================================================
# uv_pad_nearest_with_distance_blur_final.py
# ============================================================

import cv2
import math
import numpy as np


# ============================================================
# GAUSSIAN KERNEL
# ============================================================

def gaussian_kernel(radius, sigma):
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
# UV PAD (VORONOI)
# ============================================================

def uv_pad_nearest(img_rgba, pad_radius):
    rgb = img_rgba[:, :, :3]
    alpha = img_rgba[:, :, 3]

    mask = (alpha > 0).astype(np.uint8)

    dist, labels = cv2.distanceTransformWithLabels(
        1 - mask,
        cv2.DIST_L2,
        5,
        labelType=cv2.DIST_LABEL_PIXEL,
    )

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
    return np.dstack([out, alpha_out]), dist


# ============================================================
# BLUR CORE
# ============================================================

def blur_pixel(y, x, img, kernel, radius):
    h, w, _ = img.shape
    acc = np.zeros(3, dtype=np.float32)
    wsum = 0.0

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
            wsum += wgt

    if wsum > 0:
        acc /= wsum

    return acc.astype(np.uint8)


# ============================================================
# DISTANCE-AWARE BLUR (FINAL)
# ============================================================

def progressive_blur_with_distance(img_padding, dist,
                                   kernel_radius=2,
                                   sigma=1.0,
                                   max_steps=20):

    h, w, _ = img_padding.shape
    out = img_padding.copy()

    kernel = gaussian_kernel(kernel_radius, sigma)

    # max distance ที่จะใช้ normalize
    d_max = dist.max()
    if d_max <= 0:
        return out

    for y in range(h):
        for x in range(w):
            d = dist[y, x]
            if d <= 0:
                continue

            # 🔑 จำนวนรอบ blur แปรตามระยะ
            steps = int((d / d_max) * max_steps)
            if steps <= 0:
                continue

            color = out[y, x, :3].astype(np.float32)

            for _ in range(steps):
                color = blur_pixel(
                    y, x,
                    out,
                    kernel,
                    kernel_radius,
                ).astype(np.float32)

                out[y, x, :3] = color.astype(np.uint8)

    return out



# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    INPUT_PATH = r"N:\VersionControl\Github_sourcetree2\texture-padding-lab\data\input\input6.png"
    OUTPUT_PATH = r"N:\VersionControl\Github_sourcetree2\texture-padding-lab\data\output\output_final.png"

    img = cv2.imread(INPUT_PATH, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError("Failed to load image")

    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    padded, dist = uv_pad_nearest(
        img_rgba=img,
        pad_radius=256,
    )

    result = progressive_blur_with_distance(
        img_padding=padded,
        dist=dist,
        kernel_radius=2,
        sigma=1.2,
        max_steps=25,  # 🔥 ตัวนี้คือ “ความฟุ้ง”
    )

    cv2.imwrite(OUTPUT_PATH, result)
