from pathlib import Path
import argparse

import cv2
import numpy as np


# ============================================================
# CORE
# ============================================================

def uv_pad_nearest(
    img_rgba: np.ndarray,
    pad_radius: int,
) -> np.ndarray:
    """
    UV padding by copying color from nearest island pixel
    (Voronoi-style, no blur, no diffusion)
    """
    rgb = img_rgba[:, :, :3]
    alpha = img_rgba[:, :, 3]

    # island mask: 1 = island, 0 = background
    mask = (alpha > 0).astype(np.uint8)

    # distance to nearest island pixel (in pixel units)
    dist, labels = cv2.distanceTransformWithLabels(
        1 - mask,
        cv2.DIST_L2,
        5,
        labelType=cv2.DIST_LABEL_PIXEL,
    )

    # map: label -> representative island pixel coordinate
    island_coords = np.column_stack(np.where(mask == 1))
    label_to_coord = {}

    for y, x in island_coords:
        label = labels[y, x]
        if label not in label_to_coord:
            label_to_coord[label] = (y, x)

    out = rgb.copy()

    # fill padding region
    ys, xs = np.where((alpha == 0) & (dist <= pad_radius))
    for y, x in zip(ys, xs):
        label = labels[y, x]
        coord = label_to_coord.get(label)
        if coord is None:
            continue

        iy, ix = coord
        out[y, x] = rgb[iy, ix]

    # output with solid alpha
    alpha_out = np.full_like(alpha, 255)
    return np.dstack([out, alpha_out])


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="UV padding using nearest island pixels"
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        type=Path,
        help="Input RGBA image",
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        type=Path,
        help="Output image path",
    )
    parser.add_argument(
        "--pad-radius",
        type=int,
        default=64,
        help="Padding radius in pixels",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    img = cv2.imread(str(args.input), cv2.IMREAD_UNCHANGED)
    if img is None or img.ndim != 3 or img.shape[2] != 4:
        raise RuntimeError("Input image must be RGBA")

    result = uv_pad_nearest(
        img_rgba=img,
        pad_radius=args.pad_radius,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), result)


if __name__ == "__main__":
    main()
