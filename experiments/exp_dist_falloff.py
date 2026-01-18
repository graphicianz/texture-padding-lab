from pathlib import Path
import cv2
import numpy as np

# ===============================
# CONFIG
# ===============================
PAD_RADIUS = 16   # padding width in pixels

def uv_pad(img_rgba: np.ndarray) -> np.ndarray:
    h, w, _ = img_rgba.shape

    rgb = img_rgba[:, :, :3]
    alpha = img_rgba[:, :, 3]

    # island mask
    mask = (alpha > 0).astype(np.uint8)

    # distance transform
    # zero = feature (island), non-zero = background
    dist, labels = cv2.distanceTransformWithLabels(
        1 - mask,
        cv2.DIST_L2,
        5,
        labelType=cv2.DIST_LABEL_PIXEL,
    )

    # find coordinates of nearest island pixel for each label
    coords = np.column_stack(np.where(mask == 1))
    label_to_coord = {}

    for y, x in coords:
        label = labels[y, x]
        if label not in label_to_coord:
            label_to_coord[label] = (y, x)

    out = rgb.copy()

    # fill padding
    ys, xs = np.where((alpha == 0) & (dist <= PAD_RADIUS))
    for y, x in zip(ys, xs):
        label = labels[y, x]
        if label in label_to_coord:
            iy, ix = label_to_coord[label]
            ratio_from_dist = 1- (dist[y, x] / PAD_RADIUS)
            out[y, x] = (rgb[iy, ix].astype(np.float32) * ratio_from_dist).astype(np.uint8)


    # blur = cv2.GaussianBlur(out, (7, 7), 0)
    # mask_pad = (alpha == 0) & (dist <= PAD_RADIUS)
    # out[mask_pad] = blur[mask_pad]

    return np.dstack([out, np.full_like(alpha, 255)])

# ===============================
# RUN
# ===============================
if __name__ == "__main__":
    in_path = Path(r"N:\VersionControl\Github_sourcetree2\texture-padding-lab\data\input\input4.png")
    out_path = Path(r"N:\VersionControl\Github_sourcetree2\texture-padding-lab\data\output\output1.png")

    img = cv2.imread(str(in_path), cv2.IMREAD_UNCHANGED)
    out = uv_pad(img)

    cv2.imwrite(str(out_path), out)
    print("done")
