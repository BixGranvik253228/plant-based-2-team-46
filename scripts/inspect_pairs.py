"""
inspect_pairs.py — Browse original + prediction image pairs side-by-side.

Usage:
    python3.8 scripts/inspect_pairs.py               # random sample from all cameras
    python3.8 scripts/inspect_pairs.py --cam 080313436-101664
    python3.8 scripts/inspect_pairs.py --date 2025-09-15
    python3.8 scripts/inspect_pairs.py --save        # save grid to docs/inspection_grid.jpg
"""

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

VDB = Path(__file__).parent.parent / "vdb"
DOCS = Path(__file__).parent.parent / "docs"

# Existing classifier colors (BGR)
GREEN = (0, 255, 0)   # likely: potato (size class)
BLUE  = (255, 0, 0)   # likely: contaminant / secondary class

def count_color(img, color, tol=40):
    lo = np.array([max(0, c - tol) for c in color], np.uint8)
    hi = np.array([min(255, c + tol) for c in color], np.uint8)
    return int(cv2.inRange(img, lo, hi).sum() / 255)


def find_pairs(cam_filter=None, date_filter=None, n=20):
    pairs = []
    cams = [VDB / cam_filter] if cam_filter else list(VDB.iterdir())
    for cam in cams:
        if not cam.is_dir():
            continue
        for pred in cam.rglob("*-prediction-*.jpg"):
            if date_filter and date_filter not in pred.name:
                continue
            orig = pred.parent / pred.name.replace("-prediction-", "-picture-")
            if orig.exists():
                pairs.append((orig, pred))
    random.shuffle(pairs)
    return pairs[:n]


def make_grid(pairs, cols=3):
    cells = []
    for orig_path, pred_path in pairs:
        img_o = cv2.imread(str(orig_path))
        img_p = cv2.imread(str(pred_path))
        if img_o is None or img_p is None:
            continue

        g_orig = count_color(img_o, GREEN)
        b_orig = count_color(img_o, BLUE)
        g_pred = count_color(img_p, GREEN)
        b_pred = count_color(img_p, BLUE)

        h, w = img_o.shape[:2]
        target_w = 640
        scale = target_w / w
        img_o = cv2.resize(img_o, (target_w, int(h * scale)))
        img_p = cv2.resize(img_p, (target_w, int(h * scale)))

        th = 24
        label_o = np.zeros((th, target_w, 3), np.uint8)
        label_p = np.zeros((th, target_w, 3), np.uint8)
        ts = orig_path.name.split("-picture-")[0]
        cv2.putText(label_o, f"ORIGINAL  {ts}", (4, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(label_p,
                    f"PREDICTION  green={g_pred}px  blue={b_pred}px",
                    (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        pair_img = np.vstack([
            label_o, img_o,
            label_p, img_p,
        ])
        cells.append(pair_img)

    if not cells:
        print("No pairs found.")
        sys.exit(1)

    # Pad to same height
    max_h = max(c.shape[0] for c in cells)
    padded = []
    for c in cells:
        pad = max_h - c.shape[0]
        if pad:
            c = np.vstack([c, np.zeros((pad, c.shape[1], 3), np.uint8)])
        padded.append(c)

    rows = []
    for i in range(0, len(padded), cols):
        row_cells = padded[i:i + cols]
        while len(row_cells) < cols:
            row_cells.append(np.zeros_like(row_cells[0]))
        rows.append(np.hstack(row_cells))
    return np.vstack(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam", default=None)
    parser.add_argument("--date", default=None)
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    pairs = find_pairs(cam_filter=args.cam, date_filter=args.date, n=args.n)
    print(f"Found {len(pairs)} pairs")

    grid = make_grid(pairs, cols=2)

    if args.save:
        out = DOCS / "inspection_grid.jpg"
        cv2.imwrite(str(out), grid)
        print(f"Saved: {out}")
    else:
        cv2.imshow("Original vs Prediction — press any key to close", grid)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
