"""
grader.py
---------
Grades each detected potato by size and defect score,
assigns A / B / C placement tier, and draws elliptical overlays
matching the style of the existing reference classifier.

Calibrated for a 1280x360 overhead conveyor frame.
From the reference image, visible potato sizes range from ~30x35px
to ~75x90px, giving the following frame-relative thresholds below.

    Grade A → Top layer     (largest, cleanest)
    Grade B → Middle layer
    Grade C → Bottom layer  (smallest or most defective)
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import numpy as np
import cv2


# ─── Size calibration for 1280x360 frame ─────────────────────────────────────
# From the reference image, potato pixel dims range ~30–90px.
# Expressed as fraction of total frame area (1280*360 = 460,800 px):
#   Tiny  ~30x35  =  1,050 px  → 0.23%
#   Small ~45x50  =  2,250 px  → 0.49%
#   Med   ~55x65  =  3,575 px  → 0.78%
#   Large ~70x85  =  5,950 px  → 1.29%
#
# Adjust these two values if your camera height / lens changes:
SMALL_POTATO_RATIO = 0.0023   # below this → size score 0  (~32x33px)
LARGE_POTATO_RATIO = 0.0140   # above this → size score 1  (~80x80px)

# Grade cutoffs (combined score 0–1)
GRADE_A_THRESHOLD = 0.70
GRADE_B_THRESHOLD = 0.40

# Drawing palette  (BGR)
GRADE_COLORS = {
    "A": (34,  200,  80),    # green  — top layer
    "B": (30,  170, 240),    # amber  — middle layer
    "C": (50,   50, 220),    # red    — bottom layer
}


# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PotatoGrade:
    grade:         str                      # 'A', 'B', 'C'
    placement:     str                      # human-readable
    score:         float                    # 0–1 combined
    size_score:    float                    # 0–1
    defect_score:  float                    # 0–1
    ellipse_wh:    Tuple[int, int]          # (width_px, height_px) of fitted ellipse
    ellipse_center: Tuple[int, int]         # (cx, cy)
    ellipse_angle:  float                   # rotation degrees
    bbox:          Tuple[int,int,int,int]   # original (x1,y1,x2,y2)

    def summary(self) -> str:
        w, h = self.ellipse_wh
        return (
            f"Grade {self.grade} | {self.placement} | "
            f"score={self.score:.2f}  size={self.size_score:.2f}  "
            f"defect={self.defect_score:.2f}  dims={w}x{h}px"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Size scoring
# ─────────────────────────────────────────────────────────────────────────────

def fit_ellipse_to_bbox(
    bbox: Tuple[int,int,int,int],
    mask_crop: Optional[np.ndarray] = None,
) -> Tuple[Tuple[int,int], Tuple[int,int], float]:
    """
    Returns ((cx, cy), (width_px, height_px), angle_deg) of the best-fit ellipse.

    If a binary mask crop is provided (same size as bbox region), we fit an
    ellipse to the actual potato contour — more accurate than the bbox.
    Falls back to the bbox inscribed ellipse if the mask is unavailable or
    the contour fit fails.
    """
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    cx = x1 + bw // 2
    cy = y1 + bh // 2

    if mask_crop is not None:
        contours, _ = cv2.findContours(mask_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours and len(max(contours, key=cv2.contourArea)) >= 5:
            c = max(contours, key=cv2.contourArea)
            # Shift contour coords back to full-frame space
            c = c + np.array([[[x1, y1]]])
            (ecx, ecy), (ew, eh), angle = cv2.fitEllipse(c)
            return (int(ecx), int(ecy)), (int(ew), int(eh)), float(angle)

    # Fallback: inscribed ellipse from bbox
    return (cx, cy), (bw, bh), 0.0


def score_size(ellipse_w: int, ellipse_h: int, frame_area: int) -> float:
    """
    Score potato size using fitted ellipse area (pi/4 * w * h).
    Normalised to [0, 1] based on SMALL/LARGE thresholds.
    """
    ellipse_area = (np.pi / 4) * ellipse_w * ellipse_h
    ratio  = ellipse_area / frame_area
    score  = (ratio - SMALL_POTATO_RATIO) / (LARGE_POTATO_RATIO - SMALL_POTATO_RATIO)
    return float(np.clip(score, 0.0, 1.0))


def score_defects(defect_confidence: float) -> float:
    """Higher defect confidence → lower quality score."""
    return float(np.clip(1.0 - defect_confidence, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# Per-potato grading
# ─────────────────────────────────────────────────────────────────────────────

def grade_potato(
    bbox: Tuple[int,int,int,int],
    frame_w: int,
    frame_h: int,
    defect_confidence: float = 0.0,
    mask_crop: Optional[np.ndarray] = None,
    size_weight: float = 0.5,
    defect_weight: float = 0.5,
) -> PotatoGrade:
    """
    Grade a single detected potato.

    Args:
        bbox              : (x1, y1, x2, y2) in full-frame pixels
        frame_w, frame_h  : full frame dimensions
        defect_confidence : 0.0 = clean, 1.0 = clearly defective
        mask_crop         : optional binary mask of the potato within its bbox
                            (same H×W as the bbox region). Enables contour fitting
                            for more accurate size measurement.
        size_weight       : fraction of final score from size  (default 0.5)
        defect_weight     : fraction of final score from defect (default 0.5)
    """
    center, (ew, eh), angle = fit_ellipse_to_bbox(bbox, mask_crop)
    f_area   = frame_w * frame_h
    s_score  = score_size(ew, eh, f_area)
    d_score  = score_defects(defect_confidence)
    combined = size_weight * s_score + defect_weight * d_score

    if combined >= GRADE_A_THRESHOLD:
        grade, placement = "A", "Top layer"
    elif combined >= GRADE_B_THRESHOLD:
        grade, placement = "B", "Middle layer"
    else:
        grade, placement = "C", "Bottom layer"

    return PotatoGrade(
        grade=grade,
        placement=placement,
        score=round(combined, 3),
        size_score=round(s_score, 3),
        defect_score=round(d_score, 3),
        ellipse_wh=(ew, eh),
        ellipse_center=center,
        ellipse_angle=angle,
        bbox=bbox,
    )


def grade_all(
    detections: List[Dict],
    frame_w: int,
    frame_h: int,
) -> List[Tuple[Dict, PotatoGrade]]:
    """
    Grade all detected potatoes and return them sorted A→C
    (first entry = Grade A = goes on top of the stack).

    detections : list of dicts with keys:
        'bbox'              → (x1, y1, x2, y2)
        'confidence'        → float
        'defect_confidence' → float (0.0 if not available)
        'mask_crop'         → optional np.ndarray binary mask
    """
    results = []
    for det in detections:
        g = grade_potato(
            bbox=det["bbox"],
            frame_w=frame_w,
            frame_h=frame_h,
            defect_confidence=det.get("defect_confidence", 0.0),
            mask_crop=det.get("mask_crop", None),
        )
        results.append((det, g))

    results.sort(key=lambda x: x[1].score, reverse=True)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Drawing — matches the style of the reference classifier
# ─────────────────────────────────────────────────────────────────────────────

def draw_grade(frame: np.ndarray, graded: List[Tuple[Dict, PotatoGrade]]) -> np.ndarray:
    """
    Draw elliptical overlays with grade labels on the frame,
    matching the circle + WxH label style of the reference system.

    Colour coding:
        Grade A (top)    → green
        Grade B (middle) → blue
        Grade C (bottom) → red
    """
    for det, g in graded:
        color  = GRADE_COLORS[g.grade]
        cx, cy = g.ellipse_center
        ew, eh = g.ellipse_wh

        # Filled translucent ellipse
        overlay = frame.copy()
        cv2.ellipse(
            overlay,
            (cx, cy),
            (max(1, ew // 2), max(1, eh // 2)),
            g.ellipse_angle,
            0, 360,
            color,
            -1,
        )
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

        # Ellipse outline
        cv2.ellipse(
            frame,
            (cx, cy),
            (max(1, ew // 2), max(1, eh // 2)),
            g.ellipse_angle,
            0, 360,
            color,
            2,
        )

        # WxH label inside the ellipse (matches reference system exactly)
        dim_label  = f"{ew}x{eh}"
        grade_label = f"Grade {g.grade}"
        font       = cv2.FONT_HERSHEY_SIMPLEX

        (dw, dh), _ = cv2.getTextSize(dim_label, font, 0.38, 1)
        (gw, gh), _ = cv2.getTextSize(grade_label, font, 0.40, 1)

        # Dim label centred in ellipse
        cv2.putText(frame, dim_label,
                    (cx - dw // 2, cy + 4),
                    font, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

        # Grade label just above dim label
        cv2.putText(frame, grade_label,
                    (cx - gw // 2, cy - dh - 2),
                    font, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

    return frame


def placement_summary(graded: List[Tuple[Dict, PotatoGrade]]) -> Dict[str, int]:
    """Return count of potatoes per placement layer."""
    counts: Dict[str, int] = {"Top layer": 0, "Middle layer": 0, "Bottom layer": 0}
    for _, g in graded:
        counts[g.placement] += 1
    return counts


def size_stats(graded: List[Tuple[Dict, PotatoGrade]]) -> Dict:
    """Return basic size statistics across all graded potatoes."""
    if not graded:
        return {}
    areas = [(g.ellipse_wh[0] * g.ellipse_wh[1]) for _, g in graded]
    wvals = [g.ellipse_wh[0] for _, g in graded]
    hvals = [g.ellipse_wh[1] for _, g in graded]
    return {
        "count":        len(graded),
        "avg_w_px":     int(np.mean(wvals)),
        "avg_h_px":     int(np.mean(hvals)),
        "avg_area_px2": int(np.mean(areas)),
        "max_area_px2": int(np.max(areas)),
        "min_area_px2": int(np.min(areas)),
    }
