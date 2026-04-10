"""
pipeline.py  (SAHI edition)
---------------------------
Real-time potato harvester pipeline using SAHI (Slicing Aided Hyper Inference)
for significantly better detection of small rocks, sticks, and debris.

How SAHI works here:
  Instead of running YOLOv8 on the full frame at once, SAHI cuts the frame
  into overlapping tiles (e.g. 512x512), runs the model on every tile, then
  merges all results back into full-frame coordinates with NMS. A rock that
  was 20x20 pixels in the full frame might be 80x80 in a tile — much easier
  to detect.

Usage:
    python pipeline.py --model best.pt
    python pipeline.py --model best.pt --source /dev/video0 --scale /dev/ttyUSB0
    python pipeline.py --model best.pt --slice-size 640 --overlap 0.25

Performance note:
    SAHI adds latency (~2-4x vs single-pass). On a Jetson Orin this is
    typically still 8-15 fps, which is fine for conveyor speeds.
    Press T at runtime to toggle SAHI on/off for comparison.

Controls:
    Q → quit    S → snapshot    R → reset stats    T → toggle SAHI on/off
"""

import cv2
import numpy as np
import time
import argparse
from collections import deque
from typing import Optional, List, Dict, Tuple

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction, get_prediction
from grader import grade_all, draw_grade, placement_summary


# ─── Class names (must match your dataset.yaml) ──────────────────────────────
POTATO_CLASS    = "potato"
FOREIGN_CLASSES = {"rock", "foreign_object", "debris", "stone"}

# ─── Alert threshold ─────────────────────────────────────────────────────────
ALERT_FOREIGN_PER_FRAME = 3


# ─────────────────────────────────────────────────────────────────────────────
# Scale sensor
# ─────────────────────────────────────────────────────────────────────────────

class ScaleSensor:
    """Reads total weight from a serial-connected load cell / scale."""

    def __init__(self, port: Optional[str], baud: int = 9600):
        self._ser = None
        if port:
            try:
                import serial
                self._ser = serial.Serial(port, baud, timeout=0.5)
                print(f"Scale connected on {port}")
            except Exception as e:
                print(f"[Warning] Scale not available: {e}")

    def read_kg(self) -> Optional[float]:
        if self._ser is None:
            return None
        try:
            raw   = self._ser.readline().decode("utf-8").strip()
            value = float("".join(c for c in raw if c.isdigit() or c == "."))
            # return value / 1000  # uncomment if scale outputs grams
            return value
        except Exception:
            return None

    def close(self):
        if self._ser:
            self._ser.close()


# ─────────────────────────────────────────────────────────────────────────────
# Weight estimator
# ─────────────────────────────────────────────────────────────────────────────

class WeightEstimator:
    """
    Estimates potato weight as a fraction of the total scale reading.

    Tracks (potato bbox area / all detected bbox area) over a rolling
    window of frames, then multiplies by the live scale reading.
    """

    def __init__(self, window: int = 60):
        self._ratios: deque = deque(maxlen=window)

    def update(self, potatoes: List[Dict], all_dets: List[Dict]):
        total  = sum(self._area(d["bbox"]) for d in all_dets)
        potato = sum(self._area(d["bbox"]) for d in potatoes)
        self._ratios.append((potato / total) if total > 0 else 1.0)

    @property
    def potato_ratio(self) -> float:
        return float(np.mean(self._ratios)) if self._ratios else 1.0

    def estimate(self, total_kg: Optional[float]) -> Optional[float]:
        return round(total_kg * self.potato_ratio, 2) if total_kg is not None else None

    @staticmethod
    def _area(bbox: Tuple) -> int:
        x1, y1, x2, y2 = bbox
        return max(0, (x2 - x1) * (y2 - y1))


# ─────────────────────────────────────────────────────────────────────────────
# SAHI detector wrapper
# ─────────────────────────────────────────────────────────────────────────────

class SAHIDetector:
    """
    Wraps a YOLOv8 model with SAHI sliced inference.

    SAHI tiles the frame into overlapping patches, runs the model on each,
    then merges results with NMS — giving much better recall on small objects
    like rocks and sticks mixed in with potatoes.
    """

    def __init__(
        self,
        model_path: str,
        conf: float = 0.45,
        slice_w: int = 480,
        slice_h: int = 360,
        overlap: float = 0.25,
        device: str = "",
    ):
        print(f"Loading model: {model_path}")
        self.detection_model = AutoDetectionModel.from_pretrained(
            model_type="yolov8",
            model_path=model_path,
            confidence_threshold=conf,
            device=device or "cpu",
        )
        self.slice_w  = slice_w
        self.slice_h  = slice_h
        self.overlap  = overlap
        self.use_sahi = True

    def predict(self, frame: np.ndarray) -> List[Dict]:
        """
        Run inference on a BGR numpy array.
        Returns list of dicts: {bbox, class_name, confidence, defect_confidence}
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.use_sahi:
            result = get_sliced_prediction(
                image=rgb,
                detection_model=self.detection_model,
                slice_height=self.slice_h,
                slice_width=self.slice_w,
                overlap_height_ratio=self.overlap,
                overlap_width_ratio=self.overlap,
                verbose=0,
            )
        else:
            # Single-pass fallback — faster but misses small objects
            result = get_prediction(
                image=rgb,
                detection_model=self.detection_model,
                verbose=0,
            )

        detections = []
        for pred in result.object_prediction_list:
            xyxy = pred.bbox.to_xyxy()
            detections.append({
                "bbox":              tuple(map(int, xyxy)),
                "class_name":        pred.category.name.lower(),
                "confidence":        float(pred.score.value),
                "defect_confidence": 0.0,
            })
        return detections

    def toggle(self):
        self.use_sahi = not self.use_sahi
        print(f"Inference mode: {'SAHI sliced' if self.use_sahi else 'single-pass'}")


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

class HarvesterPipeline:

    def __init__(
        self,
        model_path: str,
        scale_port: Optional[str] = None,
        conf: float = 0.45,
        slice_w: int = 480,
        slice_h: int = 360,
        overlap: float = 0.25,
        device: str = "",
    ):
        self.detector = SAHIDetector(model_path, conf, slice_w, slice_h, overlap, device)
        self.scale    = ScaleSensor(scale_port)
        self.weight   = WeightEstimator(window=60)

        self.frame_count   = 0
        self.total_foreign = 0
        self.session_start = time.time()
        self._grade_counts = {"A": 0, "B": 0, "C": 0}
        self._alert_frames: List[int] = []
        self._inference_ms: deque = deque(maxlen=30)

    def reset_stats(self):
        self.frame_count   = 0
        self.total_foreign = 0
        self.session_start = time.time()
        self._grade_counts = {"A": 0, "B": 0, "C": 0}
        self.weight        = WeightEstimator()
        print("Stats reset.")

    # ── Per-frame processing ─────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> Dict:
        h, w = frame.shape[:2]

        t0          = time.perf_counter()
        detections  = self.detector.predict(frame)
        self._inference_ms.append((time.perf_counter() - t0) * 1000)

        potatoes = [d for d in detections if d["class_name"] == POTATO_CLASS]
        foreign  = [d for d in detections if d["class_name"] in FOREIGN_CLASSES]

        # Grade potatoes — sorted A→C (top layer first in list)
        graded    = grade_all(potatoes, w, h)
        total_kg  = self.scale.read_kg()
        self.weight.update(potatoes, detections)
        potato_kg = self.weight.estimate(total_kg)

        self.frame_count   += 1
        self.total_foreign += len(foreign)
        for _, g in graded:
            self._grade_counts[g.grade] += 1

        alert = len(foreign) >= ALERT_FOREIGN_PER_FRAME
        if alert:
            self._alert_frames.append(self.frame_count)

        return {
            "frame":            self._draw(frame.copy(), graded, foreign,
                                           total_kg, potato_kg, alert),
            "graded_potatoes":  graded,
            "foreign_objects":  foreign,
            "total_weight_kg":  total_kg,
            "potato_weight_kg": potato_kg,
            "potato_ratio":     self.weight.potato_ratio,
            "alert":            alert,
        }

    # ── Rendering ────────────────────────────────────────────────────────────

    def _draw(self, frame, graded, foreign, total_kg, potato_kg, alert):
        h, w = frame.shape[:2]

        # Potato grade overlays (green=A, amber=B, red=C)
        draw_grade(frame, graded)

        # Foreign object overlays
        for det in foreign:
            x1, y1, x2, y2 = det["bbox"]
            label = f"{det['class_name'].upper()}  {det['confidence']:.0%}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 220), 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), (0, 0, 180), -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

        self._draw_hud(frame, graded, foreign, total_kg, potato_kg)

        if alert:
            cv2.rectangle(frame, (0, 0), (w, 42), (0, 0, 185), -1)
            cv2.putText(
                frame,
                f"  ALERT: {len(foreign)} foreign objects detected this frame!",
                (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2, cv2.LINE_AA,
            )
        return frame

    def _draw_hud(self, frame, graded, foreign, total_kg, potato_kg):
        placement = placement_summary(graded)
        elapsed   = time.time() - self.session_start
        avg_ms    = np.mean(self._inference_ms) if self._inference_ms else 0
        mode      = "SAHI" if self.detector.use_sahi else "single-pass"

        lines = [
            f"Frame {self.frame_count}  |  {elapsed:.0f}s  |  {avg_ms:.0f}ms/frame  [{mode}]",
            f"Potatoes this frame : {len(graded)}",
            f"  Grade A -> top    : {placement['Top layer']}",
            f"  Grade B -> middle : {placement['Middle layer']}",
            f"  Grade C -> bottom : {placement['Bottom layer']}",
            f"Foreign objects    : {len(foreign)}  (session: {self.total_foreign})",
            f"Potato ratio       : {self.weight.potato_ratio:.1%}",
        ]
        if total_kg is not None:
            lines += [
                f"Total weight       : {total_kg:.2f} kg",
                f"Potato weight est. : {potato_kg:.2f} kg",
            ]
        else:
            lines.append("Weight sensor      : not connected")
        lines.append("T=toggle SAHI  S=snap  R=reset  Q=quit")

        panel_h = len(lines) * 22 + 16
        overlay = frame.copy()
        cv2.rectangle(overlay, (8, 48), (402, 48 + panel_h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, frame)

        for i, line in enumerate(lines):
            y     = 48 + 16 + i * 22
            color = (175, 175, 175) if not line.startswith("  ") else (125, 125, 125)
            if "SAHI" in line and "[SAHI]" in line:
                color = (80, 210, 120)
            if line.startswith("T="):
                color = (110, 110, 110)
            cv2.putText(frame, line, (16, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Potato harvester — SAHI pipeline")
    parser.add_argument("--model",      required=True)
    parser.add_argument("--source",     default="0",         help="Camera index or video path")
    parser.add_argument("--scale",      default=None,        help="Serial port, e.g. /dev/ttyUSB0")
    parser.add_argument("--conf",       type=float, default=0.45)
    # Defaults tuned for 1280x360 frame: 480x360 tiles = ~3 cols x 1 row
    parser.add_argument("--slice-w",    type=int,   default=480,  help="SAHI tile width  px (default 480 for 1280x360)")
    parser.add_argument("--slice-h",    type=int,   default=360,  help="SAHI tile height px (default 360 for 1280x360)")
    parser.add_argument("--overlap",    type=float, default=0.25, help="SAHI tile overlap (default 0.25)")
    parser.add_argument("--device",     default="",          help="cpu / cuda:0 / mps")
    parser.add_argument("--no-sahi",    action="store_true", help="Start in single-pass mode")
    parser.add_argument("--no-display", action="store_true", help="Headless / no window")
    parser.add_argument("--save",       default=None,        help="Save output video to path")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    cap    = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {args.source}")

    fw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    writer = None
    if args.save:
        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), fps, (fw, fh))
        print(f"Saving to: {args.save}")

    pl = HarvesterPipeline(
        model_path=args.model,
        scale_port=args.scale,
        conf=args.conf,
        slice_w=args.slice_w,
        slice_h=args.slice_h,
        overlap=args.overlap,
        device=args.device,
    )
    if args.no_sahi:
        pl.detector.toggle()

    print(f"\nRunning — SAHI tiles={args.slice_w}x{args.slice_h}px  overlap={args.overlap:.0%}  conf={args.conf}")
    print("T = toggle SAHI on/off for live comparison\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = pl.process_frame(frame)

        if pl.frame_count % 30 == 0 and result["graded_potatoes"]:
            print(f"\n── Frame {pl.frame_count}  stacking order (top -> bottom) ──")
            for i, (_, g) in enumerate(result["graded_potatoes"], 1):
                print(f"  {i:2d}. {g.summary()}")
            if result["alert"]:
                print(f"  !! ALERT: {len(result['foreign_objects'])} foreign objects !!")

        if writer:
            writer.write(result["frame"])

        if not args.no_display:
            cv2.imshow("Potato Harvester", result["frame"])
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                p = f"snap_{pl.frame_count}.png"
                cv2.imwrite(p, result["frame"])
                print(f"Saved {p}")
            elif key == ord("r"):
                pl.reset_stats()
            elif key == ord("t"):
                pl.detector.toggle()

    # Summary
    elapsed = time.time() - pl.session_start
    avg_ms  = np.mean(pl._inference_ms) if pl._inference_ms else 0
    print(f"\n{'─'*50}")
    print(f"Frames     : {pl.frame_count}  ({pl.frame_count/max(elapsed,1):.1f} fps)")
    print(f"Avg latency: {avg_ms:.0f} ms/frame")
    print(f"Grade A    : {pl._grade_counts['A']}")
    print(f"Grade B    : {pl._grade_counts['B']}")
    print(f"Grade C    : {pl._grade_counts['C']}")
    print(f"Foreign    : {pl.total_foreign}  ({len(pl._alert_frames)} alert frames)")
    print(f"Potato %   : {pl.weight.potato_ratio:.1%}")
    print(f"{'─'*50}\n")

    cap.release()
    if writer:
        writer.release()
    if not args.no_display:
        cv2.destroyAllWindows()
    pl.scale.close()


if __name__ == "__main__":
    main()
