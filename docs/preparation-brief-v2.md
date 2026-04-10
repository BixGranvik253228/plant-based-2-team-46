# BrabantHack 26 — Track Plant-Based 2 (VDBorne)
## Preparation Brief v2.0 — 8-Hour Sprint Edition

---

## The Core Insight (read this first)

> **90% of the project is data-independent.** Hardware, pipeline, ISOBUS integration, AWS/Snowflake, zone mapping — all of it can be built and demo-ready before the hackathon starts. The VDBorne dataset is only needed for one thing: fine-tuning the final classification head. The pretrained models already know what organic objects, stones, and surface defects look like.

This means you walk in with a working system. You just plug in their data.

---

## Transfer Learning Models — Licensing & Availability

| Model | License | Weights free? | Pip install | Commercial? |
|---|---|---|---|---|
| **YOLOv11s** (Ultralytics) | AGPL-3.0 | Yes, auto-download | `ultralytics` | ⚠️ Must open-source product OR buy Enterprise license |
| **YOLOv8s** (Ultralytics) | AGPL-3.0 | Yes | `ultralytics` | ⚠️ Same restriction |
| **MobileNetV3-Small** (torchvision) | BSD-3-Clause | Yes | `torchvision` | ✅ Fully free |
| **ResNet-18/50** (torchvision) | BSD-3-Clause | Yes | `torchvision` | ✅ Fully free |
| **EfficientNet-B0** (torchvision) | BSD-3-Clause | Yes | `torchvision` | ✅ Fully free |
| **RT-DETR** (Baidu/PaddleDetection) | Apache 2.0 | Yes | `paddledetection` | ✅ Fully free |
| **RT-DETR** (Ultralytics version) | AGPL-3.0 | Yes | `ultralytics` | ⚠️ Same as YOLO |
| **DINOv2** (Meta) | Apache 2.0 | Yes | `torch.hub` | ✅ Fully free |
| **Florence-2** (Microsoft) | MIT | Yes | `transformers` | ✅ Fully free |
| **Grounding DINO** (IDEA Research) | Apache 2.0 | Yes | `groundingdino-py` | ✅ Fully free |
| **SAM 2** (Meta) | Apache 2.0 | Yes | `sam2` | ✅ Fully free |

### Recommended production stack (commercially safe)
- **Detection**: Grounding DINO (Apache 2.0) zero-shot → then fine-tune with RT-DETR Baidu (Apache 2.0)
- **Bruise classifier**: MobileNetV3-Small (BSD-3-Clause)
- **Segmentation** (optional): SAM2 (Apache 2.0)

For the hackathon demo YOLO is fine (AGPL is OK for a demo). For the pitch, state the production stack will use Apache/BSD models to give VDBorne a clean commercial path without licensing cost.

### Zero-shot strategy — run this in Hour 1 before any labeling

```python
# Grounding DINO — zero-shot, no training needed
# pip install groundingdino-py

from groundingdino.util.inference import load_model, load_image, predict

model = load_model("groundingdino_swint_ogc.pth", "GroundingDINO_SwinT_OGC.cfg.py")
image_source, image = load_image("conveyor_frame.jpg")

boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption="potato . stone . soil clod . damaged potato",
    box_threshold=0.35,
    text_threshold=0.25
)
# Returns bounding boxes labeled "potato", "stone", "soil clod" — no training
```

Expected zero-shot accuracy: ~60–70% on unseen conveyor images. Enough to show live results immediately. Fine-tune YOLOv11 / RT-DETR as hours pass.

- **Florence-2** (MIT): alternative zero-shot with `<OPEN_VOCABULARY_DETECTION>` task prompt
- **DINOv2** (Apache 2.0): frozen feature extractor + 5-minute linear probe — few-shot with as few as 10 labeled images per class

---

## 850nm NIR — Why It's Essential & How to Build It

### The physics

Potato skin is ~0.1–0.3mm thick. At visible wavelengths (400–700nm), light only reflects off the surface. At 850nm (near-infrared), light penetrates **1–3mm** into the flesh before scattering back out.

Bruised tissue at 1–2mm depth:
- Ruptured cells → different water distribution → different NIR scattering
- Appears as darker, irregular patch at 850nm
- **Invisible at visible wavelengths for 24–72 hours after impact**

Without NIR, you can only detect already-visible damage. With NIR, you catch it at the moment of impact.

### Hardware: exact components

| Component | Part | Price | Where |
|---|---|---|---|
| Camera | Raspberry Pi Camera Module 3 NoIR | €25 | raspberrypi.com |
| Bandpass filter | 850nm ±10nm, M12 or CS-mount | €10–15 | aliexpress / Edmund Optics |
| Illumination | 850nm IR LED ring light (60°, 12V) | €15–25 | aliexpress |
| Cable | CSI ribbon, 30cm | €3 | standard |
| **Total** | | **~€55** | |

Alternative for higher quality: **Basler dart daA1280-54uc** (industrial, global shutter) + 850nm filter = ~€400 but proper industrial spec.

### Integration in the pipeline

```
Frame capture loop (30fps):
  ├── RGB frame (global shutter camera)     → YOLOv11: detect potato/clod/stone
  └── NIR frame (NoIR camera + 850nm LED)  → synchronized capture

Per detected potato ROI:
  ├── Crop RGB region
  ├── Crop NIR region (same coordinates, calibrated homography)
  └── Stack: [B, G, R, L*, V, NIR] → 6-channel input to bruise classifier
```

### Camera synchronization

Both cameras trigger on the same GPIO pulse from the Jetson:
```python
import RPi.GPIO as GPIO  # or Jetson.GPIO
import time

TRIGGER_PIN = 17

def trigger_both_cameras():
    GPIO.output(TRIGGER_PIN, GPIO.HIGH)
    time.sleep(0.001)  # 1ms pulse
    GPIO.output(TRIGGER_PIN, GPIO.LOW)
    # Both cameras capture on rising edge
```

For the hackathon demo: software sync (capture RGB then NIR within 10ms) is sufficient since the belt moves slowly (~0.5 m/s → 5mm per 10ms).

### Why this makes your pitch unbeatable
Every other team will build an RGB-only system. You are the only team that addresses the fundamental problem — blackspot bruising is invisible at harvest. NIR is the technically correct answer. It directly justifies the 90% KPI that RGB alone cannot reliably hit.

---

## 8-Hour Execution Plan

### The night before (preparation)

**Do all of this before the hackathon starts:**

```bash
# 1. Pre-install everything
pip install ultralytics torch torchvision albumentations \
            groundingdino-py transformers supervision \
            opencv-python geopandas shapely scikit-learn \
            paho-mqtt pyserial pynmea2 python-can \
            boto3 snowflake-connector-python sqlite3 \
            roboflow flask fastapi uvicorn

# 2. Pre-download model weights (these download on first use — do it now)
python3 -c "from ultralytics import YOLO; YOLO('yolo11s.pt')"
python3 -c "from ultralytics import YOLO; YOLO('rtdetr-l.pt')"
python3 -c "import torchvision.models as m; m.mobilenet_v3_small(weights='IMAGENET1K_V1')"
python3 -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')"

# 3. Pre-download surrogate dataset
python3 -c "
from roboflow import Roboflow
rf = Roboflow(api_key='YOUR_KEY')
rf.workspace('roboflow-universe-projects').project('potato-disease-detection').version(1).download('yolov11')
"

# 4. Test AgIsoStack++ VT simulator builds correctly
git clone https://github.com/Open-Agriculture/AgIsoStack-plus-plus
cd AgIsoStack-plus-plus && cmake -B build && cmake --build build
```

**Pre-build artifacts to bring:**
- [ ] Trained surrogate model weights (`yolo11s_potato_surrogate.pt`)
- [ ] AgIsoStack++ VT simulator binary
- [ ] Snowflake schema SQL file
- [ ] AWS S3 bucket already created + IAM credentials
- [ ] SQLite schema pre-tested
- [ ] Demo video of conveyor belt (YouTube, ~30s, potato harvester belt footage)

---

### Hour-by-hour plan (8 hours)

```
Hour 1 (0:00–1:00) — SETUP + DATA
  ├── Get VDBorne dataset (images + any labels)
  ├── Understand image format, resolution, lighting conditions
  ├── Run Grounding DINO zero-shot immediately on 10 images
  │   → shows "it works" even before any labeling
  └── Start labeling in Roboflow (team member 2 does this while team member 1 codes)

Hour 2 (1:00–2:00) — INFERENCE PIPELINE
  ├── Camera capture loop (or video file input for demo)
  ├── YOLOv11 inference on conveyor frames
  ├── MQTT broker started (Mosquitto)
  └── SQLite logger connected

Hour 3 (2:00–3:00) — FINE-TUNING
  ├── If ≥50 labeled images: run YOLOv11 fine-tune (50 epochs, ~15 min on GPU)
  ├── If <50 labels: use Grounding DINO + surrogate model as demo
  └── Bruise classifier: MobileNetV3 on NIR+RGB crops (even 30 images is enough for demo)

Hour 4 (3:00–4:00) — ISOBUS DISPLAY
  ├── Start AgIsoStack++ VT simulator (PC)
  ├── Implement Object Pool (damage %, stone count, zone warning)
  ├── Wire MQTT → VT numeric value updates
  └── Screenshot/record for presentation

Hour 5 (4:00–5:00) — ZONE MAPPING
  ├── GPS track simulation (or real GPS if hardware available)
  ├── DBSCAN clustering on high-damage positions
  ├── GeoJSON zone export
  └── Matplotlib damage map visualization

Hour 6 (5:00–6:00) — AWS + SNOWFLAKE
  ├── boto3 S3 upload from SQLite sync queue
  ├── Snowflake COPY INTO from S3 stage
  └── Snowflake query → damage heatmap (optional, nice-to-have)

Hour 7 (6:00–7:00) — INTEGRATION + POLISH
  ├── End-to-end demo run: video → inference → display → zone map → S3 → Snowflake
  ├── Measure and record actual accuracy metrics
  └── Fix any broken connections

Hour 8 (7:00–8:00) — PRESENTATION
  ├── Slide deck: problem → solution → demo → business case → architecture
  ├── Practice live demo walkthrough (2 min)
  └── Prepare judge questions
```

### If things go wrong

| Problem | Fallback |
|---|---|
| No VDBorne data on Day 1 | Use surrogate pretrained model + harvester YouTube footage |
| No ISOBUS hardware | Use AgIsoStack++ PC VT simulator (identical visually) |
| Model accuracy <90% | Show learning curve — "with 2h of labeling we're at 78%, full dataset will hit 90%+" |
| No internet (can't sync to AWS) | Show local SQLite log + show S3 code — "syncs automatically when online" |
| Jetson not available | Run on laptop GPU — state Jetson as production target |

---

## What Works Without Any VDBorne Data

Everything below is **fully functional before you walk in the door**:

| Component | Status | Notes |
|---|---|---|
| Camera capture pipeline | Ready | OpenCV / GStreamer |
| NIR hardware setup | Ready | Physical build, no data needed |
| YOLOv11 + Grounding DINO | Ready | Zero-shot on any conveyor footage |
| Bruise classifier pipeline | Ready | Needs their images to hit 90% KPI |
| MQTT internal bus | Ready | No data dependency |
| SQLite offline store | Ready | Schema fixed |
| ISOBUS VT display | Ready | AgIsoStack++ PC simulator |
| GPS geotagging | Ready | NMEA / synthetic track |
| DBSCAN zone mapping | Ready | Works with any GPS + damage data |
| GeoJSON export | Ready | |
| AWS S3 sync | Ready | Bucket pre-created |
| Snowflake pipeline | Ready | Schema pre-deployed |
| ISOXML export | Ready | Standard format |

**Needs VDBorne data to complete:**

| Component | What's missing |
|---|---|
| 90%+ bruise recall KPI | Their labeled post-wash images (damaged vs intact) |
| Accurate stone/clod detection | Their specific soil/stone appearance |
| Field zone accuracy | Real GPS harvest track |
| ISOBUS terminal compatibility | Which VT model is on their harvester |

---

## Minimum Winning Demo (what judges see)

1. **Video input** — conveyor belt footage (theirs or YouTube)
2. **Live inference** — bounding boxes: potato (green), clod (brown), stone (grey), damaged potato (red)
3. **ISOBUS display** — PC VT simulator showing: `DAMAGE: 23% ⚠️ SLOW DOWN`
4. **Zone map** — field plot with two red high-risk zones identified
5. **Snowflake** — live query showing damage trend by GPS zone
6. **NIR slide** — show the same potato in RGB (looks fine) vs NIR (bruise visible)

That's a full end-to-end system in 8 hours. Everything except step 2's accuracy is data-independent.

---

## Hardware BOM (Bill of Materials)

### Prototype (hackathon)

| Item | Spec | Price |
|---|---|---|
| Edge compute | Jetson Orin Nano 8GB DevKit | €499 |
| RGB camera | OAK-D S2 (global shutter, stereo, USB3) | €299 |
| NIR camera | RPi Camera Module 3 NoIR | €25 |
| 850nm bandpass filter | M12 mount, ±10nm | €12 |
| 850nm LED ring light | 60°, 12V, ~3W | €18 |
| CAN interface | PEAK PCAN-USB | €119 |
| ISOBUS connector | Deutsch HD10-9-1939 male | €12 |
| GPS (fallback) | u-blox NEO-M8N USB | €28 |
| MicroSD | 128GB A2 | €15 |
| Power supply | 12V 5A (harvester battery) | €0 (harvester) |
| **Total** | | **~€1,027** |

### Production target

| Item | Spec | Price |
|---|---|---|
| Edge compute | Jetson Orin NX 16GB | €649 |
| RGB camera | Basler ace2 a2A1920 (global shutter, 160fps) | €380 |
| NIR camera | Basler dart daA1280 + 850nm filter | €420 |
| CAN interface | Built-in CAN on Jetson + SN65HVD230 transceiver | €8 |
| Enclosure | IP67 aluminum, DIN rail mount | €85 |
| Vibration mount | Camera isolation bracket | €40 |
| **Total** | | **~€1,582** |

ROI calculation: €1,582 hardware vs €45,000–135,000/season damage cost = **ROI in 3–10 harvest days**.

---

## Architecture Diagram (for slides)

```
┌─────────────────────────────────────────────────────────────┐
│                    HARVESTER CONVEYOR BELT                  │
│                                                             │
│  [RGB Camera]──┐    [NIR Camera 850nm]──┐                  │
│               ↓                        ↓                   │
│         ┌──────────────────────────────────┐               │
│         │      JETSON ORIN NANO 8GB        │               │
│         │                                  │               │
│         │  YOLOv11s ──→ potato/clod/stone  │               │
│         │      ↓                           │               │
│         │  MobileNetV3 ──→ intact/damaged  │               │
│         │  (RGB+L*+V+NIR 6-channel)        │               │
│         │                                  │               │
│         │  MQTT broker ←── GPS (ISOBUS)    │               │
│         │       │                          │               │
│         │  ┌────┴─────────────────┐        │               │
│         │  │   SQLite WAL store   │        │               │
│         │  └────┬─────────────────┘        │               │
│         └───────┼──────────────────────────┘               │
│                 │                                           │
│     ┌───────────┼───────────┐                              │
│     ↓           ↓           ↓                              │
│ [ISOBUS VT]  [GPS Zone   [WiFi sync]                       │
│ cab display] [Map DBSCAN] [on connect]                     │
└─────────────────────────────────────────────────┬───────────┘
                                                  │
                          ┌───────────────────────┘
                          ↓
              ┌───────────────────────┐
              │      AWS CLOUD        │
              │                       │
              │  S3: harvest data     │
              │  Lambda: COPY trigger │
              │  IoT Core: MQTT/TLS   │
              └──────────┬────────────┘
                         ↓
              ┌───────────────────────┐
              │      SNOWFLAKE        │
              │                       │
              │  harvest.detections   │
              │  harvest.field_zones  │
              │  Snowpark: analytics  │
              └──────────┬────────────┘
                         ↓
              ┌───────────────────────┐
              │  FARM MANAGEMENT SYS  │
              │  365FarmNet / JD OC   │
              │  ISOXML TaskData.zip  │
              └───────────────────────┘
```

---

## The Pitch (30 seconds)

> "Today, every harvester is a black box. Damage only shows up 48 hours later in the barn. By then it's too late — you can't go back and slow down the machine.
>
> We mount a €55 NIR camera and a Jetson on the elevator. Real-time: potato, stone, or clod — and whether that potato has a bruise forming under the skin. The cab display shows damage percentage and tells the driver to slow down — on the existing screen, no extra hardware.
>
> GPS logs every event. High-risk field zones appear on a map. The data flows to Snowflake. Next season, VDBorne knows exactly where to drive slower before they even start.
>
> €1,000 hardware. €45,000–135,000 damage prevented per machine per season."

---

## Pre-Hackathon Checklist

### Hardware to bring
- [ ] Laptop with NVIDIA GPU (for training demo)
- [ ] USB webcam (simulate camera input)
- [ ] 850nm LED (to show NIR concept physically)
- [ ] RPi Camera Module 3 NoIR (if available)
- [ ] PEAK PCAN-USB (if doing real ISOBUS)

### Software to pre-install and test
- [ ] All pip packages installed and import-tested
- [ ] YOLOv11s weights downloaded (`yolo11s.pt`)
- [ ] RT-DETR weights downloaded (`rtdetr-l.pt`)
- [ ] MobileNetV3 weights downloaded
- [ ] Grounding DINO installed and tested with one image
- [ ] AgIsoStack++ built and VT simulator runs
- [ ] Mosquitto installed and starts
- [ ] AWS credentials configured (`~/.aws/credentials`)
- [ ] Snowflake account + schema deployed
- [ ] boto3 S3 upload tested
- [ ] Jupyter notebook runs end-to-end on synthetic data

### Accounts to create
- [ ] Roboflow (free) — for labeling VDBorne images on-site
- [ ] AWS account (free tier) — S3 bucket pre-created
- [ ] Snowflake trial account — schema pre-deployed
- [ ] HuggingFace account (optional) — for Florence-2 / DINOv2

### Files to prepare
- [ ] `object_pool.iop` — ISOBUS VT UI definition (pre-built)
- [ ] `potato_belt.yaml` — YOLO dataset config
- [ ] `snowflake_schema.sql` — table creation script
- [ ] `aws_setup.sh` — S3 bucket + IAM policy creation
- [ ] Demo slide deck skeleton (8 slides max)
- [ ] 30-second harvester conveyor belt video downloaded offline
