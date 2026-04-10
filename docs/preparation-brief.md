# BrabantHack 26 — Track Plant-Based 2 (VDBorne)
## Full Preparation Brief

---

## The Challenge in One Sentence

Build a vision-based system mounted on a potato harvester that classifies potatoes/clods/stones on the conveyor belt, detects mechanical damage in real-time, maps damage to GPS field zones, displays results on the existing harvester screen, and syncs to farm management systems — all offline, all brands.

---

## Success Criteria (non-negotiable KPIs)

| Requirement | Threshold |
|---|---|
| Damaged potato detection accuracy | ≥ 90% |
| Stone and clod detection accuracy | ≥ 90% |
| Connectivity required | None (fully offline) |
| Harvester display integration | Existing screen, no extra hardware |
| FMS sync | Automatic (no USB) |
| Harvester brand compatibility | All brands |

---

## The Pitch Frame (use this verbatim)

> "We add a georeferenced damage intelligence layer at the point of harvest — the one moment where intervention (slow down, raise the web, flag the high-risk zone) is still possible — turning a black-box process into actionable, field-level quality data."

**Why it's novel**: Packhouse sorting (Tomra, Greefa, Aweta) is the existing tech — it operates post-harvest. There is no commercial on-harvester real-time damage detection product. This gap is real and open.

---

## About Van den Borne (VDBorne)

- Family farm in Reusel, Noord-Brabant. ~450 hectares, primarily starch potatoes for AVEBE.
- Run by Jacob van den Borne — the most referenced precision agriculture farmer in Europe.
- Technology stack: RTK-GPS (sub-cm accuracy), full drone fleet (NDVI + multispectral), EC mapping on all fields, variable rate application, John Deere Operations Center as FMS.
- They know yield history per 3x3m patch going back years.
- Contact for this track: Marnik van Geelen (marnik@vdbornecampus.com)

**How to use this in your pitch**: Reference their specific data assets (EC maps, NDVI layers, historical yield maps). Frame your solution as an additional data layer that connects to infrastructure they already have.

---

## Domain Vocabulary (impress the judges)

| Term | Meaning |
|---|---|
| Blackspot bruise | Internal bruise invisible at harvest, appears 24-72h later as grey-black tissue |
| Shatter bruise | Skin crack from impact — visible immediately, infection entry point |
| Skinning | Friction damage removing thin skin — causes water loss and Fusarium |
| EC mapping | Electrical conductivity scan of field — identifies clay content, moisture, compaction zones |
| High-risk zones | Sub-field areas (heavy soil, headlands, wet corners) with elevated damage probability |
| NDVI | Canopy health index from drone/satellite — also indicates harvest timing |
| RTK-GPS | Sub-centimeter GPS, used for georeferencing every detection event |
| Management zones | Sub-field areas with homogeneous soil properties |
| ISOBUS | CAN bus standard for agricultural machines (ISO 11783) |
| VT (Virtual Terminal) | The harvester cab display — receives UI via ISOBUS |
| ISOXML | File format for FMS data exchange (ISO 11783-10) |
| Lenticel | Breathing pores on potato skin — entry points for Erwinia when damaged |
| Skin set | Lignification of potato skin; incomplete skin set = higher skinning damage |
| Drop height | Key harvester parameter — reducing by 10cm can halve blackspot incidence |
| Destoning | Pre-plant bed preparation to remove stones |

---

## Technical Architecture

### Recommended Pipeline

```
Camera (global shutter RGB, 640x480, 60fps)
    |
Frame capture (OpenCV/GStreamer on Jetson)
    |
YOLOv11s — 3-class detection (potato, clod, stone)
    [TensorRT FP16, ~85fps on Jetson Orin Nano]
    |
Crop ROIs of "potato" detections
    |
MobileNetV3-Small — binary classifier (intact / damaged)
    [TensorRT INT8, ~5ms per crop]
    |
MQTT broker (Mosquitto, localhost)
    |
    +-- ISOBUS VT updater (AgIsoStack++, C++ daemon)
    |   → updates damage % on harvester display in real-time
    |
    +-- SQLite logger (WAL mode)
    |   → timestamps, GPS lat/lon, counts per frame
    |
    +-- GPS geotagger (NMEA via serial / ISOBUS PGN 65267)
    |
    +-- WiFi sync daemon
        → POST ISOXML TaskData.zip to FMS (365FarmNet / JD Ops Center)
```

### Hardware Target

| Component | Recommended | Budget Alternative |
|---|---|---|
| Edge compute | Jetson Orin Nano 8GB (~€500) | RPi 5 + Hailo-8L (~€130) |
| Camera | Global shutter USB3 industrial (Basler/FLIR) | OAK-D S2 (~€300) |
| NIR illumination | 850nm LED strip + NoIR camera mod (~€35) | None (visible damage only) |
| CAN interface | PEAK PCAN-USB (~€120) | Waveshare CAN HAT for RPi (~€25) |
| GPS | Read from ISOBUS (free) | Any NMEA USB GPS (~€30) |
| Storage | MicroSD / eMMC on Jetson | Same |

**Total prototype cost estimate: €500–900** — cite this in the pitch as a key commercial viability argument.

---

## Model Details

### Object Detection: YOLOv11s

```bash
pip install ultralytics

# Fine-tune from COCO weights
yolo train model=yolo11s.pt data=potato_belt.yaml \
  epochs=100 imgsz=640 batch=16 \
  mosaic=1.0 degrees=15 flipud=0.5 mixup=0.1

# Export for Jetson (TensorRT FP16)
yolo export model=best.pt format=engine half=True device=0
```

Classes: `0=potato`, `1=clod`, `2=stone`

**Why YOLOv11s**: NMS-free (lower latency), best Ultralytics accuracy/speed, same export pipeline as v8.

### Damage Classifier: MobileNetV3-Small

```python
import torchvision.models as models
import torch.nn as nn

model = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
model.classifier[-1] = nn.Linear(1024, 2)  # intact / damaged

# 5-channel input (BGR + L* + V) outperforms 3-channel by ~5%
# Prepend a conv layer to accept 5 channels:
model.features[0][0] = nn.Conv2d(5, 16, kernel_size=3, stride=2, padding=1, bias=False)
```

**Input trick**: Feed `BGR + L*(CIE Lab) + V(HSV)` as 5 channels. Bruises show reduced L* before visible browning.

### Accuracy Path to 90%

| Target | Achievable? | Key requirement |
|---|---|---|
| Stone/clod ≥90% | Yes (relatively easy) | 500+ labeled images, shape differences are large |
| Damaged potato ≥90% | Yes for visible damage | Good labeled dataset + 850nm NIR illumination |
| Subsurface blackspot ≥90% | No with RGB alone | Requires hyperspectral (future work — say this clearly) |

**Critical negotiation**: Ask VDBorne for the washing process images at the hackathon start. Post-wash = cleaner, more uniform skin = bruises far more visible. This is likely your highest-value data source.

---

## Data Sources

| Source | Content | How to access |
|---|---|---|
| VDBorne (provided) | Harvester camera images + washing images | Given at hackathon |
| Roboflow Universe | Potato/stone/clod conveyor datasets | roboflow.com — search "potato" |
| PDOK WFS | Dutch field boundaries (BRP) | service.pdok.nl/rvo/brpgewaspercelen/wfs/v1_0 |
| PlantVillage | Surface lesion textures (transfer learning) | Kaggle / paperswithcode |
| COCO pretrained | Starting weights for YOLOv11 | Built into Ultralytics |

### Data Augmentation (Albumentations)

Key augmentations for conveyor belt scenario:
- `MotionBlur(blur_limit=(3,15))` — simulates belt movement
- `RandomBrightnessContrast(±0.4)` — outdoor lighting variation
- `CoarseDropout(max_holes=8)` — simulates mud patches occluding potato
- `GaussNoise(var_limit=(10,50))` — camera sensor noise
- `HueSaturationValue(±20/30/20)` — soil color variation across fields

---

## ISOBUS Integration (Harvester Display)

### What it is
ISOBUS = CAN bus at 250 kbit/s, 9-pin Deutsch connector on every modern harvester. The Virtual Terminal (VT) is the cab display — it accepts UI content via ISO 11783-6 protocol.

### How to push data to the display
1. Connect edge device to ISOBUS via CAN-USB adapter (PEAK PCAN-USB)
2. Use **AgIsoStack++** (open-source, MIT, `github.com/Open-Agriculture/AgIsoStack-plus-plus`)
3. Upload an **Object Pool** (UI definition) to the VT at startup — defines buttons, numbers, warnings
4. Send `VT Change Numeric Value` commands to update damage %, stone count, etc. in real-time
5. VT renders on existing screen — no extra screen needed

```bash
# Physical connection
# 9-pin Deutsch HD10-9-1939 male connector:
# Pin 1: CAN-Shield/Ground
# Pin 3: CAN+ (CAN-H)
# Pin 8: CAN- (CAN-L)
# Pin 9: 12V power
```

### Hackathon shortcut
AgIsoStack++ ships a **PC VT simulator** that renders the Object Pool on your laptop. Build and test display integration without real harvester hardware. Show this in the demo.

### Address claiming (automatic)
AgIsoStack++ handles ISO 11783-5 address claiming automatically. Your device appears on the bus as a new ECU, claims address, uploads UI — existing harvester ECUs ignore it.

---

## Offline Storage & FMS Sync

### Local storage: SQLite + WAL
```sql
-- Core tables
CREATE TABLE detection_events (
    id INTEGER PRIMARY KEY, ts REAL, lat REAL, lon REAL,
    damage_count INTEGER, stone_count INTEGER, clod_count INTEGER,
    damage_pct REAL, synced INTEGER DEFAULT 0
);
CREATE TABLE field_sessions (
    id INTEGER PRIMARY KEY, start_ts REAL, end_ts REAL,
    field_id TEXT, isoxml_path TEXT, synced INTEGER DEFAULT 0
);
```
Enable WAL: `PRAGMA journal_mode=WAL;`

### FMS sync: ISOXML (ISO 11783-10)
Output format: `TaskData.zip` containing `TASKDATA.XML` + binary `.bin` TLG files.

Target FMS (Netherlands):
- **365FarmNet**: REST API, POST to `/api/v2/tasks/import`
- **John Deere Operations Center**: REST API, GeoJSON + ISOXML

Sync logic: poll for WiFi every 30s → POST pending TaskData.zip → mark synced in SQLite.

### Field zone damage map
Aggregate damage counts per 10m GPS interval → cluster high-damage positions with DBSCAN → export zone polygons as GeoJSON → embed in ISOXML as Treatment Zones.

```python
from sklearn.cluster import DBSCAN
import numpy as np

coords = np.array([[e['lat'], e['lon']] for e in high_damage_events])
labels = DBSCAN(eps=0.0001, min_samples=3).fit_predict(coords)
# Each unique label = one high-risk zone polygon
```

---

## GPS Integration

### Read from harvester ISOBUS (preferred)
PGN 65267 (0xFF13) broadcast at 10Hz — contains lat/lon in 1e-7 degree resolution. Read via AgIsoStack++ or python-can. No extra GPS hardware needed.

### Fallback: NMEA serial
```python
import serial, pynmea2

with serial.Serial("/dev/ttyUSB0", 115200, timeout=1) as ser:
    line = ser.readline().decode("ascii", errors="replace")
    if line.startswith("$GNGGA"):
        msg = pynmea2.parse(line)
        lat, lon = msg.latitude, msg.longitude
```

Accuracy needed for zone mapping (~10-50m zones): standard GPS (3-5m) is sufficient. RTK not required for prototype.

---

## Business Case (quantify this in the pitch)

- Potato bruise damage costs growers ~€50–150/ton in downgrading
- A harvester processes ~30 ton/day × 30 days = 900 tons/season
- Potential loss per machine: **€45,000–135,000/season**
- Your system: ~€700 hardware, ~€2,000 installation → ROI in days

**Contractor angle**: System transforms service from "providing capacity" to "guaranteeing quality." New revenue model: data analysis + harvest quality advice as added value.

**Storage angle**: Knowing a batch has high damage % → store separately → avoid chain reaction of rot that can render entire batch unmarketable.

---

## Competitor Landscape

| System | Where | Gap |
|---|---|---|
| Tomra Blizzard/Nimbus | Packhouse grading line | Post-harvest only, not on harvester |
| Greefa IQ Grader | Packhouse | Same |
| Grimme/AVR/Dewulf | Harvester OEM | No vision-based damage detection |
| Electronic potato (accelerometer sphere) | Research/calibration | Manual, periodic, not continuous |
| Hummingbird Technologies | Drone pre-harvest | Canopy analysis, not harvest damage |

**Your system fills the exact gap between field and packhouse.**

---

## Limitations to Address Proactively (shows maturity)

1. **Blackspot bruise**: Invisible at harvest (subsurface, appears 24-72h later). Your system detects visible surface damage only. Propose NIR (850nm) as upgrade for partial subsurface detection.
2. **Labeling effort**: 90% accuracy requires 500-1000 labeled images per class. Negotiate access to VDBorne data immediately.
3. **Lighting variability**: Outdoor harvesting has highly variable light. Global shutter camera + active LED illumination (ring light) mitigates this.
4. **Harvester vibration**: Causes camera blur. Global shutter + short exposure time (< 1ms) + mechanical isolation bracket.

---

## Demo Plan (48h hackathon)

| Hour | Milestone |
|---|---|
| 0-2 | Read VDBorne data, understand image format, set up labeling tool (Roboflow) |
| 2-6 | Label 200+ images (potato/clod/stone + damaged/intact), augment to 1000+ |
| 6-14 | Train YOLOv11s + MobileNetV3-Small, evaluate, iterate |
| 14-18 | Build SQLite logger + MQTT pipeline |
| 18-24 | Build ISOBUS VT display integration (use PC simulator) |
| 24-30 | Build ISOXML export + FMS sync mock |
| 30-36 | Build damage zone map (DBSCAN + GeoJSON) |
| 36-42 | Integration testing, measure accuracy metrics |
| 42-48 | Prepare presentation: demo video, business case, architecture diagram |

**Minimum viable demo**: YOLOv11 running on video footage, damage % updating in real time on the ISOBUS VT PC simulator screen, damage zone plotted on a field map. This covers all KPIs visually.

---

## Key Libraries

```txt
ultralytics          # YOLOv11 training and export
torch torchvision    # PyTorch for bruise classifier
opencv-python        # Frame capture and preprocessing
albumentations       # Data augmentation
onnxruntime          # Inference on CPU/edge
paho-mqtt            # MQTT client
python-can           # CAN bus / ISOBUS access
pynmea2              # GPS NMEA parsing
geopandas shapely    # Spatial zone computation
scikit-learn         # DBSCAN zone clustering
sqlite3              # Built-in, local storage
requests             # FMS HTTP sync
```

---

## One-Page Pitch Structure

1. **Problem**: €45-135k/season/machine lost to harvest damage — invisible until storage
2. **Root cause**: No real-time quality signal at the harvester
3. **Solution**: Camera + AI on conveyor belt → real-time damage map → instant driver feedback
4. **Demo**: [show live inference + VT display + zone map]
5. **Technical depth**: YOLOv11 + MobileNetV3, TensorRT on Jetson, ISOBUS VT, ISOXML FMS sync
6. **Business model**: €700 hardware kit + data subscription; transforms contractor value prop
7. **Scalability**: Fleet data → federated learning → industry-wide bruise risk model
8. **Ask**: Partner with VDBorne for field validation season 2026
