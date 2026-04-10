# Technical Approach — BrabantHack 26 Track 2 (VDBorne)
## Potato Harvest Damage Detection: End-to-End Architecture

---

## System Overview

```
[Harvester Conveyor Belt]
        |
   [Camera + NIR LED]
        |
   [Edge Device: Jetson Orin Nano]
        |
   +-----------+-----------+
   |                       |
[Inference Pipeline]   [GPS / ISOBUS]
   |                       |
[MQTT Broker (local)]------+
   |
   +--[ISOBUS VT Display]  (real-time cab display)
   +--[SQLite WAL Store]   (offline-first log)
   +--[Sync Daemon]        (WiFi → AWS S3 → Snowflake)
```

---

## Step-by-Step Technical Approach

### Step 1: Image Capture

**Camera setup on conveyor belt:**
- Mount camera above conveyor belt, pointing down at ~45–90° angle
- Use global shutter to avoid motion blur from belt movement
- Add 850nm NIR LED ring light for bruise enhancement
- Target: 60fps at 640×480 minimum

**Camera interface (GStreamer on Jetson):**
```python
import cv2

# Zero-copy capture on Jetson via GStreamer pipeline
def gstreamer_pipeline(sensor_id=0, width=640, height=480, fps=60):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, framerate={fps}/1 ! "
        f"nvvidconv ! video/x-raw, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! appsink"
    )

cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
```

For USB cameras (prototype/hackathon):
```python
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 60)
```

---

### Step 2: Object Detection (potato / clod / stone)

**Model: YOLOv11s fine-tuned from COCO weights**

```python
from ultralytics import YOLO

# Training
model = YOLO("yolo11s.pt")
results = model.train(
    data="potato_belt.yaml",  # custom dataset config
    epochs=100,
    imgsz=640,
    batch=16,
    mosaic=1.0,
    degrees=15,
    flipud=0.5,
    mixup=0.1,
    copy_paste=0.1,
    lr0=0.01,
    device=0
)

# Export for Jetson TensorRT (FP16)
model.export(format="engine", half=True, device=0)
```

**Dataset YAML (potato_belt.yaml):**
```yaml
path: ./data/potato_belt
train: images/train
val: images/val
test: images/test

nc: 3
names: ["potato", "clod", "stone"]
```

**Expected performance (Jetson Orin Nano, TensorRT FP16):**
- YOLOv11s: ~85 FPS at 640×640
- YOLOv11n: ~140 FPS at 416×416 (use for production)

---

### Step 3: Damage Classification

**Model: MobileNetV3-Small, 5-channel input**

The key insight: feed BGR + L* (CIE Lab lightness) + V (HSV value) as 5 channels.
Bruises reduce L* before becoming visibly obvious in RGB.

```python
import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import numpy as np

class BruiseClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
        # Adapt first conv to accept 5-channel input
        old_conv = base.features[0][0]
        base.features[0][0] = nn.Conv2d(
            5, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )
        # Initialize extra channels from mean of RGB weights
        with torch.no_grad():
            base.features[0][0].weight[:, :3] = old_conv.weight
            base.features[0][0].weight[:, 3:] = old_conv.weight[:, :2].mean(dim=1, keepdim=True)
        base.classifier[-1] = nn.Linear(1024, 2)
        self.model = base

    def forward(self, x):
        return self.model(x)

def prepare_input(bgr_crop):
    """Convert 224x224 BGR crop to 5-channel tensor"""
    lab = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2Lab)
    hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
    l_channel = lab[:, :, 0:1].astype(np.float32) / 255.0
    v_channel = hsv[:, :, 2:3].astype(np.float32) / 255.0
    bgr_norm = bgr_crop.astype(np.float32) / 255.0
    combined = np.concatenate([bgr_norm, l_channel, v_channel], axis=2)
    return torch.tensor(combined).permute(2, 0, 1).unsqueeze(0)
```

---

### Step 4: GPS Geotagging

**Read GPS from ISOBUS (preferred — no extra hardware):**
```python
import can

bus = can.interface.Bus(channel="can0", bustype="socketcan", bitrate=250000)

def read_gps_from_isobus():
    """Read PGN 65267 (0xFF13) — position broadcast by harvester at 10Hz"""
    for msg in bus:
        if msg.arbitration_id & 0x3FFFF == 0xFF13:
            # Parse lat/lon from 8-byte CAN frame
            lat_raw = int.from_bytes(msg.data[0:4], "little")
            lon_raw = int.from_bytes(msg.data[4:8], "little")
            lat = (lat_raw / 1e7) - 210  # ISO 11783-7 offset
            lon = (lon_raw / 1e7) - 210
            return lat, lon
```

**Fallback — NMEA serial GPS:**
```python
import serial, pynmea2

with serial.Serial("/dev/ttyUSB0", 115200, timeout=1) as ser:
    line = ser.readline().decode("ascii", errors="replace")
    if line.startswith(("$GNGGA", "$GPGGA")):
        msg = pynmea2.parse(line)
        if msg.gps_qual > 0:
            return msg.latitude, msg.longitude
```

---

### Step 5: Local Storage (SQLite, offline-first)

```python
import sqlite3, time

DB_PATH = "/data/harvest.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL NOT NULL,
            lat REAL, lon REAL,
            potato_count INTEGER DEFAULT 0,
            damaged_count INTEGER DEFAULT 0,
            stone_count INTEGER DEFAULT 0,
            clod_count INTEGER DEFAULT 0,
            damage_pct REAL,
            frame_path TEXT,
            synced INTEGER DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            start_ts REAL, end_ts REAL,
            field_id TEXT,
            s3_key TEXT,
            snowflake_loaded INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    return conn

def log_detection(conn, lat, lon, counts):
    conn.execute("""
        INSERT INTO detections (ts, lat, lon, potato_count, damaged_count,
                                stone_count, clod_count, damage_pct)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (time.time(), lat, lon,
          counts["potato"], counts["damaged"],
          counts["stone"], counts["clod"],
          counts["damaged"] / max(counts["potato"], 1) * 100))
    conn.commit()
```

---

### Step 6: ISOBUS Display Integration

**Object Pool (UI definition for VT cab display):**

The harvester display UI shows:
- Current damage % (large number, color-coded: green/yellow/red)
- Stone count (last 10 seconds)
- Speed recommendation (slow down if damage > threshold)
- GPS zone warning

```cpp
// AgIsoStack++ C++ — VT client setup
// See: github.com/Open-Agriculture/AgIsoStack-plus-plus

#include "isobus/isobus/isobus_virtual_terminal_client.hpp"

auto vtClient = std::make_shared<isobus::VirtualTerminalClient>(partner, internalECU);
vtClient->set_object_pool(0, objectPoolData, objectPoolSize, "PotatoScan");
vtClient->initialize(true);

// Update damage % on display (object ID 100 = damage number field)
vtClient->send_change_numeric_value(100, damagePct);  // called from Python via subprocess or IPC
```

**Python wrapper to update display:**
```python
import subprocess

def update_vt_display(damage_pct: float, stone_count: int):
    """Trigger C++ AgIsoStack++ daemon to update VT display values"""
    subprocess.run(["/usr/local/bin/isobus-updater",
                    str(int(damage_pct * 10)),  # x10 for fixed-point
                    str(stone_count)])
```

---

### Step 7: AWS Integration

**Architecture:**
```
Edge Device (Jetson)
    |
    +-- [WiFi detect loop]
    |        |
    |   [AWS IoT Core] ← MQTT over TLS (when online)
    |        |
    |   [AWS S3 bucket]  ← raw detection logs + frames
    |        |
    |   [AWS Lambda]  ← triggered on S3 upload
    |        |
    |   [Snowflake]  ← COPY INTO from S3 stage
```

**Sync daemon — upload to AWS S3:**
```python
import boto3, json, time
import sqlite3

s3 = boto3.client("s3",
    region_name="eu-west-1",
    aws_access_key_id=AWS_KEY,
    aws_secret_access_key=AWS_SECRET
)

BUCKET = "vdborne-harvest-data"

def sync_to_s3(db_path: str):
    conn = sqlite3.connect(db_path)
    pending = conn.execute(
        "SELECT * FROM detections WHERE synced=0 ORDER BY ts"
    ).fetchall()

    if not pending:
        return

    # Batch as NDJSON (newline-delimited JSON — Snowflake native format)
    lines = []
    ids = []
    for row in pending:
        lines.append(json.dumps({
            "ts": row[1], "lat": row[2], "lon": row[3],
            "potato_count": row[4], "damaged_count": row[5],
            "stone_count": row[6], "clod_count": row[7],
            "damage_pct": row[8]
        }))
        ids.append(row[0])

    key = f"detections/{int(time.time())}.ndjson"
    s3.put_object(Bucket=BUCKET, Key=key, Body="\n".join(lines).encode())

    # Mark synced
    conn.execute(
        f"UPDATE detections SET synced=1 WHERE id IN ({','.join('?'*len(ids))})",
        ids
    )
    conn.commit()
    print(f"Synced {len(ids)} records → s3://{BUCKET}/{key}")
```

**AWS IoT Core (MQTT over TLS — alternative real-time path):**
```python
import paho.mqtt.client as mqtt
import ssl, json

def publish_to_iot_core(detection: dict):
    client = mqtt.Client()
    client.tls_set(
        ca_certs="certs/AmazonRootCA1.pem",
        certfile="certs/device.pem.crt",
        keyfile="certs/private.pem.key",
        tls_version=ssl.PROTOCOL_TLSv1_2
    )
    client.connect("xxxx.iot.eu-west-1.amazonaws.com", 8883)
    client.publish("harvester/detections", json.dumps(detection))
    client.disconnect()
```

---

### Step 8: Snowflake Integration

**Schema:**
```sql
-- In Snowflake
CREATE DATABASE vdborne;
CREATE SCHEMA harvest;

CREATE TABLE harvest.detections (
    ts          TIMESTAMP_NTZ,
    lat         FLOAT,
    lon         FLOAT,
    potato_count   INTEGER,
    damaged_count  INTEGER,
    stone_count    INTEGER,
    clod_count     INTEGER,
    damage_pct     FLOAT,
    field_id    VARCHAR(50),
    loaded_at   TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

CREATE TABLE harvest.field_zones (
    zone_id     VARCHAR(50),
    field_id    VARCHAR(50),
    risk_level  VARCHAR(20),    -- low / medium / high
    center_lat  FLOAT,
    center_lon  FLOAT,
    polygon     VARIANT,        -- GeoJSON stored as VARIANT
    created_at  TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);
```

**S3 External Stage + COPY INTO:**
```sql
-- Create S3 external stage
CREATE OR REPLACE STAGE harvest.s3_stage
    URL = 's3://vdborne-harvest-data/'
    CREDENTIALS = (AWS_KEY_ID='...' AWS_SECRET_KEY='...')
    FILE_FORMAT = (TYPE='JSON', STRIP_OUTER_ARRAY=FALSE);

-- COPY from stage (triggered by Lambda or scheduled task)
COPY INTO harvest.detections (ts, lat, lon, potato_count, damaged_count,
                               stone_count, clod_count, damage_pct)
FROM (
    SELECT
        TO_TIMESTAMP($1:ts::FLOAT),
        $1:lat::FLOAT,
        $1:lon::FLOAT,
        $1:potato_count::INTEGER,
        $1:damaged_count::INTEGER,
        $1:stone_count::INTEGER,
        $1:clod_count::INTEGER,
        $1:damage_pct::FLOAT
    FROM @harvest.s3_stage
)
FILE_FORMAT = (TYPE='JSON')
ON_ERROR = 'CONTINUE';
```

**Python Snowflake connector (for analytics / FMS reporting):**
```python
import snowflake.connector

conn = snowflake.connector.connect(
    account="vdborne.eu-west-1",
    user="harvest_app",
    password=SNOWFLAKE_PASSWORD,
    database="vdborne",
    schema="harvest",
    warehouse="compute_wh"
)

# Query: damage heatmap per field zone
df = pd.read_sql("""
    SELECT
        lat, lon, damage_pct,
        AVG(damage_pct) OVER (
            ORDER BY ts
            ROWS BETWEEN 50 PRECEDING AND CURRENT ROW
        ) AS rolling_damage_pct
    FROM detections
    WHERE ts > DATEADD('hour', -24, CURRENT_TIMESTAMP())
    ORDER BY ts
""", conn)
```

**Snowpark for zone clustering (run in Snowflake):**
```python
from snowflake.snowpark import Session
from snowflake.snowpark.functions import col

session = Session.builder.configs(connection_params).create()

df = session.table("harvest.detections").filter(col("damage_pct") > 20)
# Export to pandas for DBSCAN zone computation
damage_df = df.to_pandas()

from sklearn.cluster import DBSCAN
import numpy as np

coords = damage_df[["lat", "lon"]].values
labels = DBSCAN(eps=0.0001, min_samples=5).fit_predict(coords)
damage_df["zone_id"] = labels

# Write back to Snowflake
session.write_pandas(damage_df, "field_zones", auto_create_table=True)
```

---

### Step 9: Field Zone Map Generation

```python
import geopandas as gpd
from shapely.geometry import Point, MultiPoint, mapping
from shapely.ops import unary_union
import json

def compute_damage_zones(detections_df, damage_threshold=20.0):
    """Cluster high-damage GPS points into risk zone polygons"""
    from sklearn.cluster import DBSCAN

    high_damage = detections_df[detections_df["damage_pct"] > damage_threshold]
    if len(high_damage) < 3:
        return []

    coords = high_damage[["lat", "lon"]].values
    labels = DBSCAN(eps=0.0001, min_samples=3).fit_predict(coords)

    zones = []
    for label in set(labels):
        if label == -1:
            continue
        cluster_pts = coords[labels == label]
        polygon = MultiPoint([Point(lon, lat) for lat, lon in cluster_pts]).convex_hull.buffer(0.0001)
        zones.append({
            "zone_id": f"zone_{label}",
            "risk_level": "high" if len(cluster_pts) > 20 else "medium",
            "geometry": mapping(polygon),
            "n_events": int(len(cluster_pts)),
            "avg_damage_pct": float(high_damage.iloc[labels == label]["damage_pct"].mean())
        })

    return zones

def export_geojson(zones, output_path):
    geojson = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": z["geometry"],
            "properties": {k: v for k, v in z.items() if k != "geometry"}
        } for z in zones]
    }
    with open(output_path, "w") as f:
        json.dump(geojson, f, indent=2)
```

---

## What Is Still Dependent on the VDBorne Dataset

| Component | Dataset Independent | Dataset Dependent | Notes |
|---|---|---|---|
| YOLOv11 training (potato/clod/stone) | Partially (Roboflow surrogates exist) | **Strongly** | Need VDBorne conveyor images for accurate generalization to their specific soil/lighting |
| Bruise classifier | Partially (fruit defect datasets) | **Strongly** | Bruise appearance is variety-specific; VDBorne uses specific potato varieties |
| NIR channel calibration | No | **Completely** | Need images with known bruised/intact labels under 850nm illumination |
| Damage threshold tuning | No | **Completely** | The 90% recall target requires calibration against ground-truth labeled images |
| GPS coordinate system | No | No | Standard NMEA/WGS84 |
| ISOBUS Object Pool design | No | Partially | Screen layout TBD; need to know which VT terminal is on their harvester |
| Snowflake schema | No | No | Generic detection schema works for any farm |
| ISOXML field boundaries | No | Yes | Field boundary polygons are farm-specific |
| Stone/clod detection accuracy | Partially | **Strongly** | Stone appearance varies by field (limestone vs granite) |
| Conveyor belt background model | No | **Strongly** | Belt color/texture affects background subtraction and model confidence |

### Minimum viable dataset from VDBorne to start:
1. **200+ labeled images** of potatoes/clods/stones on their specific conveyor belt
2. **50+ images** of damaged vs undamaged potatoes (post-wash images are best)
3. **GPS track** from one harvest run (to test zone mapping)
4. **VT terminal model** (to design compatible Object Pool)

### What you can build without VDBorne data:
- Full inference pipeline (using surrogate datasets)
- ISOBUS VT integration (using PC simulator)
- AWS S3 + Snowflake pipeline (using synthetic data)
- GPS zone mapping (using synthetic GPS tracks)
- The complete demo, labeled "trained on surrogate data, to be fine-tuned on VDBorne images"

---

## AWS + Snowflake Architecture Diagram

```
EDGE (offline)                    CLOUD (online)
─────────────────────────────     ──────────────────────────────────────
Camera → YOLOv11 → MQTT          S3: vdborne-harvest-data/
  ↓                                  ├─ detections/*.ndjson
  SQLite WAL                         ├─ frames/*.jpg (thumbnails)
  ↓                                  └─ sessions/*.zip (ISOXML)
  [WiFi available?]                      ↓
  ↓ YES                           Lambda: trigger COPY INTO
  boto3 S3 upload ──────────→    Snowflake: harvest.detections
  IoT Core MQTT ────────────→        ↓
                                  Snowpark: zone clustering
                                      ↓
                                  Snowflake: harvest.field_zones
                                      ↓
                                  365FarmNet / JD Ops Center API
                                  (ISOXML export from Snowflake query)
```

---

## Key Libraries Install

```bash
pip install ultralytics torch torchvision opencv-python \
            albumentations onnxruntime paho-mqtt python-can \
            pynmea2 geopandas shapely scikit-learn \
            boto3 snowflake-connector-python snowflake-snowpark-python \
            pyserial requests fastapi uvicorn
```
