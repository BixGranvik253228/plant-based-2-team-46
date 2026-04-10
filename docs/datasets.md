# Dataset Reference — BrabantHack 26 Track 2

## Priority Order for Hackathon

### 1. Zenodo — WUR Potato Bruise Detection (BEST for bruise classifier)
- **URL**: search `potato bruise detection` on zenodo.org
- **Images**: ~400
- **Classes**: bruised / unbruised (tuber surface)
- **Account**: None required — direct download
- **Why**: authentic WUR agricultural research data, directly citable to judges
- **Use for**: MobileNetV3-Small bruise classifier training

### 2. Roboflow — Potato Defect Detection (BEST for YOLO detector)
- **URL**: universe.roboflow.com → search "potato defect detection"
- **Images**: ~1,500
- **Classes**: bruise, crack, rot, good (bounding boxes)
- **Account**: Free Roboflow account + API key
- **Download**: `pip install roboflow` then 3 lines of Python (YOLO format)
- **Use for**: YOLOv11 fine-tuning (potato/clod/stone detector)

### 3. Roboflow — Rock / Stone / Clod Detection
- **URL**: universe.roboflow.com → search "rock detection" or "stone soil"
- **Images**: ~600
- **Classes**: rock, stone, clod, soil
- **Account**: Free Roboflow account
- **Use for**: Stone and clod detection classes in YOLO model

### 4. MVTec Anomaly Detection — Hazelnut (no-account fallback)
- **URL**: mvtec.com/company/research/datasets/mvtec-ad
- **Images**: 1,258 (train: 391 good; test: 70 good + 70 each defect type)
- **Classes**: crack, cut, hole, print (pixel-level anomaly masks)
- **Account**: None — direct download
- **Why**: Industrial surface defect benchmark; same pipeline architecture as potato bruising
- **Use for**: Demo/proof-of-concept when potato-specific data unavailable

### 5. Kaggle — Potato Tuber Disease
- **URL**: kaggle.com/datasets/hafiznouman786/potato-disease-detection-dataset
- **Images**: ~1,500
- **Classes**: healthy, blackscurf, common_scab, dry_rot, blackleg
- **Account**: Free Kaggle account
- **Format**: Classification folders (no bboxes — needs manual annotation for detection)
- **Use for**: Bruise classifier (crop + classify pipeline)

---

## What's Missing (Must Get From VDBorne)

| Gap | Why critical |
|---|---|
| Conveyor belt background images | All surrogate datasets lack the dark rubber belt background; model will miss false positives |
| Potato variety-specific appearance | VDBorne grows specific starch varieties (Seresta, Aveka) with distinct skin color/texture |
| Post-wash damage images | Bruises most visible after washing — this is VDBorne's best data source |
| Stone/clod from their specific fields | Stone mineralogy (limestone NL vs granite) affects color signature |
| Labeled damage severity | 90% recall KPI requires calibrated threshold on real damaged potatoes |

**Ask Marnik van Geelen (marnik@vdbornecampus.com) on Day 1 for**:
1. 200+ conveyor belt images (even unlabeled) — will label on-site with Roboflow
2. Post-wash images of damaged vs. intact potatoes
3. Which VT terminal is on their harvester (brand/model)
4. GPS track file from one harvest run

---

## Quick Download Commands

```bash
# Roboflow (after getting free API key from app.roboflow.com)
pip install roboflow
python3 -c "
from roboflow import Roboflow
rf = Roboflow(api_key='YOUR_KEY')
# Potato defect detector
rf.workspace('roboflow-universe-projects').project('potato-disease-detection').version(1).download('yolov11')
# Rock/clod detector
rf.workspace('roboflow-universe-projects').project('rock-detection').version(1).download('yolov11')
"

# MVTec hazelnut (no account)
# Download from: https://www.mvtec.com/company/research/datasets/mvtec-ad
# Extract hazelnut/ folder, use as anomaly detection benchmark
```
