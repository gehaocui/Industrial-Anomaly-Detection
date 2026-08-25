# Industrial Anomaly Detection with Severity Estimation

Industrial visual anomaly detection based on **PatchCore**, extended with **normal-sample-calibrated defect severity estimation**.

This project builds on the original PatchCore implementation from Amazon Science and adds a severity analysis layer that converts anomaly detection outputs into interpretable defect severity scores and levels.

## Overview

PatchCore is an unsupervised industrial anomaly detection method that learns a memory bank of normal image patch features and identifies anomalous regions through nearest-neighbor feature distances.

The original PatchCore pipeline provides:

* Image-level anomaly scores
* Pixel-level anomaly localization
* Anomaly heatmaps

This project extends the pipeline with:

* Defect severity estimation
* Normal-only anomaly score calibration
* LOW / MEDIUM / HIGH severity classification
* Defect area and anomaly intensity analysis
* A `predict_with_severity()` inference interface
* Experimental evaluation on the MVTec AD Bottle dataset

## Pipeline

```text
Normal Training Images
        │
        ▼
Pretrained CNN Feature Extractor
        │
        ▼
Patch Features
        │
        ▼
PatchCore Memory Bank
        │
        ├──────────────────────┐
        │                      │
        ▼                      ▼
 Test Image            Held-out Normal Images
        │                      │
        ▼                      ▼
PatchCore Inference     Severity Calibration
        │
        ├── Anomaly Score
        │
        └── Anomaly Heatmap
                 │
                 ▼
        Defect Severity Estimator
                 │
        ┌────────┼─────────┐
        ▼        ▼         ▼
      Area     Mean      Peak
     Ratio   Intensity  Intensity
        └────────┼─────────┘
                 ▼
        Local Severity Score
                 │
                 ×
        Anomaly Confidence
                 │
                 ▼
           Severity Score
                 │
                 ▼
        LOW / MEDIUM / HIGH
```

## Severity Estimation

A first version of the severity estimator used only the normalized anomaly heatmap.

Testing showed that this approach could incorrectly assign high severity to normal images because every anomaly map was independently normalized.

The current implementation therefore introduces **normal-only calibration**.

### 1. Normal Score Calibration

A set of held-out normal samples is passed through PatchCore.

The upper normal boundary is defined using the 99th percentile of normal image anomaly scores:

```text
normal_threshold = percentile(normal_scores, 99)
```

An anomaly confidence value is then calculated from how far a test image exceeds the calibrated normal range:

```text
confidence =
clip(
    (image_score - normal_threshold) / score_scale,
    0,
    1
)
```

This prevents local heatmap noise in normal images from automatically producing high severity scores.

### 2. Local Defect Severity

The anomaly heatmap is robustly normalized using its 5th and 99th percentiles.

Pixels above the anomaly threshold are treated as the estimated defect region.

Three local characteristics are extracted:

```text
Defect Area Ratio
Mean Anomaly Intensity
Peak Anomaly Intensity
```

The local severity score is calculated as:

```text
Local Severity =
0.20 × Area Component
+ 0.35 × Mean Intensity
+ 0.45 × Peak Intensity
```

The final severity score combines local defect information with the calibrated image-level anomaly confidence:

```text
Severity Score =
Local Severity × Anomaly Confidence
```

The current severity levels are:

```text
LOW       Severity < 0.45
MEDIUM    0.45 ≤ Severity < 0.70
HIGH      Severity ≥ 0.70
```

The weighting and severity thresholds are currently heuristic rather than learned parameters.

## MVTec AD Bottle Evaluation

The current portfolio experiment evaluates the extension on the **Bottle** category of MVTec AD.

### Experimental Setup

```text
Backbone:              ResNet50
Feature Layers:        layer2 + layer3
Input Size:            224 × 224
Patch Size:            3
Nearest Neighbors:     1

Normal Memory Samples: 32
Calibration Samples:   32
Test Images:           83
```

The 32 calibration images are normal samples that are separate from the normal images used to construct the PatchCore memory bank.

## Results

| Defect Type   | Images | Avg. PatchCore Score | Avg. Confidence | Avg. Severity | LOW | MEDIUM | HIGH |
| ------------- | -----: | -------------------: | --------------: | ------------: | --: | -----: | ---: |
| Good          |     20 |               0.1017 |          0.0048 |        0.0038 |  20 |      0 |    0 |
| Broken Large  |     20 |               0.5319 |          1.0000 |        0.8148 |   0 |      0 |   20 |
| Broken Small  |     22 |               0.5282 |          1.0000 |        0.7744 |   0 |      0 |   22 |
| Contamination |     21 |               0.3897 |          0.9449 |        0.7737 |   1 |      2 |   18 |

### Image-Level AUROC

```text
PatchCore Anomaly Score AUROC: 1.0000
Severity Score AUROC:          1.0000
```

All three defect categories achieved a higher average severity score than normal Bottle images:

```text
Broken Large   0.8148 > Good 0.0038
Broken Small   0.7744 > Good 0.0038
Contamination  0.7737 > Good 0.0038
```

The full per-image results are available in:

```text
experiments/bottle_full_test.csv
```

> **Note:** These results represent the current evaluation configuration on the MVTec AD **Bottle category only**. They should not be interpreted as performance on the complete 15-category MVTec AD benchmark.

## Key Extensions

### Defect Severity Estimator

Added:

```text
src/patchcore/severity.py
```

The module provides:

```python
SeverityResult(
    severity_score,
    severity_level,
    defect_area_ratio,
    mean_intensity,
    peak_intensity,
    anomaly_confidence,
)
```

### Normal-Only Calibration

Severity confidence is calibrated exclusively using normal samples, preserving the unsupervised nature of the anomaly detection pipeline.

No defect labels are required for calibration.

### Extended Inference API

The original PatchCore API remains unchanged:

```python
scores, masks = model.predict(images)
```

A new interface adds severity analysis:

```python
scores, masks, severities = model.predict_with_severity(images)
```

For a DataLoader:

```python
scores, masks, severities, labels, masks_gt = (
    model.predict_with_severity(test_loader)
)
```

### Calibration Example

```python
normal_scores, _, _, _ = model.predict(
    calibration_loader
)

model.severity_estimator.calibrate(
    normal_scores
)

scores, masks, severities, labels, masks_gt = (
    model.predict_with_severity(test_loader)
)
```

## Project Structure

```text
Industrial-Anomaly-Detection/
│
├── bin/
│   ├── run_patchcore.py
│   └── load_and_evaluate_patchcore.py
│
├── src/
│   └── patchcore/
│       ├── backbones.py
│       ├── common.py
│       ├── datasets/
│       ├── metrics.py
│       ├── patchcore.py
│       ├── sampler.py
│       ├── severity.py
│       └── utils.py
│
├── experiments/
│   └── bottle_full_test.csv
│
├── images/
├── models/
├── requirements.txt
├── LICENSE
└── NOTICE
```

## Installation

Clone the repository:

```bash
git clone https://github.com/gehaocui/Industrial-Anomaly-Detection.git
cd Industrial-Anomaly-Detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The project uses packages including:

```text
PyTorch
Torchvision
FAISS
NumPy
SciPy
Scikit-learn
Scikit-image
Timm
Matplotlib
```

When running project scripts directly, expose the source directory through `PYTHONPATH`:

```bash
PYTHONPATH=src python bin/run_patchcore.py ...
```

## Dataset

The project is compatible with the **MVTec Anomaly Detection Dataset (MVTec AD)**.

The expected directory structure is:

```text
datasets/
└── mvtec_ad/
    └── bottle/
        ├── train/
        │   └── good/
        │
        ├── test/
        │   ├── good/
        │   ├── broken_large/
        │   ├── broken_small/
        │   └── contamination/
        │
        └── ground_truth/
```

Datasets are excluded from Git tracking through `.gitignore`.

Please download MVTec AD separately and follow its dataset license and usage requirements.

## Why Severity Calibration Matters

An important finding during development was that anomaly heatmap intensity alone is not sufficient for severity estimation.

A normal image may contain relatively strong local variations even when its overall PatchCore anomaly score is low.

For example, before calibration:

```text
GOOD severity:   ~0.88
BROKEN severity: ~0.88
```

Both images were incorrectly classified as HIGH severity.

After introducing normal-score calibration:

```text
GOOD
PatchCore Score: 0.1136
Confidence:      0.0000
Severity:        0.0000
Level:           LOW

BROKEN LARGE
PatchCore Score: 0.4623
Confidence:      1.0000
Severity:        ~0.87
Level:           HIGH
```

This calibration step makes severity estimation depend on both:

```text
Is the image globally anomalous?
+
How large and intense is the localized anomaly?
```

rather than relying only on independently normalized heatmaps.

## Limitations

The current severity estimator is an experimental extension.

Current limitations include:

* Severity weights are manually defined rather than learned.
* LOW / MEDIUM / HIGH thresholds are heuristic.
* Evaluation has currently been performed on the MVTec AD Bottle category.
* MVTec AD provides anomaly categories but does not provide official defect severity labels.
* Therefore, severity values should be interpreted as relative defect indicators rather than ground-truth industrial severity labels.

Future work may include:

* Evaluation across all MVTec AD categories
* Learned severity calibration
* Category-specific calibration
* Threshold optimization
* Interactive anomaly visualization
* Real-time industrial inspection demos

## Original PatchCore

This repository is based on the official implementation of:

**Towards Total Recall in Industrial Anomaly Detection**
Karsten Roth, Latha Pemula, Joaquin Zepeda, Bernhard Schölkopf, Thomas Brox, Peter Gehler

Original implementation:

`amazon-science/patchcore-inspection`

PatchCore is used here as the anomaly detection baseline. The defect severity estimation and normal-score calibration modules are extensions added in this repository.

The original Git history, attribution, `LICENSE`, and `NOTICE` files are retained.

## License

The PatchCore implementation is distributed under the **Apache License 2.0**.

Please refer to the repository `LICENSE` and `NOTICE` files for details.
