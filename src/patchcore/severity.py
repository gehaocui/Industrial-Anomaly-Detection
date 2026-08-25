from dataclasses import dataclass

import numpy as np


@dataclass
class SeverityResult:
    severity_score: float
    severity_level: str
    defect_area_ratio: float
    mean_intensity: float
    peak_intensity: float


class DefectSeverityEstimator:
    """
    Estimate defect severity from a PatchCore anomaly map.

    The estimator combines:
    1. Defect area ratio
    2. Mean anomaly intensity
    3. Peak anomaly intensity
    """

    def __init__(
        self,
        pixel_threshold=0.60,
        medium_threshold=0.45,
        high_threshold=0.70,
    ):
        self.pixel_threshold = pixel_threshold
        self.medium_threshold = medium_threshold
        self.high_threshold = high_threshold

    def _normalize_map(self, anomaly_map):
        anomaly_map = np.asarray(anomaly_map, dtype=np.float32)

        if anomaly_map.size == 0:
            return np.zeros_like(anomaly_map)

        # Robust normalization to reduce the influence of extreme pixels.
        low = np.percentile(anomaly_map, 5)
        high = np.percentile(anomaly_map, 99)

        if high - low < 1e-8:
            return np.zeros_like(anomaly_map)

        normalized = (anomaly_map - low) / (high - low)

        return np.clip(normalized, 0.0, 1.0)

    def analyze(self, anomaly_map):
        normalized_map = self._normalize_map(anomaly_map)

        defect_mask = normalized_map >= self.pixel_threshold

        defect_area_ratio = float(np.mean(defect_mask))

        if np.any(defect_mask):
            defect_values = normalized_map[defect_mask]

            mean_intensity = float(np.mean(defect_values))
            peak_intensity = float(
                np.percentile(defect_values, 95)
            )
        else:
            mean_intensity = 0.0
            peak_intensity = 0.0

        # Convert area ratio to a bounded severity component.
        # 25% affected pixels are treated as maximum area severity.
        area_component = min(
            defect_area_ratio / 0.25,
            1.0
        )

        severity_score = (
            0.20 * area_component
            + 0.35 * mean_intensity
            + 0.45 * peak_intensity
        )

        severity_score = float(
            np.clip(severity_score, 0.0, 1.0)
        )

        if severity_score >= self.high_threshold:
            severity_level = "HIGH"
        elif severity_score >= self.medium_threshold:
            severity_level = "MEDIUM"
        else:
            severity_level = "LOW"

        return SeverityResult(
            severity_score=severity_score,
            severity_level=severity_level,
            defect_area_ratio=defect_area_ratio,
            mean_intensity=mean_intensity,
            peak_intensity=peak_intensity,
        )