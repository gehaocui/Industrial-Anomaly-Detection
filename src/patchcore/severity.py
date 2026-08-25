from dataclasses import dataclass

import numpy as np


@dataclass
class SeverityResult:
    severity_score: float
    severity_level: str
    defect_area_ratio: float
    mean_intensity: float
    peak_intensity: float
    anomaly_confidence: float


class DefectSeverityEstimator:
    """
    Estimate defect severity from PatchCore outputs.

    Severity combines:
    1. Image-level anomaly confidence
    2. Defect area ratio
    3. Mean anomaly intensity
    4. Peak anomaly intensity

    Image-level confidence is calibrated using normal samples only.
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

        self.normal_score_threshold = None
        self.score_scale = None

    def calibrate(self, normal_scores):
        """
        Calibrate image-level anomaly confidence using normal samples.
        """

        normal_scores = np.asarray(
            normal_scores,
            dtype=np.float32
        ).reshape(-1)

        if normal_scores.size == 0:
            raise ValueError(
                "normal_scores must contain at least one score"
            )

        # 99th percentile defines the upper normal boundary
        self.normal_score_threshold = float(
            np.percentile(normal_scores, 99)
        )

        score_std = float(np.std(normal_scores))

        self.score_scale = max(
            self.normal_score_threshold,
            3.0 * score_std,
            1e-8,
        )

    def _normalize_map(self, anomaly_map):
        anomaly_map = np.asarray(
            anomaly_map,
            dtype=np.float32
        )

        if anomaly_map.size == 0:
            return np.zeros_like(anomaly_map)

        low = np.percentile(anomaly_map, 5)
        high = np.percentile(anomaly_map, 99)

        if high - low < 1e-8:
            return np.zeros_like(anomaly_map)

        normalized = (
            anomaly_map - low
        ) / (high - low)

        return np.clip(
            normalized,
            0.0,
            1.0
        )

    def _get_anomaly_confidence(self, image_score):
        if (
            self.normal_score_threshold is None
            or self.score_scale is None
        ):
            return 1.0

        confidence = (
            float(image_score)
            - self.normal_score_threshold
        ) / self.score_scale

        return float(
            np.clip(confidence, 0.0, 1.0)
        )

    def analyze(
        self,
        anomaly_map,
        image_score=None,
    ):
        normalized_map = self._normalize_map(
            anomaly_map
        )

        defect_mask = (
            normalized_map >= self.pixel_threshold
        )

        defect_area_ratio = float(
            np.mean(defect_mask)
        )

        if np.any(defect_mask):
            defect_values = normalized_map[
                defect_mask
            ]

            mean_intensity = float(
                np.mean(defect_values)
            )

            peak_intensity = float(
                np.percentile(
                    defect_values,
                    95
                )
            )
        else:
            mean_intensity = 0.0
            peak_intensity = 0.0

        area_component = min(
            defect_area_ratio / 0.25,
            1.0,
        )

        local_severity = (
            0.20 * area_component
            + 0.35 * mean_intensity
            + 0.45 * peak_intensity
        )

        if image_score is None:
            anomaly_confidence = 1.0
        else:
            anomaly_confidence = (
                self._get_anomaly_confidence(
                    image_score
                )
            )

        severity_score = (
            local_severity
            * anomaly_confidence
        )

        severity_score = float(
            np.clip(
                severity_score,
                0.0,
                1.0
            )
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
            anomaly_confidence=anomaly_confidence,
        )