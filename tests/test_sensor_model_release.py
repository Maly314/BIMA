"""Release thresholds against the full independent synthetic benchmark."""

import unittest
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import classification_report

from analysis.sensor_ml_pipeline import LABELS, SENSOR_COUNT, _feature_matrix, _load_dataset, assess_data_quality, extract_features

ROOT = Path(__file__).resolve().parents[1]


class SensorModelReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.specs, cls.arrays = _load_dataset(ROOT / "synthetic-data/v2/evaluation_generator_b.npz")
        cls.bundle = joblib.load(ROOT / "local-models/four-imu-synthetic.joblib")
        cls.features, cls.labels, _ = _feature_matrix(cls.specs, cls.arrays)
        cls.predictions = cls.bundle["model"].predict(cls.features)

    def test_independent_benchmark_meets_explicit_floor(self):
        report = classification_report(self.labels, self.predictions, target_names=LABELS, output_dict=True, zero_division=0)
        self.assertGreaterEqual(report["accuracy"], 0.68)
        self.assertGreaterEqual(report["macro avg"]["f1-score"], 0.69)
        self.assertGreaterEqual(report[LABELS[1]]["precision"], 0.95)
        self.assertGreaterEqual(report[LABELS[1]]["recall"], 0.75)
        self.assertGreaterEqual(report[LABELS[2]]["recall"], 0.60)

    def test_every_supported_rate_remains_above_chance(self):
        cursor = 0
        by_rate = {52: [], 60: [], 100: []}
        for spec in self.specs:
            count = len(extract_features(self.arrays[spec.sequence_id])[0])
            by_rate[spec.sample_rate_hz].extend(range(cursor, cursor + count))
            cursor += count
        for rate, indices in by_rate.items():
            accuracy = float(np.mean(self.predictions[indices] == self.labels[indices]))
            self.assertGreaterEqual(accuracy, 0.67, f"{rate} Hz regression: {accuracy}")

    def test_sensor_order_swap_has_bounded_accuracy_change(self):
        correct = total = 0
        for spec in self.specs:
            sequence = self.arrays[spec.sequence_id].copy()
            raw = sequence[:, 1:].reshape(len(sequence), SENSOR_COUNT, 6).copy()
            sequence[:, 1:] = raw[:, [3, 2, 1, 0], :].reshape(len(sequence), -1)
            prediction = self.bundle["model"].predict(extract_features(sequence)[0])
            correct += int(np.sum(prediction == LABELS.index(spec.label)))
            total += len(prediction)
        self.assertGreaterEqual(correct / total, 0.67)

    def test_packet_loss_does_not_collapse_accuracy(self):
        correct = total = 0
        for spec in self.specs:
            sequence = np.delete(self.arrays[spec.sequence_id], np.arange(17, len(self.arrays[spec.sequence_id]), 29), axis=0)
            prediction = self.bundle["model"].predict(extract_features(sequence)[0])
            correct += int(np.sum(prediction == LABELS.index(spec.label)))
            total += len(prediction)
        self.assertGreaterEqual(correct / total, 0.68)

    def test_each_individual_imu_failure_is_reported(self):
        sequence = self.arrays[self.specs[0].sequence_id].copy()
        for sensor in range(SENSOR_COUNT):
            broken = sequence.copy()
            broken[:, 1 + sensor*6:1 + (sensor+1)*6] = np.nan
            quality = assess_data_quality(broken)
            self.assertEqual(quality["overall"], "degraded")
            self.assertEqual(quality["sensors"][sensor]["state"], "unavailable")


if __name__ == "__main__":
    unittest.main()
