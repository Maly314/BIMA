import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

from analysis.sensor_ml_pipeline import (
    G, LABELS, SENSOR_COUNT, SequenceSpec, assess_data_quality, extract_features,
    make_specs, read_capture_csv, save_dataset, score_sequence,
    simulate_sequence, train_and_evaluate,
)


class SyntheticSensorPipelineTests(unittest.TestCase):
    def spec(self, family="train_a", label=LABELS[0], seed=10, fs=100):
        return SequenceSpec(family, f"{family}-{label}-{seed}", label, seed, fs, 6.0, "test")

    def test_all_four_imus_and_all_axes_are_generated(self):
        data = simulate_sequence(self.spec(label=LABELS[1]))
        self.assertEqual(data.shape, (600, 1 + SENSOR_COUNT * 6))
        raw = data[:, 1:].reshape(len(data), SENSOR_COUNT, 6)
        self.assertTrue(np.all(np.nanstd(raw, axis=0) > 0))

    def test_generation_is_reproducible_but_families_are_distinct(self):
        first = simulate_sequence(self.spec(seed=44))
        second = simulate_sequence(self.spec(seed=44))
        evaluation = simulate_sequence(self.spec(family="eval_b", seed=44))
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, evaluation))

    def test_supported_sample_rates_have_monotonic_timestamps(self):
        for fs in (52, 60, 100):
            data = simulate_sequence(self.spec(fs=fs))
            self.assertTrue(np.all(np.diff(data[:, 0]) > 0))
            self.assertAlmostEqual(1 / np.median(np.diff(data[:, 0])), fs, places=3)

    def test_sensor_values_stay_inside_configured_icm20948_ranges(self):
        for family in ("train_a", "eval_b"):
            for label in LABELS:
                data = simulate_sequence(self.spec(family=family, label=label, seed=91))
                raw = data[:, 1:].reshape(len(data), SENSOR_COUNT, 6)
                self.assertLess(np.nanmax(np.abs(raw[:, :, :3])), 16 * G)
                self.assertLess(np.nanmax(np.abs(raw[:, :, 3:])), np.deg2rad(2000))

    def test_sensor_order_swap_does_not_change_sorted_predictions_features_much(self):
        data = simulate_sequence(self.spec(label=LABELS[1], seed=88))
        swapped = data.copy()
        shaped = swapped[:, 1:].reshape(len(swapped), SENSOR_COUNT, 6)
        shaped[:] = shaped[:, [3, 2, 1, 0], :]
        base, _ = extract_features(data)
        changed, _ = extract_features(swapped)
        # Per-sensor blocks move, but global extrema/correlations and raw feature
        # multiset remain identical. This catches silent channel loss.
        np.testing.assert_allclose(np.sort(base, axis=1), np.sort(changed, axis=1), rtol=1e-5, atol=1e-5)

    def test_missing_sensor_produces_finite_quality_features(self):
        data = simulate_sequence(self.spec(family="eval_b", label=LABELS[2], seed=52))
        data[len(data)//2:, 1:7] = np.nan
        features, _ = extract_features(data)
        self.assertTrue(np.isfinite(features).all())
        self.assertGreater(float(features[:, 1].max()), 0.0)
        quality = assess_data_quality(data)
        self.assertEqual(quality["overall"], "degraded")
        self.assertEqual(quality["sensors"][0]["state"], "unavailable")

    def test_two_missing_sensors_fail_quality_closed(self):
        data = simulate_sequence(self.spec(label=LABELS[1], seed=53))
        data[:, 1:13] = np.nan
        self.assertEqual(assess_data_quality(data)["overall"], "unusable")

    def test_bad_timestamps_and_short_sequences_fail_closed(self):
        data = simulate_sequence(self.spec())
        duplicate = data.copy(); duplicate[3, 0] = duplicate[2, 0]
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            extract_features(duplicate)
        with self.assertRaisesRegex(ValueError, "shorter"):
            extract_features(data[:10])
        low_rate = simulate_sequence(self.spec(fs=20))
        with self.assertRaisesRegex(ValueError, "below the 25 Hz minimum"):
            extract_features(low_rate)

    def test_axis_rotation_preserves_orientation_robust_features(self):
        data = simulate_sequence(self.spec(label=LABELS[1], seed=62))
        rotated = data.copy()
        raw = rotated[:, 1:].reshape(len(rotated), SENSOR_COUNT, 6)
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        raw[:, :, :3] = raw[:, :, :3] @ rotation.T
        raw[:, :, 3:] = raw[:, :, 3:] @ rotation.T
        original_features, _ = extract_features(data)
        rotated_features, _ = extract_features(rotated)
        np.testing.assert_allclose(original_features, rotated_features, rtol=2e-5, atol=2e-5)

    def test_packet_loss_and_timing_jitter_do_not_crash_extractor(self):
        data = simulate_sequence(self.spec(label=LABELS[1], seed=63))
        data = np.delete(data, np.arange(17, len(data), 29), axis=0)
        intervals = np.diff(data[:, 0], prepend=data[0, 0])
        data[:, 0] = np.cumsum(np.maximum(intervals * (1 + 0.02 * np.sin(np.arange(len(data)))), 1e-6))
        features, _ = extract_features(data)
        self.assertTrue(np.isfinite(features).all())

    def test_current_capture_csv_schema_loads_all_physical_sensor_slots(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "capture.csv"
            header = ["t"] + [f"s{s}_{axis}" for s in range(1, 5) for axis in ("ax", "ay", "az", "gx", "gy", "gz")]
            lines = [",".join(header)]
            for index in range(180):
                lines.append(",".join([f"{index/60:.6f}"] + [str((column % 6) * .01 + index * 1e-5) for column in range(24)]))
            path.write_text("\n".join(lines), encoding="utf-8")
            loaded = read_capture_csv(path)
            self.assertEqual(loaded.shape, (180, 25))
            self.assertEqual(assess_data_quality(loaded)["overall"], "ok")

    def test_dataset_manifests_prevent_train_test_leakage_and_model_roundtrips(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            train_path = root / "train.npz"
            eval_path = root / "eval.npz"
            save_dataset(train_path, make_specs("train_a", 8, 6, 1000))
            save_dataset(eval_path, make_specs("eval_b", 5, 6, 500_000))
            model_path, report_path = root / "model.joblib", root / "report.json"
            report = train_and_evaluate(train_path, eval_path, model_path, report_path)
            self.assertEqual(report["training_generator"], "train_a")
            self.assertTrue(report["evaluation_generator"].startswith("eval_b"))
            self.assertEqual(report["training_sequences"], 24)
            bundle = joblib.load(model_path)
            self.assertEqual(tuple(bundle["labels"]), LABELS)
            self.assertEqual(bundle["feature_count"], 111)
            self.assertEqual(json.loads(report_path.read_text())["clinical_ground_truth"], False)
            scored = score_sequence(bundle, simulate_sequence(self.spec(family="eval_b", label=LABELS[1], seed=999)))
            self.assertEqual(len(scored["windows"]), 2)
            self.assertEqual(scored["quality"]["overall"], "ok")

    def test_overlapping_seeds_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            train_path, eval_path = root / "train.npz", root / "eval.npz"
            save_dataset(train_path, make_specs("train_a", 2, 6, 7))
            # Force an overlapping seed while retaining the separate generator.
            specs = make_specs("eval_b", 2, 6, 7)
            save_dataset(eval_path, specs)
            with self.assertRaisesRegex(ValueError, "seeds overlap"):
                train_and_evaluate(train_path, eval_path, root / "m", root / "r")


if __name__ == "__main__":
    unittest.main()
