"""Train or run a local sustained-shakiness movement baseline.

This is an exploratory model for adult test recordings. It predicts five-second
windows, not people or diagnoses. Artifact recordings are hard negatives. Raw
accelerometer and gyroscope channels are normalized consistently so recordings
made before and after the app calibration rewrite remain comparable.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SENSORS = ("s1", "s2")
RAW_AXES = tuple(f"{sensor}_{axis}" for sensor in SENSORS for axis in ("ax", "ay", "az", "gx", "gy", "gz"))
DERIVED_CHANNELS = ("s1_accel_motion", "s2_accel_motion", "s1_gyro_motion", "s2_gyro_motion")
WINDOW_SAMPLES = 50  # five seconds at the observed 10 Hz sample rate
WINDOW_STRIDE = 10   # update once per second while retaining temporal context
ACTIVE_THRESHOLDS = np.array([0.12, 0.12, 0.06, 0.06], dtype=float)


def read_csv(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    with path.open(newline="", encoding="utf-8-sig") as source:
        rows = list(csv.DictReader(source))
    if not rows or "t" not in rows[0] or any(column not in rows[0] for column in RAW_AXES):
        raise ValueError(f"{path} does not contain the expected raw movement-sensor columns.")

    timestamps = np.array([float(row["t"]) for row in rows], dtype=float)
    raw = {column: np.array([float(row[column]) for row in rows], dtype=float) for column in RAW_AXES}
    values = np.column_stack([raw[column] for column in RAW_AXES])
    if not np.isfinite(timestamps).all() or not np.isfinite(values).all():
        raise ValueError(f"{path} contains missing or non-numeric sensor values.")
    if len(timestamps) < WINDOW_SAMPLES:
        raise ValueError(f"{path} needs at least {WINDOW_SAMPLES} samples.")

    intervals = np.diff(timestamps)
    if np.any(intervals <= 0):
        raise ValueError(f"{path} timestamps are not strictly increasing.")

    derived: list[np.ndarray] = []
    normalization: dict[str, dict[str, float | list[float]]] = {}
    for sensor in SENSORS:
        acceleration = np.column_stack([raw[f"{sensor}_{axis}"] for axis in ("ax", "ay", "az")])
        gyro = np.column_stack([raw[f"{sensor}_{axis}"] for axis in ("gx", "gy", "gz")])

        acceleration_magnitude = np.linalg.norm(acceleration, axis=1)
        gravity_reference = float(np.median(acceleration_magnitude))
        acceleration_motion = np.abs(acceleration_magnitude - gravity_reference)
        acceleration_floor = float(np.quantile(acceleration_motion, 0.10))
        acceleration_motion = np.maximum(0.0, acceleration_motion - 2.5 * acceleration_floor)

        gyro_bias = np.median(gyro, axis=0)
        gyro_motion = np.linalg.norm(gyro - gyro_bias, axis=1)
        gyro_floor = float(np.quantile(gyro_motion, 0.10))
        gyro_motion = np.maximum(0.0, gyro_motion - 2.5 * gyro_floor)

        derived.extend((acceleration_motion, gyro_motion))
        normalization[sensor] = {
            "gravity_reference": round(gravity_reference, 6),
            "acceleration_floor": round(acceleration_floor, 6),
            "gyro_bias": [round(float(value), 6) for value in gyro_bias],
            "gyro_floor": round(gyro_floor, 6),
        }

    # Arrange features in the documented channel order rather than per-sensor order.
    series = np.column_stack((derived[0], derived[2], derived[1], derived[3]))
    profile = {
        "rows": len(rows),
        "duration_s": round(float(timestamps[-1] - timestamps[0]), 3),
        "median_sample_interval_s": round(float(np.median(intervals)), 6),
        "gaps_over_0_15_s": int(np.sum(intervals > 0.15)),
        "normalization": normalization,
    }
    return timestamps, series, profile


def longest_active_fraction(active: np.ndarray) -> float:
    longest = current = 0
    for value in active:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest / len(active)


def spectral_peak_share(values: np.ndarray) -> float:
    centered = values - values.mean()
    power = np.abs(np.fft.rfft(centered))[1:] ** 2
    total = float(power.sum())
    return float(power.max() / total) if total > 1e-12 else 0.0


def make_windows(timestamps: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, list[dict[str, float]]]:
    features: list[np.ndarray] = []
    windows: list[dict[str, float]] = []
    for start in range(0, len(values) - WINDOW_SAMPLES + 1, WINDOW_STRIDE):
        stop = start + WINDOW_SAMPLES
        window = values[start:stop]
        active = window > ACTIVE_THRESHOLDS
        mean = window.mean(axis=0)
        peak_ratio = window.max(axis=0) / np.maximum(mean, 1e-6)
        longest = np.array([longest_active_fraction(active[:, channel]) for channel in range(window.shape[1])])
        spectral = np.array([spectral_peak_share(window[:, channel]) for channel in range(window.shape[1])])
        feature = np.concatenate((
            mean,
            window.std(axis=0),
            np.median(window, axis=0),
            np.quantile(window, 0.90, axis=0),
            np.quantile(window, 0.99, axis=0),
            window.max(axis=0),
            active.mean(axis=0),
            longest,
            peak_ratio,
            spectral,
        ))
        features.append(feature)
        windows.append({"start_s": round(float(timestamps[start]), 3), "end_s": round(float(timestamps[stop - 1]), 3)})
    if not features:
        raise ValueError(f"Need at least {WINDOW_SAMPLES} samples to make one prediction window.")
    return np.vstack(features), windows


def new_model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.7, max_iter=3_000, random_state=11))


def recording_weights(window_counts: list[int], labels: list[int]) -> np.ndarray:
    """Give shaky and non-shaky classes equal total weight, then balance sources."""
    weights: list[np.ndarray] = []
    for index, (count, label) in enumerate(zip(window_counts, labels)):
        same_class_sources = sum(1 for candidate in labels if candidate == label)
        source_total = 0.5 / same_class_sources
        weights.append(np.full(count, source_total / count, dtype=float))
    combined = np.concatenate(weights)
    return combined * sum(window_counts) / combined.sum()


def fit_model(features: np.ndarray, labels: np.ndarray, weights: np.ndarray):
    model = new_model()
    model.fit(features, labels, logisticregression__sample_weight=weights)
    return model


def train(shaky_csv: Path, still_csv: Path, artifact_csv: Path | None, output_model: Path, output_report: Path) -> dict:
    sources = [("shaky", shaky_csv, 1), ("mostly_still", still_csv, 0)]
    if artifact_csv:
        sources.append(("artifact", artifact_csv, 0))

    feature_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    counts: list[int] = []
    source_labels: list[int] = []
    profiles: dict[str, dict] = {}
    source_windows: dict[str, np.ndarray] = {}
    for name, path, label in sources:
        timestamps, series, profile = read_csv(path)
        window_features, _ = make_windows(timestamps, series)
        feature_parts.append(window_features)
        label_parts.append(np.full(len(window_features), label, dtype=int))
        counts.append(len(window_features))
        source_labels.append(label)
        profiles[name] = profile
        source_windows[name] = window_features

    features = np.vstack(feature_parts)
    labels = np.concatenate(label_parts)
    weights = recording_weights(counts, source_labels)

    # Window-level CV remains a separability smoke test; overlapping windows from
    # the same recordings make it optimistic, so the report labels it accordingly.
    predictions = np.zeros_like(labels)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=11)
    for train_indices, test_indices in splitter.split(features, labels):
        fold_model = fit_model(features[train_indices], labels[train_indices], weights[train_indices])
        predictions[test_indices] = fold_model.predict(features[test_indices])

    artifact_holdout = None
    if artifact_csv:
        base_features = np.vstack((source_windows["shaky"], source_windows["mostly_still"]))
        base_labels = np.concatenate((
            np.ones(len(source_windows["shaky"]), dtype=int),
            np.zeros(len(source_windows["mostly_still"]), dtype=int),
        ))
        base_weights = recording_weights(
            [len(source_windows["shaky"]), len(source_windows["mostly_still"])],
            [1, 0],
        )
        holdout_model = fit_model(base_features, base_labels, base_weights)
        artifact_probabilities = holdout_model.predict_proba(source_windows["artifact"])[:, 1]
        artifact_holdout = {
            "windows": int(len(artifact_probabilities)),
            "false_positive_rate_at_0_5": round(float(np.mean(artifact_probabilities >= 0.5)), 4),
            "mean_shaky_probability": round(float(np.mean(artifact_probabilities)), 4),
            "max_shaky_probability": round(float(np.max(artifact_probabilities)), 4),
        }

    report = {
        "purpose": "Exploratory adult sustained-shakiness window classifier with artifact hard negatives",
        "labels": {"0": "not_shaky_or_artifact", "1": "sustained_shaky"},
        "feature_version": 2,
        "window_samples": WINDOW_SAMPLES,
        "window_stride": WINDOW_STRIDE,
        "derived_channels": list(DERIVED_CHANNELS),
        "training_windows": {name: count for (name, _, _), count in zip(sources, counts)},
        "recording_balanced_weights": True,
        "within_recording_validation_accuracy": round(float(accuracy_score(labels, predictions)), 4),
        "within_recording_confusion_matrix": confusion_matrix(labels, predictions).tolist(),
        "artifact_holdout_before_inclusion": artifact_holdout,
        "data_profiles": profiles,
        "important_caveat": "Cross-validation uses overlapping windows from the same recordings and is optimistic. Artifact holdout tests only one independent negative recording. This is not performance on a new person, placement, sensor set, or clinical population.",
        "inputs": {name: str(path) for name, path, _ in sources},
    }

    fitted = fit_model(features, labels, weights)
    if artifact_csv:
        final_artifact_probabilities = fitted.predict_proba(source_windows["artifact"])[:, 1]
        report["artifact_after_inclusion_training_score"] = {
            "windows": int(len(final_artifact_probabilities)),
            "false_positive_rate_at_0_5": round(float(np.mean(final_artifact_probabilities >= 0.5)), 4),
            "mean_shaky_probability": round(float(np.mean(final_artifact_probabilities)), 4),
            "max_shaky_probability": round(float(np.max(final_artifact_probabilities)), 4),
            "caveat": "This score is measured on the artifact recording after it was included in training.",
        }
    output_model.parent.mkdir(parents=True, exist_ok=True)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": fitted,
        "derived_channels": DERIVED_CHANNELS,
        "window_samples": WINDOW_SAMPLES,
        "window_stride": WINDOW_STRIDE,
        "feature_version": 2,
        "labels": report["labels"],
    }, output_model)
    output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def predict(model_path: Path, sensor_csv: Path) -> list[dict]:
    bundle = joblib.load(model_path)
    timestamps, series, _ = read_csv(sensor_csv)
    features, windows = make_windows(timestamps, series)
    probabilities = bundle["model"].predict_proba(features)[:, 1]
    return [
        {**window, "shaky_probability": round(float(probability), 4), "prediction": "sustained_shaky" if probability >= 0.5 else "not_shaky_or_artifact"}
        for window, probability in zip(windows, probabilities)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shaky", type=Path, help="CSV labeled as sustained shaky")
    parser.add_argument("--still", type=Path, help="CSV labeled as mostly still")
    parser.add_argument("--artifact", type=Path, help="CSV containing intentional artifacts; trained as a hard negative")
    parser.add_argument("--model", type=Path, default=Path("local-models/movement-baseline.joblib"))
    parser.add_argument("--report", type=Path, default=Path("local-models/movement-baseline-report.json"))
    parser.add_argument("--predict", type=Path, help="CSV to score after training")
    args = parser.parse_args()

    if args.shaky and args.still:
        result = train(args.shaky, args.still, args.artifact, args.model, args.report)
        print(json.dumps(result, indent=2))
    if args.predict:
        if not args.model.exists():
            raise SystemExit(f"Model not found: {args.model}. Train it first with --shaky and --still.")
        print(json.dumps(predict(args.model, args.predict), indent=2))
    elif not (args.shaky and args.still):
        parser.error("Provide --shaky and --still to train, or --predict with an existing --model.")


if __name__ == "__main__":
    main()
