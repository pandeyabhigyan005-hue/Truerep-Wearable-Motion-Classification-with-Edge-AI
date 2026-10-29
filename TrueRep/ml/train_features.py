"""Train the bicep-curl and lateral-raise models from exercise-specific CSV datasets with a shared neutral CSV.

Expected headerless CSV columns:
    ax, ay, az, gx, gy, gz[, label]

Bicep curl keeps its legacy per-file labels. Lateral-raise rows use their
explicit 0/1/2 labels, so a CSV containing both good and bad labeled rows is
split by label. Text-only Serial Monitor status lines are ignored; malformed
sensor rows are rejected.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import m2cgen as m2c
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix


ROOT = Path(__file__).resolve().parent
OUTPUT_HEADER = ROOT.parent / "firmware" / "model.h"

WINDOW_SIZE = 50
WINDOW_STEP = 25
TEST_FRACTION = 0.20
RANDOM_SEED = 42

# Stable output indices shared with the ESP32 and dashboard:
# 0 = NEUTRAL, 1 = GOOD FORM, 2 = BAD FORM.
CLASS_IDS = {"neutral": 0, "good": 1, "bad": 2}
CLASS_NAMES = ["NEUTRAL", "GOOD FORM", "BAD FORM"]

# Separate source files and separate classifiers for each exercise.
# Bicep curl uses the existing good/bad data, plus a new real_neutral.csv.
# Lateral-raise good/bad files are separate; real_neutral.csv is shared.
DATASETS = {
    "bicep_curl": {
        "neutral": "data/real_neutral.csv",
        "good": "data/real_good_form.csv",
        "bad": "data/real_bad_form.csv",
    },
    "lateral_raise": {
        "neutral": "data/real_neutral.csv",
        "good": "data/lateral_raise_good_form.csv",
        "bad": "data/lateral_raise_bad_form.csv",
    },
}


def load_recording(
    path: Path, class_name: str, exercise: str
) -> list[tuple[int, np.ndarray]]:
    """Load labeled contiguous sample segments from one CSV recording.

    Bicep files may use legacy labels (good=0, bad=1), so their configured
    file role remains authoritative. Lateral-raise labels are canonical
    (neutral=0, good=1, bad=2), and the good file may contain both 1 and 2.
    """
    role_class_id = CLASS_IDS[class_name]
    if not path.is_file():
        raise FileNotFoundError(path)

    frame = pd.read_csv(path, header=None)
    if frame.shape[1] not in (6, 7):
        raise ValueError(
            f"{path.name}: expected 6 sensor columns and an optional label "
            f"column; found {frame.shape[1]} columns."
        )

    sensor_frame = frame.iloc[:, :6].apply(pd.to_numeric, errors="coerce")
    has_labels = frame.shape[1] == 7
    labels = (
        pd.to_numeric(frame.iloc[:, 6], errors="coerce")
        if has_labels else None
    )

    # Serial Monitor status text can be copied between captured sample blocks.
    # Ignore only rows that contain no numeric sensor values and no label.
    no_sample = sensor_frame.isna().all(axis=1)
    if has_labels:
        no_sample &= labels.isna()
    dropped_logs = int(no_sample.sum())
    if dropped_logs:
        print(f"{path.name}: skipped {dropped_logs} text-only/blank log rows.")
        frame = frame.loc[~no_sample].reset_index(drop=True)
        sensor_frame = sensor_frame.loc[~no_sample].reset_index(drop=True)
        if has_labels:
            labels = labels.loc[~no_sample].reset_index(drop=True)

    values = sensor_frame.to_numpy(dtype=np.float32)
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError(
            f"{path.name}: contains a sample row with missing or nonnumeric "
            "sensor values; only wholly text-only log rows can be skipped."
        )

    if has_labels:
        label_values = labels.to_numpy(dtype=np.float64)
        if not np.isfinite(label_values).all() or not np.equal(
            label_values, np.round(label_values)
        ).all():
            raise ValueError(f"{path.name}: label column contains invalid values.")
        label_values = label_values.astype(np.int32)

        if exercise == "lateral_raise" and class_name == "good":
            # The user's updated CSV contains canonical GOOD (1) and BAD (2)
            # examples. Preserve the explicit labels instead of trusting its name.
            allowed = {CLASS_IDS["good"], CLASS_IDS["bad"]}
            if not set(np.unique(label_values)).issubset(allowed):
                raise ValueError(
                    f"{path.name}: expected lateral-raise good/bad labels "
                    "1 and/or 2."
                )
            row_classes = label_values
        else:
            allowed_by_role = {
                "neutral": {0},
                "good": {0, 1} if exercise == "bicep_curl" else {1},
                "bad": {1, 2} if exercise == "bicep_curl" else {2},
            }
            if not set(np.unique(label_values)).issubset(allowed_by_role[class_name]):
                raise ValueError(
                    f"{path.name}: labels do not match its configured "
                    f"{exercise} {class_name} recording."
                )
            row_classes = np.full(len(values), role_class_id, dtype=np.int32)
    else:
        row_classes = np.full(len(values), role_class_id, dtype=np.int32)

    # Preserve ordering and split whenever the label changes. This prevents a
    # feature window from spanning good and bad samples across a label boundary.
    segments: list[tuple[int, np.ndarray]] = []
    segment_start = 0
    for i in range(1, len(values) + 1):
        if i == len(values) or row_classes[i] != row_classes[segment_start]:
            segment_class = int(row_classes[segment_start])
            segment = values[segment_start:i]
            if len(segment) < WINDOW_SIZE * 2:
                raise ValueError(
                    f"{path.name}: a {CLASS_NAMES[segment_class]} segment has "
                    f"only {len(segment)} samples; needs at least "
                    f"{WINDOW_SIZE * 2}."
                )
            segments.append((segment_class, segment))
            segment_start = i

    if has_labels and exercise == "lateral_raise" and class_name == "good":
        counts = pd.Series(row_classes).value_counts().sort_index().to_dict()
        print(
            f"{path.name}: routed labeled rows by class "
            f"(GOOD=1, BAD=2): {counts}."
        )
    return segments

def window_features(samples: np.ndarray) -> np.ndarray:
    """Match the ESP32's 24 features: mean, std, abs-max, RMS per axis."""
    rows = []
    for start in range(0, len(samples) - WINDOW_SIZE + 1, WINDOW_STEP):
        window = samples[start : start + WINDOW_SIZE]
        features = []
        for axis in range(6):
            axis_data = window[:, axis]
            features.extend(
                [
                    float(np.mean(axis_data)),
                    float(np.std(axis_data)),
                    float(np.max(np.abs(axis_data))),
                    float(np.sqrt(np.mean(axis_data**2))),
                ]
            )
        rows.append(features)
    return np.asarray(rows, dtype=np.float64)


def split_recording(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split chronologically before windowing, with a window-sized gap."""
    split_at = int(len(samples) * (1.0 - TEST_FRACTION))
    test_at = split_at + WINDOW_SIZE
    training = samples[:split_at]
    validation = samples[test_at:]
    if len(training) < WINDOW_SIZE or len(validation) < WINDOW_SIZE:
        raise ValueError("A recording is too short after the chronological split.")
    return training, validation


def cpp_compatible(code: str) -> str:
    """Replace m2cgen C compound literals with ordinary C++ assignments."""
    pattern = re.compile(
        r"memcpy\(\s*(var\d+)\s*,\s*\(double\[\]\)\s*\{([^}]*)\}\s*,"
        r"\s*\d+\s*\*\s*sizeof\(double\)\s*\);"
    )

    def replace_initializer(match: re.Match[str]) -> str:
        variable = match.group(1)
        values = [value.strip() for value in match.group(2).split(",")]
        return "\n".join(f"{variable}[{i}] = {value};" for i, value in enumerate(values))

    return pattern.sub(replace_initializer, code)


def prefixed_model_code(exercise: str, model: RandomForestClassifier) -> str:
    code = cpp_compatible(m2c.export_to_c(model))
    prefix = exercise.replace("-", "_")
    # The generated helper names are global in C++; prefix them so both models
    # can live together in one generated model.h.
    for name in ("add_vectors", "mul_vector_number", "score"):
        code = re.sub(rf"\b{name}\b", f"{prefix}_{name}", code)
    return code.strip()


def train_one_exercise(
    exercise: str, files: dict[str, str]
) -> tuple[RandomForestClassifier, dict[str, object]]:
    train_x, train_y, test_x, test_y = [], [], [], []
    seen_classes: set[int] = set()

    for class_name, file_name in files.items():
        segments = load_recording(ROOT / file_name, class_name, exercise)
        for class_id, samples in segments:
            train_samples, test_samples = split_recording(samples)
            class_train = window_features(train_samples)
            class_test = window_features(test_samples)
            if len(class_train) == 0 or len(class_test) == 0:
                raise ValueError(f"{file_name}: not enough samples to create windows.")
            train_x.append(class_train)
            test_x.append(class_test)
            train_y.extend([class_id] * len(class_train))
            test_y.extend([class_id] * len(class_test))
            seen_classes.add(class_id)

    if seen_classes != {0, 1, 2}:
        missing = sorted({0, 1, 2} - seen_classes)
        raise ValueError(
            f"{exercise}: training data is missing classes "
            f"{[CLASS_NAMES[i] for i in missing]}."
        )

    x_train = np.vstack(train_x)
    x_test = np.vstack(test_x)
    y_train = np.asarray(train_y, dtype=np.int32)
    y_test = np.asarray(test_y, dtype=np.int32)

    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=6,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    report: dict[str, object] = {
        "accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=[0, 1, 2]
        ).tolist(),
        "training_windows": int(len(y_train)),
        "validation_windows": int(len(y_test)),
    }
    print(f"\n{exercise}: balanced validation accuracy "
          f"{report['accuracy'] * 100:.1f}%")
    print("Confusion matrix (rows=true, columns=predicted; "
          "NEUTRAL, GOOD FORM, BAD FORM):")
    print(np.asarray(report["confusion_matrix"]))
    return model, report

def main() -> int:
    missing = [
        file_name
        for files in DATASETS.values()
        for file_name in files.values()
        if not (ROOT / file_name).is_file()
    ]
    if missing:
        print("Training data is incomplete. No model was generated.", file=sys.stderr)
        print("Missing recordings:", file=sys.stderr)
        for file_name in missing:
            print(f"  - {file_name}", file=sys.stderr)
        print(
            "Each exercise needs separate NEUTRAL, GOOD FORM, and BAD FORM "
            "recordings. Only the neutral recording is shared; keep good/bad exercise recordings separate.",
            file=sys.stderr,
        )
        return 2

    generated = []
    for exercise, files in DATASETS.items():
        model, _ = train_one_exercise(exercise, files)
        generated.append(prefixed_model_code(exercise, model))

    header = [
        "#ifndef MODEL_H",
        "#define MODEL_H",
        "#define MODEL_HAS_EXERCISE_MODELS 1",
        "",
        "#include <string.h>",
        "",
        "// Output indices: 0=NEUTRAL, 1=GOOD FORM, 2=BAD FORM.",
        "// Generated by train_features.py; do not edit by hand.",
        "",
        *generated,
        "",
        "#endif  // MODEL_H",
        "",
    ]
    OUTPUT_HEADER.write_text("\n".join(header), encoding="utf-8")
    print(f"\nGenerated both exercise models in: {OUTPUT_HEADER}")
    print("The ESP32 sketch must dispatch to the score function for the "
          "dashboard-selected exercise.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
