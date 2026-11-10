"""Train Dooboo's production Random Forest classifier."""
import argparse
import csv
import os
import tempfile
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GroupShuffleSplit, train_test_split

try:
    from .features import FEATURE_COUNT, FEATURE_NAMES, FEATURE_VERSION, extract_features
except ImportError:  # Running as a script from the classifier directory.
    from features import FEATURE_COUNT, FEATURE_NAMES, FEATURE_VERSION, extract_features

CLASSES = (
    "far_loud", "far_normal", "far_small",
    "near_loud", "near_normal", "near_small",
)
SUPPORTED_EXTENSIONS = {".wav", ".m4a", ".mp3", ".flac", ".ogg"}


def _read_group_manifest(manifest_path: Path, data_dir: Path) -> dict[str, str]:
    groups = {}
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"file", "group_id"}.issubset(reader.fieldnames):
            raise ValueError("Group manifest must contain 'file' and 'group_id' columns.")
        for row in reader:
            relative_name = Path(row["file"]).as_posix()
            group_id = row["group_id"].strip()
            if not relative_name or not group_id:
                raise ValueError("Group manifest contains an empty file or group_id.")
            if relative_name in groups:
                raise ValueError(f"Duplicate file in group manifest: {relative_name}")
            groups[relative_name] = group_id
    return groups


def load_dataset(data_dir: Path, groups_csv: Path | None = None):
    rows, labels, filenames = [], [], []
    failures = []
    group_map = _read_group_manifest(groups_csv, data_dir) if groups_csv else {}

    for label in CLASSES:
        class_dir = data_dir / label
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")
        for audio_path in sorted(class_dir.iterdir()):
            if audio_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            relative_name = audio_path.relative_to(data_dir).as_posix()
            try:
                rows.append(extract_features(audio_path))
                labels.append(label)
                filenames.append(relative_name)
            except Exception as exc:
                failures.append(f"{relative_name}: {exc}")

    for failure in failures:
        print(f"Skipped unreadable file: {failure}")
    if not rows:
        raise RuntimeError(f"No readable supported audio files found under {data_dir}")

    counts = {label: labels.count(label) for label in CLASSES}
    insufficient = [label for label, count in counts.items() if count < 5]
    if insufficient:
        raise RuntimeError(
            "At least 5 readable recordings per class are required for a "
            f"stratified holdout split; insufficient classes: {insufficient}"
        )

    if groups_csv:
        missing = sorted(set(filenames) - set(group_map))
        if missing:
            raise ValueError(
                "Group manifest is missing entries for: " + ", ".join(missing[:10])
            )
        unknown = sorted(set(group_map) - set(filenames))
        if unknown:
            print(f"Warning: ignoring {len(unknown)} manifest entries not in this dataset.")
        groups = np.asarray([group_map[name] for name in filenames])
    else:
        # One file must represent one independent original recording.
        groups = np.asarray(filenames)

    return np.vstack(rows), np.asarray(labels), groups


def split_dataset(X, y, groups, use_group_split: bool):
    if not use_group_split:
        return train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

    splitter = GroupShuffleSplit(n_splits=100, test_size=0.2, random_state=42)
    all_classes = set(CLASSES)
    best_split = None
    best_distance = float("inf")
    for train_indices, test_indices in splitter.split(X, y, groups):
        if set(y[train_indices]) != all_classes or set(y[test_indices]) != all_classes:
            continue
        distance = abs(len(test_indices) / len(y) - 0.2)
        if distance < best_distance:
            best_distance = distance
            best_split = (train_indices, test_indices)
    if best_split is None:
        raise RuntimeError(
            "Could not create a group-safe split containing all six classes in "
            "both train and test sets. Add more independent source recordings "
            "or correct the group manifest."
        )
    train_indices, test_indices = best_split
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    classifier_dir = Path(__file__).resolve().parent
    repository_dir = classifier_dir.parent
    parser.add_argument("--data-dir", type=Path, default=repository_dir / "data")
    parser.add_argument(
        "--output", type=Path, default=classifier_dir / "random_forest_model.joblib"
    )
    parser.add_argument(
        "--groups-csv",
        type=Path,
        help="Optional CSV with file,group_id columns to keep related audio crops together.",
    )
    args = parser.parse_args()

    X, y, groups = load_dataset(args.data_dir, args.groups_csv)
    X_train, X_test, y_train, y_test = split_dataset(
        X, y, groups, use_group_split=args.groups_csv is not None
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Samples: {len(y)} | Features: {X.shape[1]}")
    print(f"Training: {len(y_train)} | Test: {len(y_test)}")
    print(f"Holdout accuracy: {accuracy:.4f}")
    print(
        classification_report(
            y_test, predictions, labels=list(CLASSES), zero_division=0
        )
    )

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "feature_version": FEATURE_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "feature_count": FEATURE_COUNT,
        "classes": list(CLASSES),
        "holdout_accuracy": float(accuracy),
        "training_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
    }

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
        joblib.dump(bundle, temp_path)
        os.replace(temp_path, output_path)
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()
    print(f"Saved model bundle: {output_path}")


if __name__ == "__main__":
    main()
