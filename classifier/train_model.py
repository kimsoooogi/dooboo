"""Train the production classifier from class-named audio directories."""
import argparse
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from features import FEATURE_NAMES, FEATURE_VERSION, extract_features

CLASSES = ("far_loud", "far_normal", "far_small", "near_loud", "near_normal", "near_small")
EXTENSIONS = {".wav", ".m4a", ".mp3", ".flac", ".ogg"}


def load_dataset(data_dir: Path):
    rows, labels, failures = [], [], []
    for label in CLASSES:
        folder = data_dir / label
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing class directory: {folder}")
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() not in EXTENSIONS:
                continue
            try:
                rows.append(extract_features(path))
                labels.append(label)
            except Exception as exc:
                failures.append(f"{path}: {exc}")
    for failure in failures:
        print(f"Skipped unreadable file: {failure}")
    counts = {label: labels.count(label) for label in CLASSES}
    insufficient = [label for label, count in counts.items() if count < 2]
    if insufficient:
        raise RuntimeError(f"Need at least 2 readable files per class: {insufficient}")
    return np.vstack(rows), np.asarray(labels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("../data"))
    parser.add_argument("--output", type=Path, default=Path("random_forest_model.joblib"))
    args = parser.parse_args()
    X, y = load_dataset(args.data_dir)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=300, random_state=42, class_weight="balanced", n_jobs=-1
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    print(f"Samples: {len(y)} | Features: {X.shape[1]} | Holdout accuracy: {accuracy:.4f}")
    print(classification_report(y_test, pred, labels=list(CLASSES), zero_division=0))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model, "feature_version": FEATURE_VERSION,
        "feature_names": FEATURE_NAMES, "feature_count": len(FEATURE_NAMES),
        "classes": list(CLASSES), "holdout_accuracy": float(accuracy),
        "training_samples": int(len(y_train)), "test_samples": int(len(y_test)),
    }, args.output)
    print(f"Saved model bundle: {args.output.resolve()}")


if __name__ == "__main__":
    main()
