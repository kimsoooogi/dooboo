import numpy as np
import pytest

from classifier.train_model import CLASSES, _read_group_manifest, split_dataset


def test_group_split_keeps_groups_disjoint_and_all_labels(tmp_path):
    labels = np.repeat(np.asarray(CLASSES), 8)
    features = np.arange(len(labels) * 4).reshape(len(labels), 4)
    groups = np.asarray([f"source-{i}" for i in range(len(labels))])

    X_train, X_test, y_train, y_test = split_dataset(
        features, labels, groups, use_group_split=True
    )

    train_groups = set(groups[np.isin(features[:, 0], X_train[:, 0])])
    test_groups = set(groups[np.isin(features[:, 0], X_test[:, 0])])
    assert train_groups.isdisjoint(test_groups)
    assert set(y_train) == set(CLASSES)
    assert set(y_test) == set(CLASSES)


def test_group_manifest_requires_columns(tmp_path):
    manifest = tmp_path / "groups.csv"
    manifest.write_text("filename,group\na.wav,one\n", encoding="utf-8")
    with pytest.raises(ValueError, match="file.*group_id"):
        _read_group_manifest(manifest, tmp_path)


def test_group_manifest_rejects_duplicate_files(tmp_path):
    manifest = tmp_path / "groups.csv"
    manifest.write_text(
        "file,group_id\nclass/a.wav,one\nclass/a.wav,two\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate file"):
        _read_group_manifest(manifest, tmp_path)
