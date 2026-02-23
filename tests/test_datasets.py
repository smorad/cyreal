"""Tests covering dataset utilities and disk sources."""
from __future__ import annotations

import gzip
import pickle
import struct
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cyreal.transforms import BatchTransform
from cyreal.datasets import (
    CelebADataset,
    CIFAR10Dataset,
    CIFAR100Dataset,
    EMNISTDataset,
    FashionMNISTDataset,
    KMNISTDataset,
    MNISTDataset,
)


def _write_idx_images(path, images):
    with gzip.open(path, "wb") as f:
        num, rows, cols = images.shape
        f.write(struct.pack(">IIII", 2051, num, rows, cols))
        f.write(images.tobytes())


def _write_idx_labels(path, labels):
    with gzip.open(path, "wb") as f:
        num = labels.shape[0]
        f.write(struct.pack(">II", 2049, num))
        f.write(labels.tobytes())


def _seed_fake_cifar10(tmp_path, split="train", samples_per_batch=1):
    archive_path = tmp_path / "cifar-10-python.tar.gz"
    archive_path.write_bytes(b"")
    batches_dir = tmp_path / "cifar-10-batches-py"
    batches_dir.mkdir(parents=True, exist_ok=True)

    if split == "train":
        names = [f"data_batch_{i}" for i in range(1, 6)]
    elif split == "test":
        names = ["test_batch"]
    else:
        raise ValueError("split must be 'train' or 'test'.")

    current_label = 0
    for name in names:
        data = []
        labels = []
        for _ in range(samples_per_batch):
            pixel_value = current_label % 256
            image = np.full((3, 32, 32), pixel_value, dtype=np.uint8)
            data.append(image.reshape(-1))
            labels.append(current_label)
            current_label += 1
        batch = {"data": np.stack(data, axis=0), "labels": labels}
        with open(batches_dir / name, "wb") as f:
            pickle.dump(batch, f, protocol=2)


def _seed_fake_cifar100(tmp_path, split="train", samples=4):
    archive_path = tmp_path / "cifar-100-python.tar.gz"
    archive_path.write_bytes(b"")
    target_dir = tmp_path / "cifar-100-python"
    target_dir.mkdir(parents=True, exist_ok=True)

    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'.")

    data = []
    fine_labels = []
    coarse_labels = []
    for idx in range(samples):
        pixel_value = idx % 256
        image = np.full((3, 32, 32), pixel_value, dtype=np.uint8)
        data.append(image.reshape(-1))
        fine_labels.append(idx)
        coarse_labels.append(idx % 5)
    batch = {
        "data": np.stack(data, axis=0),
        "fine_labels": fine_labels,
        "coarse_labels": coarse_labels,
    }
    with open(target_dir / split, "wb") as f:
        pickle.dump(batch, f, protocol=2)


def _seed_fake_celeba(tmp_path: Path):
    pil = pytest.importorskip("PIL.Image")
    image_module = pil

    image_dir = tmp_path / "img_align_celeba"
    image_dir.mkdir(parents=True, exist_ok=True)

    samples = [
        ("000001.jpg", 0, np.array([1, -1, 1], dtype=np.int8), 10),
        ("000002.jpg", 1, np.array([-1, 1, -1], dtype=np.int8), 40),
        ("000003.jpg", 2, np.array([1, 1, -1], dtype=np.int8), 90),
    ]

    for name, _, _, pixel in samples:
        image = np.full((4, 5, 3), pixel, dtype=np.uint8)
        image_module.fromarray(image, mode="RGB").save(image_dir / name)

    with (tmp_path / "list_eval_partition.txt").open("w", encoding="utf-8") as f:
        f.write(f"{len(samples)}\n")
        for name, partition, _, _ in samples:
            f.write(f"{name} {partition}\n")

    with (tmp_path / "list_attr_celeba.txt").open("w", encoding="utf-8") as f:
        f.write(f"{len(samples)}\n")
        f.write("Smiling Young Male\n")
        for name, _, attrs, _ in samples:
            values = " ".join(str(int(v)) for v in attrs)
            f.write(f"{name} {values}\n")


MNIST_LIKE_DATASETS = [
    pytest.param(MNISTDataset, {}, id="mnist"),
    pytest.param(FashionMNISTDataset, {}, id="fashion"),
    pytest.param(KMNISTDataset, {}, id="kmnist"),
    pytest.param(EMNISTDataset, {"subset": "letters"}, id="emnist-letters"),
]


@pytest.mark.parametrize("dataset_cls,extra_kwargs", MNIST_LIKE_DATASETS)
def test_idx_dataset_reads_cached_idx(tmp_path, dataset_cls, extra_kwargs):
    num, rows, cols = 3, 2, 2
    images = np.arange(num * rows * cols, dtype=np.uint8).reshape(num, rows, cols)
    labels = np.arange(num, dtype=np.uint8)

    images_path = tmp_path / "train_images.gz"
    labels_path = tmp_path / "train_labels.gz"
    _write_idx_images(images_path, images)
    _write_idx_labels(labels_path, labels)

    dataset = dataset_cls(split="train", cache_dir=tmp_path, **extra_kwargs)
    assert len(dataset) == num
    example = dataset[0]
    img = example["image"]
    label = example["label"]
    assert img.shape == (rows, cols, 1)
    assert img.dtype == np.uint8
    assert label == 0
    np.testing.assert_array_equal(img[..., 0], images[0])


@pytest.mark.parametrize("dataset_cls,extra_kwargs", MNIST_LIKE_DATASETS)
def test_idx_disk_sources_stream_from_disk(tmp_path, dataset_cls, extra_kwargs):
    num, rows, cols = 3, 2, 2
    images = np.arange(num * rows * cols, dtype=np.uint8).reshape(num, rows, cols)
    labels = np.arange(num, dtype=np.uint8)

    images_path = tmp_path / "train_images.gz"
    labels_path = tmp_path / "train_labels.gz"
    _write_idx_images(images_path, images)
    _write_idx_labels(labels_path, labels)

    common_kwargs = {
        "split": "train",
        "cache_dir": tmp_path,
        "ordering": "sequential",
        "prefetch_size": 2,
    }
    common_kwargs.update(extra_kwargs)
    source = dataset_cls.make_disk_source(**common_kwargs)

    batched = BatchTransform(
        batch_size=2,
        element_spec_override=source.element_spec(),
    )(source)

    state = batched.init_state(jax.random.PRNGKey(0))
    batch, mask, state = batched.next(state)
    np.testing.assert_array_equal(np.asarray(batch["image"][..., 0]), images[:2])
    np.testing.assert_array_equal(np.asarray(batch["label"]), np.array([0, 1], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(mask), np.array([True, True]))

    batch2, mask2, _ = batched.next(state)
    np.testing.assert_array_equal(np.asarray(batch2["image"][0, ..., 0]), images[2])
    np.testing.assert_array_equal(np.asarray(batch2["label"][:1]), np.array([2], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(mask2), np.array([True, False]))


def test_cifar10_disk_source_streams_from_disk(tmp_path):
    _seed_fake_cifar10(tmp_path, split="test", samples_per_batch=3)

    source = CIFAR10Dataset.make_disk_source(
        split="test",
        cache_dir=tmp_path,
        ordering="sequential",
        prefetch_size=2,
    )
    batched = BatchTransform(
        batch_size=2,
        element_spec_override=source.element_spec(),
    )(source)

    state = batched.init_state(jax.random.PRNGKey(0))
    batch, mask, state = batched.next(state)
    np.testing.assert_array_equal(np.asarray(batch["label"]), np.array([0, 1], dtype=np.int32))
    np.testing.assert_array_equal(
        np.asarray(batch["image"][0, 0, 0]),
        np.array([0, 0, 0], dtype=np.uint8),
    )
    np.testing.assert_array_equal(
        np.asarray(batch["image"][1, 0, 0]),
        np.array([1, 1, 1], dtype=np.uint8),
    )
    np.testing.assert_array_equal(np.asarray(mask), np.array([True, True]))

    batch2, mask2, _ = batched.next(state)
    np.testing.assert_array_equal(np.asarray(batch2["label"][:1]), np.array([2], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(mask2), np.array([True, False]))
    np.testing.assert_array_equal(
        np.asarray(batch2["image"][1]),
        np.zeros((32, 32, 3), dtype=np.uint8),
    )


def test_cifar100_disk_source_streams_from_disk(tmp_path):
    _seed_fake_cifar100(tmp_path, split="test", samples=3)

    source = CIFAR100Dataset.make_disk_source(
        split="test",
        cache_dir=tmp_path,
        ordering="sequential",
        prefetch_size=2,
    )

    batched = BatchTransform(
        batch_size=2,
        element_spec_override=source.element_spec(),
    )(source)

    state = batched.init_state(jax.random.PRNGKey(0))
    batch, mask, state = batched.next(state)
    np.testing.assert_array_equal(np.asarray(batch["label"]), np.array([0, 1], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(batch["coarse_label"]), np.array([0, 1], dtype=np.int32))
    np.testing.assert_array_equal(
        np.asarray(batch["image"][0, 0, 0]),
        np.array([0, 0, 0], dtype=np.uint8),
    )
    np.testing.assert_array_equal(
        np.asarray(batch["image"][1, 0, 0]),
        np.array([1, 1, 1], dtype=np.uint8),
    )
    np.testing.assert_array_equal(np.asarray(mask), np.array([True, True]))

    batch2, mask2, _ = batched.next(state)
    np.testing.assert_array_equal(np.asarray(batch2["label"][:1]), np.array([2], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(batch2["coarse_label"][:1]), np.array([2], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(mask2), np.array([True, False]))


def test_celeba_dataset_reads_local_layout(tmp_path):
    _seed_fake_celeba(tmp_path)

    dataset = CelebADataset(split="valid", data_dir=tmp_path)
    assert len(dataset) == 1

    sample = dataset[0]
    assert sample["image"].shape == (4, 5, 3)
    assert sample["image"].dtype == np.uint8
    np.testing.assert_array_equal(
        np.asarray(sample["attributes"]),
        np.array([0, 1, 0], dtype=np.int32),
    )
    assert dataset.attribute_names == ("Smiling", "Young", "Male")


def test_celeba_disk_source_streams_from_disk(tmp_path):
    _seed_fake_celeba(tmp_path)

    source = CelebADataset.make_disk_source(
        split="test",
        data_dir=tmp_path,
        ordering="sequential",
        prefetch_size=2,
    )
    batched = BatchTransform(
        batch_size=2,
        element_spec_override=source.element_spec(),
    )(source)

    state = batched.init_state(jax.random.PRNGKey(0))
    batch, mask, _ = batched.next(state)
    np.testing.assert_array_equal(np.asarray(mask), np.array([True, False]))
    np.testing.assert_array_equal(
        np.asarray(batch["attributes"][0]),
        np.array([1, 1, 0], dtype=np.int32),
    )
    assert np.asarray(batch["image"])[0].shape == (4, 5, 3)