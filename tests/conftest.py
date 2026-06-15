"""Shared pytest fixtures."""

from __future__ import annotations

import pytest


GT_DATA_ROOT = "data/GT"


@pytest.fixture(scope="session")
def gt_data_root() -> str:
    return GT_DATA_ROOT


@pytest.fixture(scope="session")
def train_split(gt_data_root: str) -> str:
    return f"{gt_data_root}/splits/train.txt"


@pytest.fixture(scope="session")
def val_split(gt_data_root: str) -> str:
    return f"{gt_data_root}/splits/val.txt"
