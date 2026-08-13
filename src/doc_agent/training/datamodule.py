"""Training — Lightning datamodule"""
from __future__ import annotations

import lightning as L

from ..contracts import *  # noqa


class DocDataModule(L.LightningDataModule):
    def setup(self, stage: str | None = None) -> None:
        raise NotImplementedError("Training: datamodule.setup (split by document)")

