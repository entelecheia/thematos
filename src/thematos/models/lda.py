import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import tomotopy as tp
from hyfi import HyFI
from hyfi.composer import BaseModel
from tqdm.auto import tqdm

from .base import TopicModel
from .types import IDF, ONE, PMI, CoherenceMetrics, ModelSummary

logger = HyFI.getLogger(__name__)


class LdaConfig(BaseModel):
    tw: str = IDF
    min_cf: int = 5
    min_df: int = 0
    rm_top: int = 0
    k: Optional[int] = None
    alpha: float = 0.1
    eta: float = 0.01


class LdaModel(TopicModel):
    model_type: str = "LDA"
    model_config: LdaConfig = LdaConfig()

    # internal attributes
    _model_: Optional[tp.LDAModel] = None

    @property
    def model_id(self) -> str:
        model_type = self.model_type.upper()
        margs = [model_type, self.batch_name, f"k({self.model_config.k})"]
        return "_".join(margs)

    @property
    def model(self) -> tp.LDAModel:
        if self._model_ is None:
            self._model_ = tp.LDAModel(
                corpus=self.corpus,
                seed=self.seed,
                **self.model_config.model_dump(),
            )
        return self._model_

    def _train(self, model: tp.LDAModel) -> None:
        train_cfg = self.train_config
        model.burn_in = train_cfg.burn_in
        model.train(0)
        logger.info("Number of docs: %s", len(model.docs))
        logger.info("Vocab size: %s", model.num_vocabs)
        logger.info("Number of words: %s", model.num_words)
        logger.info("Removed top words: %s", model.removed_top_words)
        logger.info(
            "Training model by iterating over the corpus %s times, %s iterations at a time",
            self.iterations,
            self.interval,
        )

        ll_per_words = []
        for i in range(0, train_cfg.iterations, train_cfg.interval):
            model.train(train_cfg.interval)
            logger.info("Iteration: %s\tLog-likelihood: %s", i, model.ll_per_word)
            ll_per_words.append((i, model.ll_per_word))
        self._ll_per_words_ = ll_per_words
        if self.verbose:
            model.summary()
        self._model_ = model
