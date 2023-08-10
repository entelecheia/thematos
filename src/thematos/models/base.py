import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import tomotopy as tp
from hyfi import HyFI
from hyfi.composer import BaseModel

from thematos.datasets import Corpus

from .prior import WordPrior
from .types import IDF, ONE, PMI, CoherenceMetrics, ModelSummary

logger = HyFI.getLogger(__name__)


class TopicModel(BaseModel):
    model_name: str = "Topic"
    model_type: str = "BASE"
    burn_in: int = 0
    interval: int = 10
    iterations: int = 100

    coherence_metrics: List[str] = ["u_mass", "c_uci", "c_npmi", "c_v"]
    seed: int = None
    eval_coherence: bool = False
    set_word_prior: bool = False
    save: bool = True
    save_full: bool = True
    verbose: bool = False

    # internal attributes
    _train_timestamp_: Optional[str] = None
    _output_dir_: Optional[Path] = None
    _model_dir_: Optional[Path] = None
    _corpus_: Optional[Corpus] = None
    _wordprior_: Optional[WordPrior] = None
    _coh_values_: Optional[CoherenceMetrics] = None
    _model_summary_: Optional[ModelSummary] = None
    _ll_per_words_: List[Tuple[int, float]] = []

    def initialize(
        self,
        corpus: Corpus,
        model_dir: Path,
        output_dir: Path,
        word_prior: Optional[WordPrior] = None,
    ):
        self._corpus_ = corpus
        self._wordprior_ = word_prior
        self._output_dir_ = output_dir
        self._model_dir_ = model_dir

    @property
    def model_id(self) -> str:
        model_type = self.model_type.upper()
        margs = [self.model_name, model_type]
        if self.k:
            margs.append(f"k({self.k})")
        return "_".join(margs)

    @property
    def model_summary(self) -> ModelSummary:
        if self._model_summary_ is None:
            raise ValueError("Model has not been trained yet.")
        return self._model_summary_

    @property
    def cohrence_values(self) -> CoherenceMetrics:
        if self._coh_values_ is None:
            raise ValueError("Model has not been trained yet.")
        return self._coh_values_

    @property
    def train_timestamp(self) -> str:
        if self._train_timestamp_ is None:
            raise ValueError("Model has not been trained yet.")
        return self._train_timestamp_

    @property
    def corpus(self) -> tp.utils.Corpus:
        if self._corpus_ is None:
            raise ValueError("Model has not been initialized yet.")
        return self._corpus_.corpus

    @property
    def corpus_ids(self) -> pd.DataFrame:
        if self._corpus_ is None:
            raise ValueError("Model has not been initialized yet.")
        return self._corpus_.corpus_ids

    @property
    def wordprior(self) -> Optional[WordPrior]:
        return self._wordprior_

    @property
    def output_dir(self) -> Path:
        if self._output_dir_ is None:
            raise ValueError("Model has not been initialized yet.")
        return self._output_dir_

    @property
    def model_dir(self) -> Path:
        if self._model_dir_ is None:
            raise ValueError("Model has not been initialized yet.")
        return self._model_dir_

    @property
    def model_file(self) -> str:
        return f"{self.model_id}-{self.train_timestamp}.mdl"

    @property
    def ll_per_words_file(self) -> str:
        return f"{self.model_id}-ll_per_word-{self.train_timestamp}.csv"

    @property
    def ll_per_words_fig_file(self) -> str:
        return f"{self.model_id}-ll_per_word-{self.train_timestamp}.png"

    @property
    def topic_dists_file(self) -> str:
        return f"{self.model_id}-topic_dists-{self.train_timestamp}.csv"

    @property
    def train_summary_file(self) -> str:
        return f"{self.model_id}-summary-{self.train_timestamp}.txt"

    @property
    def ll_per_words(self) -> pd.DataFrame:
        if not self._ll_per_words_:
            raise ValueError("Model not trained yet.")
        return pd.DataFrame(self._ll_per_words_, columns=["iter", "ll_per_word"])
