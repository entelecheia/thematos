"""Workflow steps: small functions that chain a load, corpus, and model through HyFI ``steps:``.

Each step returns what the next one takes as its first argument (HyFI ``pipe``). They work the
same when called from plain Python.
"""

import itertools
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from thematos.datasets import Corpus
from thematos.models import TopicModel

logger = logging.getLogger(__name__)


def build_corpus(data: pd.DataFrame | None = None, *, corpus: Corpus) -> Corpus:
    """Tokenize ``data`` (or ``corpus.data_file`` when no data is piped in) into ``corpus``."""
    if data is not None:
        corpus.data = data
        corpus._corpus_ = None  # rebuild from the new data
    return corpus.load()


def train(corpus: Corpus, *, model: TopicModel) -> TopicModel:
    """Train ``model`` on ``corpus`` and save the model file and its distributions."""
    return model.train(corpus).save()


def summarize(model: TopicModel) -> TopicModel:
    return model.summarize()


def plot(model: TopicModel) -> TopicModel:
    return model.plot()


def sweep(
    corpus: Corpus,
    *,
    model: TopicModel,
    k: list[int],
    alpha: list[float] | None = None,
    eta: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Train, save, summarize, and plot one model per (k, alpha, eta) combination.

    Models are named ``<name>(0)``, ``<name>(1)``, ... in sweep order, and every model's summary
    is appended to ``<output_dir>/model-summary.jsonl``. Returns the summaries.
    """
    alpha = alpha or [model.model_args.alpha]
    eta = eta or [model.model_args.eta]
    summaries = []
    for i, (k_, alpha_, eta_) in enumerate(itertools.product(k, alpha, eta)):
        run = model.model_copy(
            update={
                "name": f"{model.name}({i})",
                "model_args": model.model_args.model_copy(
                    update={"k": k_, "alpha": alpha_, "eta": eta_}
                ),
            }
        )
        logger.info("Sweep run %d: k=%s alpha=%s eta=%s", i, k_, alpha_, eta_)
        run.train(corpus).save().summarize().plot()
        summaries.append(run.model_summary_dict)
    return summaries


def infer(
    corpus: Corpus,
    *,
    model: TopicModel,
    output_file: str | Path | None = None,
    iterations: int = 100,
    tolerance: float = -1,
    num_workers: int = 0,
    together: bool = False,
) -> pd.DataFrame:
    """Load ``model`` (as saved under its ``output_dir``) and infer topics for ``corpus``."""
    model.load()
    return model.infer(
        corpus,
        output_file=output_file,
        iterations=iterations,
        tolerance=tolerance,
        num_workers=num_workers,
        together=together,
    )
