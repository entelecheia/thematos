import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tomotopy as tp
from lexikanon.stopwords import Stopwords
from pydantic import Field, PrivateAttr

from thematos._base import Model

from .ngrams import NgramConfig

logger = logging.getLogger(__name__)


def read_dataframe(path: str | Path) -> pd.DataFrame:
    """Read a ``.parquet``, ``.csv``, or ``.jsonl`` file into a DataFrame."""
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    raise ValueError(f"unsupported data file type: {path}")


class Corpus(Model):
    """Documents from a DataFrame (``data``) or a file (``data_file``), as a tomotopy corpus."""

    data: pd.DataFrame | None = Field(None, exclude=True, repr=False)
    data_file: Path | None = None
    id_col: str = "id"
    text_col: str = "text"
    stopwords: Stopwords | None = Field(default_factory=Stopwords)
    ngrams: NgramConfig | None = Field(default_factory=NgramConfig)
    ngramize: bool = True
    verbose: bool = False

    _corpus_: tp.utils.Corpus | None = PrivateAttr(None)

    def load_data(self) -> pd.DataFrame:
        if self.data is None:
            if self.data_file is None:
                raise ValueError("Corpus needs `data` or `data_file`")
            self.data = read_dataframe(self.data_file)
            if self.verbose:
                logger.info(self.data[self.text_col].head())
        return self.data

    @property
    def corpus(self) -> tp.utils.Corpus:
        if self._corpus_ is None:
            self.load()
        return self._corpus_

    def load(self) -> "Corpus":
        """Tokenize the documents (and concatenate n-grams) into a tomotopy corpus."""
        if self._corpus_ is not None:
            logger.info("Corpus is already loaded")
            return self
        logger.info("Loading corpus...")
        corpus = tp.utils.Corpus(tokenizer=tp.utils.SimpleTokenizer(), stopwords=self.stopwords)
        logger.info("Processing documents in the column '%s'...", self.text_col)
        docs = self.docs
        corpus.process(docs)
        logger.info("Total %d documents are loaded.", len(docs))
        self._corpus_ = corpus
        if self.ngramize:
            self.concat_ngrams()
        return self

    @property
    def docs(self) -> list[str]:
        texts = self.load_data()[self.text_col]
        doc = texts.iloc[0]
        if self.verbose:
            logger.info("The first document: %s", doc)
        # a column of token lists (ndarray after a parquet round trip) is joined into strings
        if isinstance(doc, np.ndarray | list):
            return texts.str.join(" ").tolist()
        return texts.tolist()

    @property
    def doc_ids(self) -> list[Any]:
        return self.load_data()[self.id_col].tolist()

    def __repr__(self) -> str:
        if self._corpus_ is not None:
            return f"<Corpus {len(self._corpus_)} documents>"
        return "<Corpus>"

    __str__ = __repr__

    def __bool__(self) -> bool:
        return self._corpus_ is not None and len(self._corpus_) > 0

    def concat_ngrams(self, delimiter: str | None = None) -> None:
        if not self.ngrams:
            logger.info("N-grams config is not set. Skipping...")
            return
        args = self.ngrams.model_dump()
        delimiter = delimiter or args.pop("delimiter")
        args.pop("delimiter", None)
        if not self.ngrams.min_score:
            args.pop("min_score", None)
        corpus = self.corpus
        cands = corpus.extract_ngrams(**args)
        if self.verbose:
            logger.info("Total %d n-grams are extracted.", len(cands))
            for cand in cands[:10] + cands[-10:]:
                logger.info(cand)
        corpus.concat_ngrams(cands, delimiter=delimiter)
        if self.verbose:
            logger.info("Total %d documents are n-gramized.", len(corpus))
