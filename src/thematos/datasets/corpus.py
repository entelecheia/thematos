import logging
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pandas as pd
import tomotopy as tp
from hyfi import HyFI
from hyfi.run import RunConfig
from hyfi.task import BatchTaskConfig
from lexikanon.stopwords import Stopwords

from .ngrams import NgramConfig

logger = logging.getLogger(__name__)


class Corpus(BatchTaskConfig):
    _config_group_: str = "/dataset"
    _config_name_: str = "topic_corpus"

    task_name: str = "topic"
    batch_name: str = "corpus"
    id_col: str = "id"
    text_col: str = "text"
    timestamp_col: Optional[str] = None
    data_load: RunConfig = RunConfig(_config_name_="load_dataframe")
    stopwords: Optional[Stopwords] = Stopwords()
    ngrams: Optional[NgramConfig] = NgramConfig()
    ngramize: bool = True
    verbose: bool = False

    _data_: Optional[pd.DataFrame] = None
    _corpus_: Optional[tp.utils.Corpus] = None
    _doc_ids_: List[Any] = None

    @property
    def data(self) -> pd.DataFrame:
        if self._data_ is None:
            self._data_ = self.load_data()
        return self._data_

    def load_data(self) -> pd.DataFrame:
        return HyFI.partial(self.data_load.config)()

    @property
    def corpus(self) -> tp.utils.Corpus:
        if self._corpus_ is None:
            self.load()
        return self._corpus_

    def load(self):
        if self._corpus_:
            logger.infro("Corpus is already loaded")
            return
        logger.info("Loading corpus...")
        self._doc_ids_ = []
        self._corpus_ = tp.utils.Corpus(
            tokenizer=tp.utils.SimpleTokenizer(),
            stopwords=self.stopwords,
        )
        logger.info("Processing documents in the column '%s'...", self.text_col)
        self._corpus_.process(self.docs)
        logger.info("Total %d documents are loaded.", len(self.docs))
        if self.ngramize:
            self.concat_ngrams()
        self.save_ids()
        self.save_config()

    @property
    def docs(self) -> List[str]:
        doc = self.data[self.text_col][0]
        # check the type of the first document to see if it is a list of words
        # or a string
        # if the type is ndarray, convert it to string
        if isinstance(doc, str):
            return self.data[self.text_col].values.tolist()
        elif isinstance(doc, np.ndarray or list):
            return self.data[self.text_col].str.join(" ").values.tolist()

    @property
    def doc_ids(self) -> List[Any]:
        if not self._doc_ids_:
            self._doc_ids_ = self.data[self.id_col].values.tolist()
        return self._doc_ids_

    @property
    def id_file(self) -> Path:
        return self.batch_dir / f"{self.batch_name}_doc_ids.parquet"

    def save_ids(self):
        """Save doc ids to file"""
        data = pd.DataFrame(self.doc_ids, columns=[self.id_col])
        HyFI.save_dataframes(data, self.id_file, verbose=self.verbose)

    def __repr__(self):
        if self._doc_ids_:
            return f"<Corpus {len(self._doc_ids_)} documents>"
        else:
            return "<Corpus>"

    def __str__(self):
        if self._doc_ids_:
            return f"<Corpus {len(self._doc_ids_)} documents>"
        else:
            return "<Corpus>"

    def __bool__(self):
        return bool(self._doc_ids_)

    def concat_ngrams(self, delimiter: Optional[str] = None):
        if not self.ngrams:
            logger.info("N-grams config is not set. Skipping...")
            return
        # extract the n-gram candidates first
        if not delimiter:
            delimiter = self.ngrams.delimiter
        args = self.ngrams.kwargs
        args.pop("delimiter", None)
        if not self.ngrams.min_score:
            args.pop("min_score", None)
        if self.verbose:
            logger.info("Extracting n-grams...")
        cands = self.corpus.extract_ngrams(**args)
        if self.verbose:
            logger.info("Total %d n-grams are extracted.", len(cands))
            for cand in cands[:10]:
                logger.info(cand)
            for cand in cands[-10:]:
                logger.info(cand)
        # concat n-grams in the corpus
        if self.verbose:
            logger.info("Concatenating n-grams...")
            logger.info(self.corpus[0])
            logger.info(self.corpus[-1])
        self.corpus.concat_ngrams(cands, delimiter=delimiter)
        if self.verbose:
            logger.info(self.corpus[0])
            logger.info(self.corpus[-1])
            logger.info("Total %d documents are n-gramized.", len(self.corpus))
