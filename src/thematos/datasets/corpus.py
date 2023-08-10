from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pandas as pd
import tomotopy as tp
from hyfi import HyFI
from hyfi.run import RunConfig
from hyfi.task import BatchTaskConfig
from hyfi.utils.contexts import elapsed_timer

logger = HyFI.getLogger(__name__)


class Corpus(BatchTaskConfig):
    _config_group_: str = "/dataset"
    _config_name_: str = "topic_corpus"

    batch_name: str = "corpus"
    id_col: str = "id"
    text_col: str = "text"
    data_load: RunConfig = RunConfig(_config_name_="load_dataframe")
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
        self._corpus_ = tp.utils.Corpus(tokenizer=tp.utils.SimpleTokenizer())
        with elapsed_timer() as elapsed:
            self._load_docs(elapsed)
        self.save_ids()

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
            if self.id_file.exists():
                ids_ = HyFI.load_dataframe(self.id_file, verbose=self.verbose)
                self._doc_ids_ = ids_[self.id_col].values.tolist()
            else:
                self._doc_ids_ = self.data[self.id_col].values.tolist()
        return self._doc_ids_

    def _load_docs(self, elapsed):
        docs = self.docs
        self._corpus_.process(docs)
        logger.info("Corpus loaded in %s", elapsed)
        logger.info("Total %d documents are loaded.", len(docs))

    @property
    def id_file(self) -> Path:
        return self.batch_dir / f"{self.batch_name}_doc_ids.csv"

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
