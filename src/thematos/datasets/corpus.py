import random
from pathlib import Path
from typing import Any, List, Optional, Union

import pandas as pd
import tomotopy as tp
from hyfi import HyFI
from hyfi.run import RunConfig
from hyfi.task import BatchTaskConfig
from hyfi.utils.contexts import elapsed_timer
from lexikanon.stopwords import Stopwords
from tqdm.auto import tqdm

logger = HyFI.getLogger(__name__)


class Corpus(BatchTaskConfig):
    _config_group_: str = "/dataset"
    _config_name_: str = "__init__"

    batch_name: str = "corpus"
    id_col: str = "id"
    text_col: str = "text"
    stopwords: Stopwords = Stopwords()
    data_load: RunConfig = RunConfig(_config_name_="load_dataframe")
    sample_size: Optional[Union[int, float]] = None
    min_num_words: int = 5
    min_word_len: int = 2
    verbose: bool = False

    _data_: Optional[pd.DataFrame] = None
    _raw_corpus_: Optional[tp.utils.Corpus] = None
    _raw_coprus_ids_: List[Any] = None
    _corpus_: Optional[tp.utils.Corpus] = None
    _corpus_ids_: List[Any] = None

    _loaded_: bool = False

    @property
    def data(self) -> pd.DataFrame:
        if self._data_ is None:
            self._data_ = self.load_data()
        return self._data_

    def load_data(self) -> pd.DataFrame:
        return HyFI.partial(self.data_load.config)()

    @property
    def raw_corpus(self) -> tp.utils.Corpus:
        if self._raw_corpus_ is None:
            self.load_raw_corpus()
        return self._raw_corpus_

    def load_raw_corpus(self):
        self._raw_corpus_ = tp.utils.Corpus(tokenizer=tp.utils.SimpleTokenizer())
        with elapsed_timer() as elapsed:
            data = self.data
            self._raw_coprus_ids_ = data[self.id_col].values.tolist()
            self._raw_corpus_.process(data[self.text_col].values.tolist())
            logger.info("Raw corpus loaded in %s", elapsed)

    @property
    def corpus(self) -> tp.utils.Corpus:
        if self._corpus_ is None:
            self.load()
        return self._corpus_

    def load(
        self,
        sample_size: Optional[Union[int, float]] = None,
        sample_seed: Optional[int] = None,
        min_num_words: Optional[int] = None,
        min_word_len: Optional[int] = None,
        reload_corpus: bool = False,
    ):
        sample_size = sample_size or self.sample_size
        sample_seed = sample_seed or self.seed
        min_word_len = min_word_len or self.min_word_len
        min_num_words = min_num_words or self.min_num_words

        if self._corpus_ and self.sample_size == sample_size and not reload_corpus:
            logger.infro("Corpus is already loaded with sample_ratio: %s", sample_size)
            return True
        else:
            logger.info("Loading corpus with sample_ratio: %s", sample_size)
        assert self.stopwords, "Load stopwords first"
        if sample_size and sample_size < 1.0 and sample_size > 0.0:
            sample_size = int(len(self.raw_corpus) * sample_size)
        if sample_size:
            random.seed(sample_seed)
            docs = random.sample(self.raw_corpus, sample_size)
        else:
            docs = self.raw_corpus
            self.sample_size = 1.0
        self._corpus_ = tp.utils.Corpus()
        self._corpus_ids_ = []

        num_discarded = 0
        for i_doc, doc in tqdm(enumerate(docs)):
            words = [w for w in doc if not self.stopwords(w) and len(w) >= min_word_len]
            if len(words) > min_num_words:
                self._corpus_.add_doc(words=words)
                self._corpus_ids_.append(self._raw_coprus_ids_[i_doc])
            else:
                if self.verbose:
                    logger.info(
                        "Skipped document %s with %d words",
                        self._raw_coprus_ids_[i_doc],
                        len(words),
                    )
                num_discarded += 1
        logger.info("Total %d documents are loaded.", i_doc - num_discarded + 1)
        if self.verbose:
            logger.info("Total %d documents are discarded.", num_discarded)
        self.save()

    @property
    def id_file(self) -> Path:
        if self.sample_size:
            sample_size = (
                int(self.sample_size * 100) if sample_size < 1.0 else sample_size
            )
            id_file = f"{self.batch_name}_sample{sample_size}_ids.csv"
        else:
            id_file = f"{self.batch_name}_ids.csv"
        return self.batch_dir / id_file

    def save(self):
        """Save corpus ids to file"""
        data = pd.DataFrame(self._corpus_ids_, columns=[self.id_col])
        HyFI.save_dataframes(data, self.id_file, verbose=self.verbose)
        if self.verbose:
            logger.info("Saved %d words to %s", len(self._words_), self.words_path)

    @property
    def corpus_ids(self) -> pd.DataFrame:
        if self._corpus_ids_:
            return pd.DataFrame(self._corpus_ids_, columns=[self.id_col])
        if self.id_file.exists():
            return HyFI.load_dataframe(self.id_file, verbose=self.verbose)
        else:
            raise ValueError("Corpus is not loaded yet")

    def __repr__(self):
        return f"<Corpus {len(self._corpus_ids_)} documents>"

    def __str__(self):
        return f"<Corpus {len(self._corpus_ids_)} documents>"

    def __bool__(self):
        return bool(self._corpus_ids_)
