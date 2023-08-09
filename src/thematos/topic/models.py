import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import tomotopy as tp
from hyfi import HyFI
from hyfi.composer import BaseModel
from hyfi.run import RunConfig
from hyfi.task import BatchTaskConfig
from hyfi.utils.contexts import elapsed_timer
from tqdm.auto import tqdm

from .corpus import Corpus
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


class LdaModel(TopicModel):
    model_type: str = "LDA"

    k: int = None
    t: int = 1
    tw: str = IDF
    gamma: float = 2
    alpha: float = 0.1
    eta: float = 0.01
    phi: float = 0.1
    min_cf: int = 5
    rm_top: int = 0
    min_df: int = 0

    # internal attributes
    _model_: Optional[tp.LDAModel] = None

    @property
    def model_id(self) -> str:
        model_type = self.model_type.upper()
        margs = [self.model_name, model_type]
        if self.k:
            margs.append(f"k({self.k})")
        return "_".join(margs)

    @property
    def model(self) -> tp.LDAModel:
        if self._model_ is None:
            cfg = self.train_config
            self._model_ = tp.LDAModel(
                tw=cfg.tw,
                k=cfg.k,
                min_cf=cfg.min_cf,
                rm_top=cfg.rm_top,
                alpha=cfg.alpha,
                eta=cfg.eta,
                corpus=self.corpus,
                seed=self.seed,
            )
        return self._model_

    def set_wordprior(self) -> None:
        if self.wordprior is None:
            logger.info("No word prior set.")
            return
        for tno, words in self.wordprior.items():
            if self.verbose:
                logger.info("Set words %s to topic #%s as prior.", words, tno)
            for word in words:
                self.model.set_word_prior(
                    word, [1.0 if i == int(tno) else 0.1 for i in range(self.k)]
                )

    def train(
        self,
    ) -> None:
        self.set_wordprior()

        self._train_timestamp_ = datetime.now().strftime("%Y%m%d_%H%M%S")
        mdl = self.model
        mdl.burn_in = self.burn_in
        mdl.train(0)
        logger.info("Number of docs: %s", len(mdl.docs))
        logger.info("Vocab size: %s", mdl.num_vocabs)
        logger.info("Number of words: %s", mdl.num_words)
        logger.info("Removed top words: %s", mdl.removed_top_words)
        logger.info(
            "Training model by iterating over the corpus %s times, %s iterations at a time",
            self.iterations,
            self.interval,
        )

        ll_per_words = []
        for i in range(0, self.iterations, self.interval):
            mdl.train(self.interval)
            logger.info("Iteration: %s\tLog-likelihood: %s", i, mdl.ll_per_word)
            ll_per_words.append((i, mdl.ll_per_word))
        self._ll_per_words_ = ll_per_words
        if self.verbose:
            mdl.summary()
        self._model_ = mdl
        self.save_model()
        if self.eval_coherence:
            self._coh_values_ = self.evaluate_coherence()
        self.save_document_topic_dists()

    def eval_coherence_value(
        self,
    ):
        assert self.model, "Model not found"
        mdl = self.model
        coh_values = {}
        for metric in self.coherence_metrics:
            coh = tp.coherence.Coherence(mdl, coherence=metric)
            average_coherence = coh.get_score()
            coh_values[metric] = average_coherence
            coherence_per_topic = [coh.get_score(topic_id=k) for k in range(mdl.k)]
            if self.verbose:
                logger.info("==== Coherence : %s ====", metric)
                logger.info("Average: %s", average_coherence)
                logger.info("Per Topic: %s", coherence_per_topic)
        self._coh_values_ = CoherenceMetrics(**coh_values)

    def save_model(self) -> None:
        model_path = self.model_dir / self.model_file
        self.model.save(str(model_path), full=self.save_full)

    def save_ll_per_words(self) -> None:
        out_file = self.output_dir / self.ll_per_words_file
        HyFI.save_dataframes(self.ll_per_words, out_file, verbose=self.verbose)

    def plot_ll_per_words(self) -> None:
        df_ll = self.ll_per_words
        ax = df_ll.plot(x="iter", y="ll_per_word", kind="line")
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Log-likelihood per word")
        ax.invert_yaxis()
        out_file = self.output_dir / self.ll_per_words_fig_file
        ax.get_figure().savefig(out_file, dpi=300, transparent=False)

    def save_train_summary(self) -> None:
        coh_values = self.cohrence_values.model_dump()
        mdl = self.model
        original_stdout = sys.stdout
        out_file = str(self.output_dir / "logs" / out_file)
        with open(out_file, "w") as f:
            sys.stdout = f  # Change the standard output to the file.
            mdl.summary()
            if coh_values:
                print("<Topic Coherence Scores>")
                for cm, cv in coh_values.items():
                    print(f"| {cm}: {cv}")
            sys.stdout = original_stdout  # Reset the standard output.

    @property
    def topic_dists(self):
        assert self.model, "Model not found"
        return [doc.get_topic_dist() for doc in self.model.docs]

    @property
    def num_topics(self):
        """Number of topics in the model

        It is the same as the number of columns in the document-topic distribution.
        """
        return len(self.topic_dists[0])

    def save_document_topic_dists(self):
        corpus_ids = self.corpus_ids
        topic_dists = self.topic_dists

        logger.info(
            "Total inferred: %s, from: %s", len(topic_dists), len(corpus_ids.index)
        )
        if len(topic_dists) == len(corpus_ids.index):
            self._save_document_topic_dists(topic_dists, corpus_ids)
        else:
            raise ValueError(
                f"Number of inferred topics ({len(topic_dists)}) does not match with number of documents ({len(corpus_ids.index)})"
            )

    def _save_document_topic_dists(self, topic_dists, corpus_ids):
        idx = range(self.num_topics)
        topic_dists_df = pd.DataFrame(topic_dists, columns=[f"topic{i}" for i in idx])
        topic_dists_df = pd.concat([corpus_ids, topic_dists_df], axis=1)
        if self.verbose:
            print(topic_dists_df.tail())

        output_path = self.output_dir / self.topic_dists_file
        HyFI.save_data(topic_dists_df, output_path, verbose=self.verbose)
