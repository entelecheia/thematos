import sys
from datetime import datetime
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import tomotopy as tp
from hyfi import HyFI
from hyfi.composer import BaseModel
from hyfi.task import BatchTaskConfig

from thematos.datasets import Corpus

from .config import LdaConfig, TrainConfig
from .prior import WordPrior
from .types import CoherenceMetrics, ModelSummary

logger = HyFI.getLogger(__name__)


class TopicModel(BatchTaskConfig):
    _config_group_ = "model"
    _config_name_ = "topic"

    task_name: str = "topic"
    batch_name: str = "model"
    model_type: str = "BASE"
    wordprior: WordPrior = WordPrior()
    corpus: Corpus = Corpus()
    model_args: LdaConfig = LdaConfig()
    train_args: TrainConfig = TrainConfig()

    coherence_metric_list: List[str] = ["u_mass", "c_uci", "c_npmi", "c_v"]
    eval_coherence: bool = False
    set_wordprior: bool = False
    save: bool = True
    save_full: bool = True
    verbose: bool = False

    # internal attributes
    _model_: Optional[Any] = None
    _timestamp_: Optional[str] = None
    _coherence_metrics_: Optional[CoherenceMetrics] = None
    _model_summary_: Optional[ModelSummary] = None
    _ll_per_words_: List[Tuple[int, float]] = []

    @property
    def model_id(self) -> str:
        model_type = self.model_type.upper()
        margs = [model_type, self.batch_id, f"k({self.model_args.k})"]
        return "_".join(margs)

    @property
    def model(self):
        return self._model_

    @property
    def coherence_metrics(self) -> Optional[CoherenceMetrics]:
        return self._coherence_metrics_

    @property
    def timestamp(self) -> str:
        if self._timestamp_ is None:
            raise ValueError("Model has not been trained yet.")
        return self._timestamp_

    @property
    def tp_corpus(self) -> tp.utils.Corpus:
        return self.corpus.corpus

    @property
    def doc_ids(self) -> List[Any]:
        return self.corpus.doc_ids

    @property
    def model_file(self) -> str:
        f_ = f"{self.model_id}-{self.timestamp}.mdl"
        return str(self.model_dir / f_)

    @property
    def ll_per_words_file(self) -> str:
        f_ = f"{self.model_id}-ll_per_word-{self.timestamp}.csv"
        return str(self.output_dir / f_)

    @property
    def ll_per_words_fig_file(self) -> str:
        f_ = f"{self.model_id}-ll_per_word-{self.timestamp}.png"
        return str(self.output_dir / f_)

    @property
    def topic_dists_file(self) -> str:
        f_ = f"{self.model_id}-topic_dists-{self.timestamp}.parquet"
        return str(self.output_dir / f_)

    @property
    def train_summary_file(self) -> str:
        f_ = f"{self.model_id}-summary-{self.timestamp}.txt"
        return str(self.output_dir / f_)

    @property
    def batch_model_summary_file(self) -> str:
        f_ = f"{self.batch_name}-summary-{self.timestamp}.jsonl"
        return str(self.output_dir / f_)

    @property
    def ll_per_words(self) -> pd.DataFrame:
        if not self._ll_per_words_:
            raise ValueError("Model not trained yet.")
        return pd.DataFrame(self._ll_per_words_, columns=["iter", "ll_per_word"])

    @property
    def topic_dists(self) -> List[np.ndarray]:
        assert self.model, "Model not found"
        return [doc.get_topic_dist() for doc in self.model.docs]

    @property
    def num_topics(self) -> int:
        """Number of topics in the model

        It is the same as the number of columns in the document-topic distribution.
        """
        return len(self.topic_dists[0])

    @property
    def model_summary(self) -> ModelSummary:
        return self._model_summary_

    def _set_wordprior(self) -> None:
        if self.wordprior is None:
            logger.info("No word prior set.")
            return
        for tno, words in self.wordprior.items():
            if self.verbose:
                logger.info("Set words %s to topic #%s as prior.", words, tno)
            for word in words:
                self.model.set_word_prior(
                    word,
                    [
                        self.wordprior.max_prior_weight
                        if i == int(tno)
                        else self.wordprior.min_prior_weight
                        for i in range(self.k)
                    ],
                )

    def train(self) -> None:
        if self.set_wordprior:
            self._set_wordprior()

        self._timestamp_ = datetime.now().strftime("%Y%m%d_%H%M%S")
        # train model
        self._train(self.model)
        # save model
        self.save_model()
        self.save_ll_per_words()
        self.plot_ll_per_words()
        if self.eval_coherence:
            self.eval_coherence_value()
        self.save_document_topic_dists()
        # self.save_model_summary()
        self.save_config()

    def _train(self, model: Any) -> None:
        raise NotImplementedError

    def eval_coherence_value(
        self,
    ):
        assert self.model, "Model not found"
        mdl = self.model
        coh_metrics = {}
        for metric in self.coherence_metric_list:
            coh = tp.coherence.Coherence(mdl, coherence=metric)
            average_coherence = coh.get_score()
            coh_metrics[metric] = average_coherence
            coherence_per_topic = [coh.get_score(topic_id=k) for k in range(mdl.k)]
            if self.verbose:
                logger.info("==== Coherence : %s ====", metric)
                logger.info("Average: %s", average_coherence)
                logger.info("Per Topic: %s", coherence_per_topic)
        self._coherence_metrics_ = CoherenceMetrics(**coh_metrics)

    def save_model(self) -> None:
        self.model.save(self.model_file, full=self.save_full)
        logger.info("Model saved to %s", self.model_file)

    def save_ll_per_words(self) -> None:
        HyFI.save_dataframes(
            self.ll_per_words, self.ll_per_words_file, verbose=self.verbose
        )

    def plot_ll_per_words(self) -> None:
        df_ll = self.ll_per_words
        ax = df_ll.plot(x="iter", y="ll_per_word", kind="line")
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Log-likelihood per word")
        ax.invert_yaxis()
        ax.get_figure().savefig(self.ll_per_words_fig_file, dpi=300, transparent=False)
        logger.info(
            "Log-likelihood per word plot saved to %s", self.ll_per_words_fig_file
        )

    def save_train_summary(self) -> None:
        coh_values = self.coherence_metrics.model_dump()
        original_stdout = sys.stdout
        with open(self.train_summary_file, "w") as f:
            sys.stdout = f  # Change the standard output to the file.
            self.model.summary()
            if coh_values:
                print("<Topic Coherence Scores>")
                for cm, cv in coh_values.items():
                    print(f"| {cm}: {cv}")
            sys.stdout = original_stdout  # Reset the standard output.

    def save_model_summary(self) -> None:
        self._model_summary_ = ModelSummary(
            timestamp=self.timestamp,
            model_id=self.model_id,
            model_type=self.model_type,
            num_docs=len(self.model.docs),
            num_words=self.model.num_words,
            total_vocabs=len(self.model.vocabs) if self.model.vocabs else None,
            used_vocabs=len(self.model.used_vocabs),
            train_config=self.train_args.model_dump(),
            ll_per_word=self.ll_per_words,
            perplexity=self.model.perplexity,
            coherence_metrics=self.coherence_metrics.model_dump()
            if self.coherence_metrics
            else None,
        )
        if not self.model_summary:
            logger.warning("Model summary is not available.")
        HyFI.append_to_jsonl(
            self.model_summary.model_dump(),
            self.model_summary_file,
        )
        logger.info("Model summary saved to %s", self.model_summary_file)

    def save_document_topic_dists(self):
        corpus_ids = self.doc_ids
        topic_dists = self.topic_dists

        logger.info("Total inferred: %s, from: %s", len(topic_dists), len(corpus_ids))
        if len(topic_dists) == len(corpus_ids):
            self._save_document_topic_dists(topic_dists, corpus_ids)
        else:
            raise ValueError(
                f"Number of inferred topics ({len(topic_dists)}) does not match with number of documents ({len(corpus_ids)})"
            )

    def _save_document_topic_dists(
        self,
        topic_dists: List[np.ndarray],
        corpus_ids: List[Any],
    ):
        idx = range(self.num_topics)
        topic_dists_df = pd.DataFrame(topic_dists, columns=[f"topic{i}" for i in idx])
        id_df = pd.DataFrame(corpus_ids, columns=["id"])
        topic_dists_df = pd.concat([id_df, topic_dists_df], axis=1)
        if self.verbose:
            print(topic_dists_df.tail())

        HyFI.save_dataframes(
            topic_dists_df, self.topic_dists_file, verbose=self.verbose
        )
