import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import tomotopy as tp
from hyfi import HyFI
from hyfi.task import BatchTaskConfig

from thematos.datasets import Corpus

from .config import LdaConfig, TrainConfig
from .prior import WordPrior
from .types import CoherenceMetrics, ModelSummary

logger = HyFI.getLogger(__name__)


class TopicModel(BatchTaskConfig):
    _config_group_ = "/model"
    _config_name_ = "topic"

    task_name: str = "topic"
    batch_name: str = "model"
    model_type: str = "BASE"
    wordprior: WordPrior = WordPrior()
    corpus: Corpus = Corpus()
    model_args: LdaConfig = LdaConfig()
    train_args: TrainConfig = TrainConfig()

    coherence_metric_list: List[str] = ["u_mass", "c_uci", "c_npmi", "c_v"]
    eval_coherence: bool = True
    set_wordprior: bool = False
    autosave: bool = True
    save_full: bool = True
    verbose: bool = False

    # internal attributes
    _model_: Optional[Any] = None
    _timestamp_: Optional[str] = None
    _coherence_metrics_: Optional[CoherenceMetrics] = None
    _model_summary_: Optional[ModelSummary] = None
    _ll_per_words_: List[Tuple[int, float]] = []
    _doc_ids_: List[Any] = None

    @property
    def model_id(self) -> str:
        model_type = self.model_type.upper()
        margs = [model_type, self.batch_id, f"k({self.model_args.k})"]
        return "_".join(margs)

    @property
    def model(self):
        return self._model_

    @property
    def coherence_metrics_dict(self) -> Dict:
        return self._coherence_metrics_.model_dump() if self._coherence_metrics_ else {}

    @property
    def model_summary_dict(self) -> Dict:
        return self._model_summary_.model_dump() if self._model_summary_ else {}

    @property
    def train_args_dict(self) -> Dict:
        return (
            self.train_args.model_dump(exclude=self.train_args._exclude_keys_)
            if self.train_args
            else {}
        )

    @property
    def model_args_dict(self) -> Dict:
        return (
            self.model_args.model_dump(exclude=self.model_args._exclude_keys_)
            if self.model_args
            else {}
        )

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
        if not self._doc_ids_:
            self._doc_ids_ = self.corpus.doc_ids
        return self._doc_ids_

    @property
    def ll_per_words(self) -> Optional[pd.DataFrame]:
        if not self._ll_per_words_:
            logger.warning("No log-likelihood per word found.")
            return None
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
    def model_file(self) -> str:
        f_ = f"{self.model_id}.mdl"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        return str(self.model_dir / f_)

    @property
    def ll_per_words_file(self) -> str:
        f_ = f"{self.model_id}-ll_per_word.csv"
        return str(self.output_dir / f_)

    @property
    def ll_per_words_fig_file(self) -> str:
        f_ = f"{self.model_id}-ll_per_word.png"
        return str(self.output_dir / f_)

    @property
    def topic_dists_file(self) -> str:
        f_ = f"{self.model_id}-topic_dists.parquet"
        return str(self.output_dir / f_)

    @property
    def train_summary_file(self) -> str:
        f_ = f"{self.model_id}-summary.txt"
        return str(self.output_dir / f_)

    @property
    def batch_model_summary_file(self) -> str:
        f_ = f"{self.batch_name}-summary.jsonl"
        return str(self.output_dir / f_)

    @property
    def ldavis_file(self) -> str:
        f_ = f"{self.model_id}-ldavis.html"
        return str(self.output_dir / f_)

    def update_model_args(self, **kwargs) -> None:
        self.model_args = self.model_args.model_copy(update=kwargs)

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
        if self.eval_coherence:
            self.eval_coherence_value()
        if self.autosave:
            self.save()

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

    def save(self) -> None:
        self.save_model()
        self.save_ll_per_words()
        self.plot_ll_per_words()
        self.save_document_topic_dists()
        self.save_ldavis()
        self.save_model_summary()
        self.save_config()

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
        coh_values = self.coherence_metrics_dict
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
            seed=self.seed,
            model_args=self.model_args_dict,
            train_args=self.train_args_dict,
            perplexity=self.model.perplexity,
            coherence=self.coherence_metrics_dict,
        )
        if not self.model_summary_dict:
            logger.warning("Model summary is not available.")
        HyFI.append_to_jsonl(
            self.model_summary_dict,
            self.batch_model_summary_file,
        )
        logger.info("Model summary saved to %s", self.batch_model_summary_file)

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

    def load(
        self,
        batch_name: Optional[str] = None,
        batch_num: Optional[int] = None,
        filepath: Optional[Union[str, Path]] = None,
        **config_kwargs,
    ):
        super().load_config(
            batch_name=batch_name,
            batch_num=batch_num,
            filepath=filepath,
            **config_kwargs,
        )
        self._load_model()
        self._load_ll_per_words()
        self._load_document_topic_dists()

    def _load_ll_per_words(self):
        ll_df = HyFI.load_dataframes(self.ll_per_words_file, verbose=self.verbose)
        self._ll_per_words_ = [(ll.iter, ll.ll_per_word) for ll in ll_df.itertuples()]

    def _load_document_topic_dists(self):
        topic_dists_df = HyFI.load_dataframes(
            self.topic_dists_file, verbose=self.verbose
        )
        self._topic_dists_ = topic_dists_df.iloc[:, 1:].values.tolist()
        self._doc_ids_ = topic_dists_df["id"].values.tolist()

    def save_ldavis(self):
        import pyLDAvis

        assert self.model, "Model not found"
        mdl = self.model

        topic_term_dists = np.stack([mdl.get_topic_word_dist(k) for k in range(mdl.k)])

        doc_topic_dists = np.stack(
            [
                doc.get_topic_dist()
                for doc in mdl.docs
                if np.sum(doc.get_topic_dist()) == 1
            ]
        )
        doc_lengths = np.array(
            [len(doc.words) for doc in mdl.docs if np.sum(doc.get_topic_dist()) == 1]
        )
        vocab = list(mdl.used_vocabs)
        term_frequency = mdl.used_vocab_freq

        # doc_topic_dists = np.stack([doc.get_topic_dist() for doc in mdl.docs])
        # doc_topic_dists /= doc_topic_dists.sum(axis=1, keepdims=True)
        # doc_lengths = np.array([len(doc.words) for doc in mdl.docs])
        # vocab = list(mdl.used_vocabs)
        # term_frequency = mdl.used_vocab_freq

        prepared_data = pyLDAvis.prepare(
            topic_term_dists=topic_term_dists,
            doc_topic_dists=doc_topic_dists,
            doc_lengths=doc_lengths,
            vocab=vocab,
            term_frequency=term_frequency,
            start_index=0,
            sort_topics=False,
        )
        pyLDAvis.save_html(prepared_data, self.ldavis_file)
        logger.info("LDAvis saved to %s", self.ldavis_file)
