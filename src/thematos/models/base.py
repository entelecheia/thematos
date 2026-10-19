import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tomotopy as tp
from matplotlib.figure import Figure
from pydantic import Field, PrivateAttr

from thematos._base import Model
from thematos.datasets import Corpus
from thematos.plots import save_image_pages

from .config import LdaConfig, TrainConfig, TrainSummaryConfig, WordcloudConfig
from .prior import WordPrior
from .types import CoherenceMetrics, ModelSummary

logger = logging.getLogger(__name__)


def save_wordlist(words: list[str], path: Path) -> None:
    """One word per line, sorted (the 1.x ``save_wordlist`` format)."""
    path.write_text("".join(f"{w}\n" for w in sorted(words)), encoding="utf-8")


class TopicModel(Model):
    """A topic model whose artifacts live under ``output_dir``.

    File names start with ``model_id`` (``<MODEL_TYPE>_<name>_k(<k>)``), so models that differ
    in ``name`` or ``k`` can share one directory.
    """

    output_dir: Path
    name: str = "model"
    model_type: str = "BASE"
    seed: int | None = None
    num_workers: int = 1
    corpus: Corpus | None = None
    wordprior: WordPrior = Field(default_factory=WordPrior)
    model_args: LdaConfig = Field(default_factory=LdaConfig)
    train_args: TrainConfig = Field(default_factory=TrainConfig)
    train_summary_args: TrainSummaryConfig = Field(default_factory=TrainSummaryConfig)
    wc_args: WordcloudConfig = Field(default_factory=WordcloudConfig)

    coherence_metric_list: list[str] = Field(
        default_factory=lambda: ["u_mass", "c_uci", "c_npmi", "c_v"]
    )
    eval_coherence: bool = True
    set_wordprior: bool = False
    save_full: bool = True
    verbose: bool = False

    # internal attributes
    _model_: Any = PrivateAttr(None)
    _timestamp_: str | None = PrivateAttr(None)
    _coherence_metrics_: CoherenceMetrics | None = PrivateAttr(None)
    _model_summary_: ModelSummary | None = PrivateAttr(None)
    _ll_per_words_: list[tuple[int, float]] = PrivateAttr(default_factory=list)
    _doc_ids_: list[Any] | None = PrivateAttr(None)
    _doc_topic_dists_df_: pd.DataFrame | None = PrivateAttr(None)
    _topic_term_dists_df_: pd.DataFrame | None = PrivateAttr(None)

    @property
    def model_id(self) -> str:
        model_id = f"{self.model_type.upper()}_{self.name}_k({self.model_args.k})"
        # every artifact path joins model_id to output_dir, so keep it from escaping
        if not (self.output_dir / model_id).resolve().is_relative_to(self.output_dir.resolve()):
            raise ValueError(f"model_id {model_id!r} resolves outside output_dir")
        return model_id

    @property
    def model(self) -> Any:
        if self._model_ is None:
            raise ValueError("Model has not been trained yet.")
        return self._model_

    @property
    def coherence_metrics_dict(self) -> dict:
        return self._coherence_metrics_.model_dump() if self._coherence_metrics_ else {}

    @property
    def model_summary_dict(self) -> dict:
        return self._model_summary_.model_dump() if self._model_summary_ else {}

    @property
    def train_args_dict(self) -> dict:
        return self.train_args.model_dump()

    @property
    def model_args_dict(self) -> dict:
        return self.model_args.model_dump()

    @property
    def timestamp(self) -> str:
        if self._timestamp_ is None:
            raise ValueError("Model has not been trained yet.")
        return self._timestamp_

    @property
    def tp_corpus(self) -> tp.utils.Corpus:
        if self.corpus is None:
            raise ValueError("No corpus: pass one to train() or set `corpus`.")
        return self.corpus.corpus

    @property
    def doc_ids(self) -> list[Any]:
        if not self._doc_ids_:
            if self.corpus is None:
                raise ValueError("No corpus to read document ids from.")
            self._doc_ids_ = self.corpus.doc_ids
        return self._doc_ids_

    @property
    def ll_per_words(self) -> pd.DataFrame | None:
        if not self._ll_per_words_:
            logger.warning("No log-likelihood per word found.")
            return None
        return pd.DataFrame(self._ll_per_words_, columns=["iter", "ll_per_word"])

    @property
    def doc_topic_dists(self) -> np.ndarray:
        dist_ = np.stack([doc.get_topic_dist() for doc in self.model.docs])
        dist_ /= dist_.sum(axis=1, keepdims=True)
        return dist_

    @property
    def topic_term_dists(self) -> np.ndarray:
        return np.stack([self.model.get_topic_word_dist(k) for k in range(self.model.k)])

    @property
    def doc_num_words(self) -> np.ndarray:
        return np.array([len(doc.words) for doc in self.model.docs])

    @property
    def used_vocab(self) -> list[str]:
        return list(self.model.used_vocabs)

    @property
    def term_frequency(self) -> np.ndarray:
        return self.model.used_vocab_freq

    @property
    def num_docs(self) -> int:
        return len(self.model.docs)

    @property
    def num_words(self) -> int:
        return self.model.num_words

    @property
    def num_total_vocabs(self) -> int | None:
        return len(self.model.vocabs) if self.model.vocabs else None

    @property
    def num_used_vocab(self) -> int:
        return len(self.model.used_vocabs)

    @property
    def num_topics(self) -> int:
        """Number of topics in the model"""
        return self.model.k

    @property
    def topic_term_dists_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.topic_term_dists, columns=self.used_vocab)

    def get_doc_topic_dists_df(
        self, doc_topic_dists: np.ndarray, doc_ids: list[Any]
    ) -> pd.DataFrame:
        if len(doc_topic_dists) != len(doc_ids):
            raise ValueError(
                f"Number of inferred topics ({len(doc_topic_dists)}) does not match "
                f"with number of documents ({len(doc_ids)})"
            )
        columns = [f"topic{i}" for i in range(self.num_topics)]
        dists_df = pd.DataFrame(doc_topic_dists, columns=columns)
        doc_id_df = pd.DataFrame(doc_ids, columns=["id"])
        return pd.concat([doc_id_df, dists_df], axis=1)

    def _output_file(self, suffix: str) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.output_dir / f"{self.model_id}{suffix}"

    @property
    def model_file(self) -> Path:
        return self._output_file(".mdl")

    @property
    def ll_per_words_file(self) -> Path:
        return self._output_file("-ll_per_word.csv")

    @property
    def ll_per_words_fig_file(self) -> Path:
        return self._output_file("-ll_per_word.png")

    @property
    def doc_topic_dists_file(self) -> Path:
        return self._output_file("-doc_topic_dists.parquet")

    @property
    def topic_term_dists_file(self) -> Path:
        return self._output_file("-topic_term_dists.parquet")

    @property
    def used_vocab_file(self) -> Path:
        return self._output_file("-used_vocab.txt")

    @property
    def train_summary_file(self) -> Path:
        return self._output_file("-summary.txt")

    @property
    def topic_top_words_file(self) -> Path:
        return self._output_file("-topic_top_words.txt")

    @property
    def topic_top_words_dists_file(self) -> Path:
        return self._output_file("-topic_top_words_dists.csv")

    @property
    def ldavis_file(self) -> Path:
        return self._output_file("-ldavis.html")

    @property
    def model_summary_file(self) -> Path:
        """One JSON line per trained model, shared by every model in ``output_dir``."""
        return self.output_dir / "model-summary.jsonl"

    @property
    def topic_wordcloud_file_format(self) -> str:
        return str(
            self.output_dir / "wordclouds" / f"{self.model_id}-wordcloud_{{topic_id:03d}}.png"
        )

    def _set_wordprior(self) -> None:
        if not self.wordprior:
            logger.info("No word prior set.")
            return
        if self.verbose:
            logger.info("Set word prior with %s.", self.wordprior)
        for tno, words in self.wordprior.items():
            if self.verbose:
                logger.info("Set words %s to topic #%s as prior.", words, tno)
            for word in words:
                self.model.set_word_prior(
                    word,
                    [
                        (
                            self.wordprior.max_prior_weight
                            if i == int(tno)
                            else self.wordprior.min_prior_weight
                        )
                        for i in range(self.num_topics)
                    ],
                )

    def train(self, corpus: Corpus | None = None) -> "TopicModel":
        """Train on ``corpus`` (or the one already set) and evaluate coherence.

        Nothing is written; call :meth:`save`, :meth:`summarize`, and :meth:`plot` for artifacts.
        """
        if corpus is not None:
            self.corpus = corpus
            self._doc_ids_ = None
        self._model_ = self._new_model()
        self._coherence_metrics_ = None
        self._ll_per_words_ = []
        if self.set_wordprior:
            self._set_wordprior()
        self._timestamp_ = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._train(self._model_)
        if self.eval_coherence:
            self.eval_coherence_value()
        return self

    def _new_model(self) -> Any:
        raise NotImplementedError

    def _train(self, model: Any) -> None:
        raise NotImplementedError

    def eval_coherence_value(self) -> None:
        mdl = self.model
        coh_metrics = {}
        for metric in self.coherence_metric_list:
            coh = tp.coherence.Coherence(mdl, coherence=metric)
            average_coherence = coh.get_score()
            coh_metrics[metric] = average_coherence
            if self.verbose:
                coherence_per_topic = [coh.get_score(topic_id=k) for k in range(mdl.k)]
                logger.info("==== Coherence : %s ====", metric)
                logger.info("Average: %s", average_coherence)
                logger.info("Per Topic: %s", coherence_per_topic)
        self._coherence_metrics_ = CoherenceMetrics(**coh_metrics)

    def save(self) -> "TopicModel":
        """Write the model file, log-likelihood trace, distributions, and top words."""
        self.save_model()
        self.save_ll_per_words()
        self.save_dists_data()
        self.save_topic_top_words()
        return self

    def summarize(self) -> "TopicModel":
        """Write the tomotopy summary and append this model's metrics to the summary jsonl."""
        self.save_train_summary()
        self.save_model_summary()
        return self

    def plot(self) -> "TopicModel":
        """Write the log-likelihood plot, topic word clouds, and LDAvis (if installed)."""
        self.plot_ll_per_words()
        self.generate_wordclouds()
        self.save_ldavis()
        return self

    def save_model(self) -> None:
        self.model.save(str(self.model_file), full=self.save_full)
        logger.info("Model saved to %s", self.model_file)

    def save_ll_per_words(self) -> None:
        df_ll = self.ll_per_words
        if df_ll is not None:
            df_ll.to_csv(self.ll_per_words_file, index=False)

    def plot_ll_per_words(self) -> None:
        df_ll = self.ll_per_words
        if df_ll is None:
            return
        fig = Figure()
        ax = fig.subplots()
        ax.plot(df_ll["iter"], df_ll["ll_per_word"])
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Log-likelihood per word")
        ax.invert_yaxis()
        fig.savefig(self.ll_per_words_fig_file, dpi=300, transparent=False)
        logger.info("Log-likelihood per word plot saved to %s", self.ll_per_words_fig_file)

    def save_train_summary(self) -> None:
        with self.train_summary_file.open("w", encoding="utf-8") as f:
            self.model.summary(file=f, **self.train_summary_args.model_dump())
            if coh_values := self.coherence_metrics_dict:
                print("<Topic Coherence Scores>", file=f)
                for cm, cv in coh_values.items():
                    print(f"| {cm}: {cv}", file=f)

    def save_model_summary(self) -> None:
        self._model_summary_ = ModelSummary(
            timestamp=self.timestamp,
            model_id=self.model_id,
            model_type=self.model_type,
            num_docs=self.num_docs,
            num_words=self.num_words,
            num_total_vocabs=self.num_total_vocabs,
            num_used_vocabs=self.num_used_vocab,
            seed=self.seed,
            model_args=self.model_args_dict,
            train_args=self.train_args_dict,
            perplexity=self.model.perplexity,
            coherence=self.coherence_metrics_dict,
        )
        with self.model_summary_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(self.model_summary_dict, ensure_ascii=False) + "\n")
        logger.info("Model summary saved to %s", self.model_summary_file)

    def save_dists_data(self) -> None:
        doc_topic_dists_df = self.get_doc_topic_dists_df(self.doc_topic_dists, self.doc_ids)
        if self.verbose:
            logger.info("==== Document-Topic Distributions ====")
            logger.info(doc_topic_dists_df.tail())
        doc_topic_dists_df.to_parquet(self.doc_topic_dists_file)
        topic_term_dists_df = self.topic_term_dists_df
        if self.verbose:
            logger.info("==== Topic-Word Distributions ====")
            logger.info(topic_term_dists_df.tail())
        topic_term_dists_df.to_parquet(self.topic_term_dists_file)
        save_wordlist(self.used_vocab, self.used_vocab_file)

    def load(self) -> "TopicModel":
        """Load the model file and saved distributions written by :meth:`save` for ``model_id``."""
        self._load_model()
        self._load_ll_per_words()
        self._load_dists_data()
        return self

    def _load_model(self) -> None:
        raise NotImplementedError

    def _load_ll_per_words(self) -> None:
        ll_df = pd.read_csv(self.ll_per_words_file)
        self._ll_per_words_ = list(zip(ll_df["iter"], ll_df["ll_per_word"], strict=True))

    def _load_dists_data(self) -> None:
        self._doc_topic_dists_df_ = pd.read_parquet(self.doc_topic_dists_file)
        self._doc_ids_ = self._doc_topic_dists_df_["id"].tolist()
        self._topic_term_dists_df_ = pd.read_parquet(self.topic_term_dists_file)

    def save_ldavis(self) -> None:
        try:
            import pyLDAvis  # type: ignore[import-not-found]
        except ImportError:
            logger.warning("pyLDAvis is not installed. Please install it to save LDAvis.")
            return

        prepared_data = pyLDAvis.prepare(
            topic_term_dists=self.topic_term_dists,
            doc_topic_dists=self.doc_topic_dists,
            doc_lengths=self.doc_num_words,
            vocab=self.used_vocab,
            term_frequency=self.term_frequency,
            start_index=0,
            sort_topics=False,
        )
        pyLDAvis.save_html(prepared_data, str(self.ldavis_file))
        logger.info("LDAvis saved to %s", self.ldavis_file)

    def get_topic_words(self, topic_id: int, top_n: int = 10) -> dict[str, float]:
        return dict(self.model.get_topic_words(topic_id, top_n=top_n))

    def save_topic_top_words(self, top_n: int = 50) -> None:
        topic_top_words: list[str] = []
        topic_top_words_dists: list[tuple[int, str, float]] = []
        for topic_id in range(self.num_topics):
            topic_words = self.get_topic_words(topic_id, top_n=top_n)
            topic_top_words.extend(topic_words.keys())
            topic_top_words_dists.extend((topic_id, w, f) for w, f in topic_words.items())
        save_wordlist(list(set(topic_top_words)), self.topic_top_words_file)
        pd.DataFrame(topic_top_words_dists, columns=["topic_id", "word", "freq"]).to_csv(
            self.topic_top_words_dists_file, index=False
        )

    def generate_wordclouds(self) -> None:
        wc_args = self.wc_args
        images = [
            wc_args.wc.generate_from_frequencies(
                self.get_topic_words(topic_id, top_n=wc_args.top_n),
                output_file=self.topic_wordcloud_file_format.format(topic_id=topic_id),
                verbose=self.verbose,
            )
            for topic_id in range(self.num_topics)
        ]
        if wc_args.make_collage:
            titles = wc_args.titles or [f"Topic {i}" for i in range(self.num_topics)]
            logger.info("Making wordcloud collage with titles: %s", titles)
            save_image_pages(
                images,
                output_dir=self.output_dir / "wordcloud_collage",
                output_file_format=self.model_id + "_wordcloud_{page_num:02d}.png",
                num_images_per_page=wc_args.num_images_per_page,
                num_cols=wc_args.num_cols,
                num_rows=wc_args.num_rows,
                titles=titles,
                title_fontsize=wc_args.title_fontsize,
                title_color=wc_args.title_color,
                figsize=wc_args.figsize,
                width_multiple=wc_args.width_multiple,
                height_multiple=wc_args.height_multiple,
                dpi=wc_args.dpi,
            )

    @property
    def inferred_doc_topic_dists_file(self) -> Path:
        return (
            self.output_dir
            / "inferred_topics"
            / f"{self.model_id}-inferred_doc_topic_dists.parquet"
        )

    def infer(
        self,
        corpus: Corpus,
        output_file: str | Path | None = None,
        iterations: int = 100,
        tolerance: float = -1,
        num_workers: int = 0,
        together: bool = False,
    ) -> pd.DataFrame:
        """Infer topic distributions for ``corpus`` and save them (with log-likelihoods)."""
        inferred_corpus, ll = self.model.infer(
            corpus.corpus,
            iterations=iterations,
            tolerance=tolerance,
            workers=num_workers,
            together=together,
        )
        logger.info("Number of documents inferred: %d", len(inferred_corpus))
        output_file = Path(output_file or self.inferred_doc_topic_dists_file)

        doc_topic_dists = np.stack([doc.get_topic_dist() for doc in inferred_corpus])
        doc_topic_dists /= doc_topic_dists.sum(axis=1, keepdims=True)

        doc_topic_dists_df = self.get_doc_topic_dists_df(doc_topic_dists, corpus.doc_ids)
        ll_df = pd.DataFrame({"log_likelihood": ll})
        doc_topic_dists_df = pd.concat([doc_topic_dists_df, ll_df], axis=1)
        if self.verbose:
            logger.info("Inferred topics:\n%s", doc_topic_dists_df.head())
        output_file.parent.mkdir(parents=True, exist_ok=True)
        doc_topic_dists_df.to_parquet(output_file)
        logger.info("Inferred topics saved to %s", output_file)
        return doc_topic_dists_df
