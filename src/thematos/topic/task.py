import itertools
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from hyfi import HyFI
from hyfi.task import BatchTaskConfig
from hyfi.utils.contexts import elapsed_timer
from tqdm.auto import tqdm

from .corpus import Corpus
from .models import LdaModel
from .prior import WordPrior
from .types import IDF, ONE, PMI, ModelSummary

logger = HyFI.getLogger(__name__)


class TopicTask(BatchTaskConfig):
    _config_group_ = "/topic"
    _config_name_ = "__init__"

    wordprior: WordPrior = WordPrior()
    corpus: Corpus = Corpus()

    model_name: str = "TopicModel"

    model: LdaModel = LdaModel()

    num_workers: int = 0
    ngram: int = None
    files: dict = None
    verbose: bool = False

    _summaries_: List[ModelSummary] = []
    active_model_id: Optional[str] = None

    model = None
    models = {}
    labels = []

    @property
    def summary_file(self) -> Path:
        summary_file = f"{self.model_name}_summaries.csv"
        return self.output_dir / summary_file

    @property
    def summaries(self) -> List[ModelSummary]:
        if self._summaries_:
            return self._summaries_
        summaries = []
        if HyFI.is_file(self.summary_file):
            data = HyFI.load_dataframe(self.summary_file, index_col=0)
            summaries.extend(ModelSummary(*row[1:]) for row in data.itertuples())
        self._summaries_ = summaries
        return summaries

    # def tune_params(
    #     self,
    #     model_type="LDA",
    #     topics=[20],
    #     alphas=[0.1],
    #     etas=[0.01],
    #     sample_ratios=[0.1],
    #     tws=[IDF],
    #     min_cf=5,
    #     rm_top=0,
    #     min_df=0,
    #     burn_in=0,
    #     interval=10,
    #     iterations=100,
    #     seed=None,
    #     eval_coherence=True,
    #     save=False,
    #     save_full=False,
    # ):
    #     """
    #     # Topics range
    #     topics = range(min_topics, max_topics, step_size)
    #     # Alpha parameter
    #     alphas = np.arange(0.01, 1, 0.3)
    #     # Beta parameter
    #     etas = np.arange(0.01, 1, 0.3)
    #     # Validation sets
    #     sample_ratios = [0.1, 0.5]
    #     """
    #     total_iters = (
    #         len(etas) * len(alphas) * len(topics) * len(tws) * len(sample_ratios)
    #     )
    #     exec_dt = datetime.now().strftime("%Y%m%d_%H%M%S")
    #     df_lls = None
    #     ys = []
    #     for i, (sr, k, a, e, tw) in tqdm(
    #         enumerate(
    #             itertools.product(sample_ratios, topics, alphas, etas, tws),
    #         ),
    #         total=total_iters,
    #     ):
    #         # train a model
    #         print(
    #             "sample_ratio: {}, k:{}, alpha:{}, eta:{}, tw:{}".format(
    #                 sr, k, a, e, str(tw)
    #             )
    #         )
    #         # self.load_corpus(sample_ratio=sr)
    #         df_ll, m_sum = self.train_model(
    #             model_type=model_type,
    #             sample_ratio=sr,
    #             k=k,
    #             tw=tw,
    #             alpha=a,
    #             eta=e,
    #             min_cf=min_cf,
    #             rm_top=rm_top,
    #             min_df=min_df,
    #             burn_in=burn_in,
    #             interval=interval,
    #             iterations=iterations,
    #             seed=seed,
    #             eval_coherence=eval_coherence,
    #             save=save,
    #             save_full=save_full,
    #         )
    #         margs = []
    #         if len(topics) > 1:
    #             margs.append("k={}".format(k))
    #         if len(alphas) > 1:
    #             margs.append("a={}".format(a))
    #         if len(etas) > 1:
    #             margs.append("e={}".format(e))
    #         if len(tws) > 1:
    #             margs.append("tw={}".format(tw))
    #         if len(sample_ratios) > 1:
    #             margs.append("sr={}".format(sr))
    #         y = ",".join(margs) if len(margs) > 0 else "ll_{}".format(i)
    #         ys.append(y)

    #         df_ll.rename(columns={"ll_per_word": y}, inplace=True)
    #         if df_lls is not None:
    #             df_lls = df_lls.merge(df_ll, on="iter")
    #         else:
    #             df_lls = df_ll

    #     out_file = "{}-{}-ll_per_word-{}.csv".format(
    #         self.model_name, model_type, exec_dt
    #     )
    #     out_file = str(self.model_dir / "output/tune" / out_file)
    #     df_lls.to_csv(out_file)
    #     ax = df_lls.plot(x="iter", y=ys, kind="line")
    #     ax.set_xlabel("Iterations")
    #     ax.set_ylabel("Log-likelihood per word")
    #     ax.invert_yaxis()
    #     out_file = "{}-{}-ll_per_word-{}.png".format(
    #         self.model_name, model_type, exec_dt
    #     )
    #     out_file = str(self.model_dir / "figures/tune" / out_file)
    #     savefig(out_file, transparent=False, dpi=300)

    # def load_model(self, model_id=None, model_file=None, reload_model=False, **kwargs):
    #     if model_id:
    #         self.active_model_id = model_id
    #     if self.active_model_id in self.models and not reload_model:
    #         print("The model is already loaded.")
    #         return True

    #     if model_file is None:
    #         model_file = "{}-{}.mdl".format(self.model_name, self.active_model_id)
    #     model_path = self.model_dir / model_file
    #     print("Loading a model from {}".format(model_path))
    #     if model_path.is_file():
    #         if not self.active_model_id:
    #             self.active_model_id = model_path.stem.split("-")[-1]
    #         model_type = self.active_model_id.split(".")[0]
    #         model_path = str(model_path)
    #         with elapsed_timer() as elapsed:
    #             if model_type == "LDA":
    #                 mdl = tp.LDAModel.load(model_path)
    #             elif model_type == "HPA":
    #                 mdl = tp.HPAModel.load(model_path)
    #             elif model_type == "HDP":
    #                 mdl = tp.HDPModel.load(model_path)
    #             else:
    #                 print("{} is not supported".format(model_type))
    #                 return False
    #             self.models[self.active_model_id] = mdl
    #             self.model = mdl
    #             print("Elapsed time is %.2f seconds" % elapsed())
    #     else:
    #         self.model = None
    #         print("Model file not found")

    # def save_labels(self, names=None, **kwargs):
    #     if names is None:
    #         print("No names are given")
    #         return

    #     if not self.labels:
    #         self.label_topics()
    #     for k in names:
    #         self.labels[int(k)]["topic_name"] = names[k]
    #         if self.verbose:
    #             print(f"{k}: {names[k]}")
    #     label_file = "{}-labels.csv".format(self.active_model_id)
    #     label_file = self.output_dir / label_file
    #     df = pd.DataFrame(self.labels)
    #     HyFI.save_data(df, label_file, index=False, verbose=self.verbose)

    # def label_topics(
    #     self,
    #     rebuild=False,
    #     use_pmiextractor=False,
    #     min_cf=10,
    #     min_df=5,
    #     max_len=5,
    #     max_cand=100,
    #     smoothing=1e-2,
    #     mu=0.25,
    #     window_size=100,
    #     top_n=10,
    #     **kwargs,
    # ):
    #     label_file = "{}-labels.csv".format(self.active_model_id)
    #     label_file = self.output_dir / label_file
    #     if label_file.is_file() and not rebuild:
    #         print("loading labels from {}".format(label_file))
    #         df = HyFI.load_data(label_file)
    #         self.labels = df.to_dict("records")
    #     else:
    #         assert self.model, "Model not found"
    #         mdl = self.model
    #         if use_pmiextractor:
    #             # extract candidates for auto topic labeling
    #             print("extract candidates for auto topic labeling")
    #             extractor = tp.label.PMIExtractor(
    #                 min_cf=min_cf, min_df=min_df, max_len=max_len, max_cand=max_cand
    #             )
    #             with elapsed_timer() as elapsed:
    #                 cands = extractor.extract(mdl)
    #                 print("Elapsed time is %.2f seconds" % elapsed())
    #                 labeler = tp.label.FoRelevance(
    #                     mdl,
    #                     cands,
    #                     min_df=min_df,
    #                     smoothing=smoothing,
    #                     mu=mu,
    #                     window_size=window_size,
    #                 )
    #                 print("Elapsed time is %.2f seconds" % elapsed())
    #             self.labeler = labeler

    #         labels = []
    #         for k in range(mdl.k):
    #             print("== Topic #{} ==".format(k))
    #             name = f"Topic #{k}"
    #             if use_pmiextractor:
    #                 lbls = ",".join(
    #                     label
    #                     for label, score in labeler.get_topic_labels(k, top_n=top_n)
    #                 )
    #                 print(
    #                     "Labels:",
    #                     ", ".join(
    #                         label
    #                         for label, score in labeler.get_topic_labels(k, top_n=top_n)
    #                     ),
    #                 )
    #             wrds = ",".join(
    #                 word for word, prob in mdl.get_topic_words(k, top_n=top_n)
    #             )
    #             if use_pmiextractor:
    #                 label = {
    #                     "topic_no": k,
    #                     "topic_num": f"topic{k}",
    #                     "topic_name": name,
    #                     "topic_labels": lbls,
    #                     "topic_words": wrds,
    #                 }
    #             else:
    #                 label = {
    #                     "topic_no": k,
    #                     "topic_num": f"topic{k}",
    #                     "topic_name": name,
    #                     "topic_words": wrds,
    #                 }
    #             labels.append(label)
    #             for word, prob in mdl.get_topic_words(k, top_n=top_n):
    #                 print(word, prob, sep="\t")
    #             print()

    #         self.labels = labels
    #         df = pd.DataFrame(self.labels)
    #         HyFI.save_data(df, label_file, index=False, verbose=self.verbose)

    # def visualize(self, **kwargs):
    #     import pyLDAvis

    #     assert self.model, "Model not found"
    #     mdl = self.model
    #     topic_term_dists = np.stack([mdl.get_topic_word_dist(k) for k in range(mdl.k)])
    #     doc_topic_dists = np.stack(
    #         [
    #             doc.get_topic_dist()
    #             for doc in mdl.docs
    #             if np.sum(doc.get_topic_dist()) == 1
    #         ]
    #     )
    #     doc_lengths = np.array(
    #         [len(doc.words) for doc in mdl.docs if np.sum(doc.get_topic_dist()) == 1]
    #     )
    #     vocab = list(mdl.used_vocabs)
    #     term_frequency = mdl.used_vocab_freq

    #     prepared_data = pyLDAvis.prepare(
    #         topic_term_dists, doc_topic_dists, doc_lengths, vocab, term_frequency
    #     )
    #     out_file = "{}-{}-ldavis.html".format(self.model_name, self.active_model_id)
    #     out_file = str(self.output_dir / "output" / out_file)
    #     pyLDAvis.save_html(prepared_data, out_file)

    # def get_topic_words(self, top_n=10):
    #     """Wrapper function to extract topics from trained tomotopy HDP model

    #     ** Inputs **
    #     top_n: int -> top n words in topic based on frequencies

    #     ** Returns **
    #     topics: dict -> per topic, an arrays with top words and associated frequencies
    #     """
    #     assert self.model, "Model not found"
    #     mdl = self.model

    #     # Get most important topics by # of times they were assigned (i.e. counts)
    #     sorted_topics = [
    #         k
    #         for k, v in sorted(
    #             enumerate(mdl.get_count_by_topics()), key=lambda x: x[1], reverse=True
    #         )
    #     ]

    #     topics = dict()
    #     # For topics found, extract only those that are still assigned
    #     for k in sorted_topics:
    #         if type(mdl) in ["tomotopy.HPAModel", "tomotopy.HDPModel"]:
    #             if not mdl.is_live_topic(k):
    #                 continue  # remove un-assigned topics at the end (i.e. not alive)
    #         topic_wp = []
    #         for word, prob in mdl.get_topic_words(k, top_n=top_n):
    #             topic_wp.append((word, prob))

    #         topics[k] = topic_wp  # store topic word/frequency array

    #     return topics

    # def topic_wordclouds(
    #     self,
    #     title_fontsize=20,
    #     title_color="green",
    #     top_n=100,
    #     ncols=5,
    #     nrows=1,
    #     dpi=300,
    #     figsize=(10, 10),
    #     save=True,
    #     mask_dir=None,
    #     wordclouds=None,
    #     save_each=False,
    #     save_masked=False,
    #     fontpath=None,
    #     colormap="PuBu",
    #     **kwargs,
    # ):
    #     """Wrapper function that generates wordclouds for ALL topics of a tomotopy model
    #     ** Inputs **
    #     topic_dic: dict -> per topic, an arrays with top words and associated frequencies
    #     save: bool -> If the user would like to save the images

    #     ** Returns **
    #     wordclouds as plots
    #     """
    #     assert self.model, "Model not found"
    #     num_topics = self.model.k

    #     if figsize is not None and isinstance(figsize, str):
    #         figsize = eval(figsize)
    #     if mask_dir is None:
    #         mask_dir = str(self.output_dir / "figures/masks")
    #     fig_output_dir = str(self.output_dir / "figures/wc")
    #     fig_filename_format = "{}-{}-wc_topic".format(
    #         self.model_name, self.active_model_id
    #     )
    #     if wordclouds is None:
    #         wordclouds_args = {}
    #     else:
    #         wordclouds_args = HyFI.to_dict(wordclouds)
    #     for k in range(num_topics):
    #         topic_freq = dict(self.model.get_topic_words(k, top_n=top_n))
    #         if k in wordclouds_args:
    #             wc_args = wordclouds_args[k]
    #         else:
    #             wc_args = {}
    #         title = wc_args.get("title", None)
    #         if title is None:
    #             if self.labels:
    #                 topic_name = self.labels[k]["name"]
    #                 if topic_name.startswith("Topic #"):
    #                     topic_name = None
    #             else:
    #                 topic_name = None
    #             if topic_name:
    #                 title = f"Topic #{k} - {topic_name}"
    #             else:
    #                 title = f"Topic #{k}"
    #             wc_args["title"] = title
    #         wc_args["word_freq"] = topic_freq
    #         wordclouds_args[k] = wc_args

    #     generate_wordclouds(
    #         wordclouds_args,
    #         fig_output_dir,
    #         fig_filename_format,
    #         title_fontsize=title_fontsize,
    #         title_color=title_color,
    #         ncols=ncols,
    #         nrows=nrows,
    #         dpi=dpi,
    #         figsize=figsize,
    #         save=save,
    #         mask_dir=mask_dir,
    #         save_each=save_each,
    #         save_masked=save_masked,
    #         fontpath=fontpath,
    #         colormap=colormap,
    #         verbose=self.verbose,
    #         **kwargs,
    #     )


# class SimpleTokenizer:
#     """Class to tokenize texts for a corpus"""

#     def __init__(
#         self,
#         stopwords=[],
#         min_word_len=2,
#         min_num_words=5,
#         verbose=False,
#         ngrams=[],
#         ngram_delimiter="_",
#         **kwargs,
#     ):
#         self.stopwords = stopwords if stopwords else []
#         self.min_word_len = min_word_len
#         self.min_num_words = min_num_words
#         self.ngram_delimiter = ngram_delimiter
#         if ngrams:
#             self.ngrams = {ngram_delimiter.join(ngram): ngram for ngram in ngrams}
#         else:
#             self.ngrams = {}
#         self.verbose = verbose
#         self.verbose = verbose
#         if verbose:
#             print(f"{self.__class__.__name__} initialized with:")
#             print(f"\tstopwords: {len(self.stopwords)}")
#             print(f"\tmin_word_len: {self.min_word_len}")
#             print(f"\tmin_num_words: {self.min_num_words}")
#             print(f"\tngrams: {len(self.ngrams)}")
#             print(f"\tngram_delimiter: {self.ngram_delimiter}")

#     def tokenize(self, text):
#         if text is None:
#             return None
#         if len(self.ngrams) > 0:
#             words = text.split()
#             for repl, ngram in self.ngrams.items():
#                 words = self.replace_seq(words, ngram, repl)
#         else:
#             words = text.split()
#         words = [
#             w for w in words if w not in self.stopwords and len(w) >= self.min_word_len
#         ]
#         if len(set(words)) > self.min_num_words:
#             return words
#         else:
#             return None

#     @staticmethod
#     def replace_seq(sequence, subseq, repl):
#         if len(sequence) < len(subseq):
#             return sequence
#         return eval(str(list(sequence)).replace(str(list(subseq))[1:-1], f"'{repl}'"))

#     def infer_topics(
#         self,
#         iterations=100,
#         min_num_words=5,
#         min_word_len=2,
#         num_workers=0,
#         use_batcher=True,
#         minibatch_size=None,
#         **kwargs,
#     ):
#         self._load_stopwords()
#         assert self.stopwords, "Load stopwords first"
#         assert self.model, "Model not found"
#         print("Infer document out of the model")

#         os.makedirs(os.path.abspath(output_dir), exist_ok=True)
#         num_workers = num_workers if num_workers else 1
#         text_key = self._dataset_._text_key
#         id_keys = self._dataset_._id_keys

#         df_ngram = HyFI.load_data(self.ngram_candidates_path)
#         ngrams = []
#         for ngram in df_ngram["words"].to_list():
#             ngrams.append(ngram.split(","))

#         simtok = SimpleTokenizer(
#             stopwords=self.stopwords,
#             min_word_len=min_word_len,
#             min_num_words=min_num_words,
#             ngrams=ngrams,
#             ngram_delimiter=self.ngram.delimiter,
#             verbose=self.verbose,
#         )

#         if self._dataset_ is None:
#             raise ValueError("corpora is not set")
#         with elapsed_timer() as elapsed:
#             self._dataset_.load()
#             self._dataset_.concat_corpora()
#             df = self._dataset_._data
#             df.dropna(subset=[text_key], inplace=True)
#             df[text_key] = apply(
#                 simtok.tokenize,
#                 df[text_key],
#                 description=f"tokenize",
#                 verbose=self.verbose,
#                 use_batcher=use_batcher,
#                 minibatch_size=minibatch_size,
#             )
#             df = df.dropna(subset=[text_key]).reset_index(drop=True)
#             if self.verbose:
#                 print(df.tail())

#             docs = []
#             indexes_to_drop = []
#             for ix in df.index:
#                 doc = df.loc[ix, text_key]
#                 mdoc = self.model.make_doc(doc)
#                 if mdoc:
#                     docs.append(mdoc)
#                 else:
#                     print(f"Skipped - {doc}")
#                     indexes_to_drop.append(ix)
#             df = df.drop(df.index[indexes_to_drop]).reset_index(drop=True)
#             if self.verbose:
#                 print(f"{len(docs)} documents are loaded from: {len(df.index)}.")

#             topic_dists, ll = self.model.infer(
#                 docs, workers=num_workers, iter=iterations
#             )
#             if self.verbose:
#                 print(topic_dists[-1:], ll)
#                 print(f"Total inferred: {len(topic_dists)}, from: {len(df.index)}")

#             if len(topic_dists) == len(df.index):
#                 idx = range(len(topic_dists[0]))
#                 df_infer = pd.DataFrame(topic_dists, columns=[f"topic{i}" for i in idx])
#                 df_infer = pd.concat([df[id_keys], df_infer], axis=1)
#                 output_path = f"{output_dir}/{output_file}"
#                 HyFI.save_data(df_infer, output_path, verbose=self.verbose)
#                 print(f"Corpus is saved as {output_path}")
#             else:
#                 print("The number of inferred is not same as the number of input.")

#             print("Elapsed time is %.2f seconds" % elapsed())
