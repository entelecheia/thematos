# Migrating from 0.x

ThematOS 1.0 runs on HyFI 2.0 and no longer imports HyFI itself. To stay on the old API, pin `thematos<1` (which needs `hyfi<2`, `lexikanon<1`, and Python below 3.13).

| 0.x | 1.0 |
|---|---|
| `from thematos import HyFI`, `thematos` console command | `import hyfi` only where you compose configs; the `hyfi` CLI |
| `TopicModel(BatchTask)`, `Corpus(BatchTask)` with `task_name`, `batch_name`, `batch`, `path` | plain Pydantic models; `LdaModel(output_dir=...)` |
| files under `workspace/<task_name>/<batch_name>/{models,outputs}` | everything under `output_dir` (`${project.workspace}/topic/model` in the config) |
| model id `LDA_model(0)_k(5)` from the batch number | `LDA_<name>_k(5)`; the sweep names models `model(0)`, `model(1)`, ... |
| `batch.seed` (random when -1), `batch.num_workers` | `seed` (random when null), `num_workers` |
| `save_config()` / `load(batch_num=...)` | the run manifest records the config; `load()` reads the files for the same `output_dir`, `name`, and `k` |
| `train()` with `autosave: true` wrote every artifact | `train(corpus)` writes nothing; chain `.save().summarize().plot()`, or use the workflow steps |
| `Corpus(data_load=...)` (a HyFI `load_dataframe` run) | `Corpus(data_file=...)` or `Corpus(data=dataframe)` |
| corpus doc ids saved to `corpus_doc_ids.parquet` | ids are in `<id>-doc_topic_dists.parquet` |
| `TopicRunner` with `calls: [train, infer]`, `LdaRunConfig`, `InferConfig` | workflows `thematos_lda_sweep` and `thematos_lda_infer` (`thematos.steps.sweep`, `thematos.steps.infer`) |
| `runner/<name>_summaries.json` | `model-summary.jsonl`, one line per model |
| groups `model/config`, `model/train`, `model/summary`, `model/plot`, `ngrams`, `runner`, `about` | `model_args`, `train_args`, `train_summary_args`, and `ngrams` are inline in `model=lda` and `dataset=topic_corpus`; `model/plot` is `plot=wordcloud` |
| `/stopwords@stopwords: __init__` (lexikanon 0.x) | `/stopwords@stopwords: default` (lexikanon 1.x) |
| `LdaConfig.k: null` | `k: 10` |
| `generate_config()` and other HyFI model helpers | removed |
| word cloud font picked by HyFI (`lang`) | `wc.font_path`; wordcloud's default font when unset |

Results did not change: with the same corpus, word priors, seed, and one worker, the tests reproduce the perplexity, coherence, and top words recorded with thematos 0.14.5 on hyfi 1.37.0 (`tests/data/metrics_1x.json`).

Fixed along the way:

- tomotopy 0.14 renamed `train(iter=)` and `infer(iter=)` to `iterations=`; 1.0 requires tomotopy 0.14 or newer.
- `WordPrior.save()` works when `data_file` does not exist yet.
