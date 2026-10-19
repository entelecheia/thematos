# Usage

## Library

Every class is a Pydantic model; construct it with keyword arguments. Unknown keys are errors.

| class | module | what it does |
|---|---|---|
| `Corpus` | `thematos.datasets` | tokenizes a text or token-list column into a tomotopy corpus, removes stopwords, concatenates frequent n-grams |
| `LdaModel` | `thematos.models` | trains `tomotopy.LDAModel`, evaluates coherence, writes artifacts under `output_dir` |
| `WordPrior` | `thematos.models` | seed words per topic, from a YAML/JSON file and/or `prior_data` |
| `WordCloud`, `WordcloudConfig` | `thematos.plots`, `thematos.models` | word cloud settings for `LdaModel.plot()` |

`Corpus` reads `data_file` (`.parquet`, `.csv`, or `.jsonl`) unless you pass a DataFrame as `data`. A column of token lists is joined with spaces before tokenizing, as in 0.x.

`LdaModel` needs an `output_dir`. Its methods return the model, so they chain:

| method | writes |
|---|---|
| `train(corpus)` | nothing; trains and evaluates `coherence_metric_list` |
| `save()` | `<id>.mdl`, `<id>-ll_per_word.csv`, `<id>-doc_topic_dists.parquet`, `<id>-topic_term_dists.parquet`, `<id>-used_vocab.txt`, `<id>-topic_top_words.txt`, `<id>-topic_top_words_dists.csv` |
| `summarize()` | `<id>-summary.txt`, one line in `model-summary.jsonl` (perplexity, coherence, arguments) |
| `plot()` | `<id>-ll_per_word.png`, `wordclouds/<id>-wordcloud_NNN.png`, `wordcloud_collage/<id>_wordcloud_NN.png`, `<id>-ldavis.html` (with pyLDAvis) |
| `load()` | reads back what `save()` wrote for the same `output_dir`, `name`, and `k` |
| `infer(corpus)` | `inferred_topics/<id>-inferred_doc_topic_dists.parquet` |

The model id is `<MODEL_TYPE>_<name>_k(<k>)`, for example `LDA_model_k(10)`.

`seed=None` (the default) lets tomotopy draw a random seed. For a reproducible run set `seed` and keep `num_workers=1`: tomotopy's result depends on the number of workers.

## Config groups

| option | builds |
|---|---|
| `model=lda` | `LdaModel` with `output_dir: ${project.workspace}/topic/model`, `words=wordprior` as `wordprior`, `plot=wordcloud` as `wc_args` |
| `dataset=topic_corpus` | `Corpus` with lexikanon's `stopwords=default` and the 0.x n-gram settings |
| `words=wordprior` | `WordPrior` |
| `plot=wordcloud` | `WordcloudConfig` |

The config files hold the same defaults as the classes. Every node has a `_target_`, so `hyfi.instantiate` builds the object:

```python
import hyfi

model = hyfi.instantiate(hyfi.compose_group("model", "lda", ["model.model_args.k=20"]))
```

## Workflows

The functions in `thematos.steps` take the previous step's result as their first argument:

- `build_corpus(data, corpus=...)`: puts a DataFrame into the corpus and tokenizes it.
- `train(corpus, model=...)`: `model.train(corpus).save()`.
- `summarize(model)`, `plot(model)`: the model methods of the same name.
- `sweep(corpus, model=..., k=[...], alpha=[...], eta=[...])`: one model per combination, named `<name>(0)`, `<name>(1)`, ...; each is saved, summarized, and plotted. Returns the summaries.
- `infer(corpus, model=..., iterations=100, ...)`: `model.load()`, then `model.infer(corpus)`.

The example workflows put `dataset` and `model` at the root of the config, so overrides read `dataset.<key>=...` and `model.<key>=...`:

```bash
# one model: load -> build corpus -> train -> summarize -> plot
hyfi run workflow=thematos_lda dataset.data_file=news.parquet dataset.text_col=tokens model.seed=42

# a sweep over k; summaries land in workspace/topic/model/model-summary.jsonl
hyfi run workflow=thematos_lda_sweep dataset.data_file=news.parquet 'workflow.steps.2.with.k=[5,10,20]'

# topics for new documents, from the model the sweep saved as model(1)
hyfi run workflow=thematos_lda_infer dataset.data_file=new.parquet "model.name='model(1)'" model.model_args.k=10
```

Each run writes `.hyfi/runs/<run id>/manifest.json` with the resolved config, versions, git state, and per-step status. That record replaces the 0.x `configs/model(N)_config.yaml` files.

The steps build `thematos.*` and `lexikanon.stopwords.Stopwords` objects, so pass both when you limit targets:

```bash
hyfi run workflow=thematos_lda dataset.data_file=news.parquet --allow 'thematos.*' --allow 'lexikanon.*'
```
