# ThematOS

[![pypi-image]][pypi-url]
[![license-image]][license-url]
[![DOI][zenodo-image]][zenodo-url]

LDA topic modeling on [tomotopy](https://bab2min.github.io/tomotopy/). ThematOS is a plain Python library; it also ships [HyFI 2.0](https://hyfi.entelecheia.ai) config groups and workflows, so a training run, a parameter sweep, or an inference run can be started from the command line and recorded in a run manifest. The name comes from the Greek θέματος, "of the theme".

- [Documentation][docs-url]
- [Migrating from 0.x](https://thematos.entelecheia.ai/migration/)

ThematOS 1.0 drops the HyFI 1.x API. Pin `thematos<1` to keep it.

## Install

```bash
uv add thematos          # or: pip install thematos
pip install pyLDAvis     # optional: LDAvis HTML next to the other artifacts
```

Python 3.11 or newer. Stopwords come from [lexikanon](https://github.com/entelecheia/lexikanon), which pulls in MeCab through fugashi; on Python 3.15 fugashi builds from source and needs the MeCab library (`apt-get install libmecab-dev` or `brew install mecab`).

## Use it as a library

```python
from thematos.datasets import Corpus
from thematos.models import LdaModel

corpus = Corpus(data_file="news.parquet", text_col="tokens")  # or Corpus(data=dataframe)
model = LdaModel(output_dir="workspace/topic/model", seed=42, model_args={"k": 10})
model.train(corpus)  # trains and computes coherence; writes nothing
model.save()  # model file, log-likelihood trace, distributions, top words
model.summarize()  # tomotopy summary and one JSON line in model-summary.jsonl
model.plot()  # log-likelihood plot and topic word clouds
model.model.perplexity
```

Every file name starts with the model id, `LDA_model_k(10)` here. Importing thematos never imports HyFI.

## Use it from HyFI

Installing thematos registers its config groups through the `hyfi.plugins` entry point:

| group | options |
|---|---|
| `model` | `lda` |
| `dataset` | `topic_corpus` (stopwords from lexikanon's `stopwords` group) |
| `words` | `wordprior` |
| `plot` | `wordcloud` |
| `workflow` | `thematos_lda`, `thematos_lda_sweep`, `thematos_lda_infer` |

```bash
hyfi run workflow=thematos_lda dataset.data_file=news.parquet dataset.text_col=tokens \
  model.model_args.k=20 model.seed=42
```

Artifacts go under the project workspace (`workspace/topic/model/` by default) and the run manifest under `.hyfi/runs/<run id>/manifest.json`. See [Usage](https://thematos.entelecheia.ai/usage/) for the sweep and inference workflows.

## Citation

```tex
@software{lee_2023_8248129,
  author       = {Young Joon Lee},
  title        = {ThematOS: A HyFI-based library for Topic Modeling},
  month        = aug,
  year         = 2023,
  publisher    = {Zenodo},
  version      = {v0.14.1},
  doi          = {10.5281/zenodo.8248128},
  url          = {https://doi.org/10.5281/zenodo.8248128}
}
```

## License

[MIT][license-url]

[zenodo-image]: https://zenodo.org/badge/DOI/10.5281/zenodo.8248128.svg
[zenodo-url]: https://doi.org/10.5281/zenodo.8248128
[pypi-image]: https://img.shields.io/pypi/v/thematos
[pypi-url]: https://pypi.org/project/thematos
[license-image]: https://img.shields.io/github/license/entelecheia/thematos
[license-url]: https://github.com/entelecheia/thematos/blob/main/LICENSE
[docs-url]: https://thematos.entelecheia.ai
