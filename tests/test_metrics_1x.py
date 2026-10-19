"""AC3: perplexity and coherence match the 1.x result for a fixed seed and corpus.

``data/metrics_1x.json`` was recorded with thematos 0.14.5 on hyfi 1.37.0 (see its
``recorded_with``): train.parquet, word_prior.yaml, seed 42, one worker. Never regenerate it to
make this test pass.

Tolerance: on the recording platform (macOS arm64) the result must match to 1e-9 (it matches
exactly). Elsewhere tomotopy uses another SIMD build, so the Gibbs sampler may take a different
path from the same seed. The corpus and vocabulary must still match exactly, and the metrics must
stay within the seed-to-seed spread measured on the recording platform: over seeds 0-19 the
largest deviation from the reference was 3.0% (perplexity), 0.40% (ll_per_word), and 0.057 (c_v),
so the bounds are 5%, 1%, and 0.1. u_mass, c_uci, and c_npmi vary too much between seeds to bound
usefully (u_mass sd 0.42), so they are only compared on the recording platform.
"""

import json
import math
import platform
from pathlib import Path

import pytest

from thematos.datasets import Corpus
from thematos.models import LdaModel, WordPrior

from .conftest import TRAIN, WORD_PRIOR

REF = json.loads((Path(__file__).parent / "data" / "metrics_1x.json").read_text(encoding="utf-8"))
SAME_PLATFORM = platform.system() == "Darwin" and platform.machine() == "arm64"


@pytest.fixture(scope="module")
def model(tmp_path_factory: pytest.TempPathFactory) -> LdaModel:
    corpus = Corpus(data_file=TRAIN, text_col="tokens")
    m = LdaModel(
        output_dir=tmp_path_factory.mktemp("metrics"),
        seed=REF["seed"],
        num_workers=REF["num_workers"],
        set_wordprior=True,
        wordprior=WordPrior(data_file=str(WORD_PRIOR)),
        model_args=REF["model_args"],
        train_args=REF["train_args"],
    )
    return m.train(corpus)


def test_corpus_and_vocabulary_match(model: LdaModel) -> None:
    assert model.num_docs == REF["num_docs"]
    assert model.num_words == REF["num_words"]
    assert model.num_used_vocab == REF["num_used_vocabs"]


@pytest.mark.skipif(not SAME_PLATFORM, reason="exact match is defined on the recording platform")
def test_metrics_match_exactly_on_recording_platform(model: LdaModel) -> None:
    got = {"perplexity": model.model.perplexity, "ll_per_word": model.model.ll_per_word}
    for key, value in got.items():
        assert math.isclose(value, REF[key], rel_tol=1e-9), (key, value, REF[key])
    for key, value in model.coherence_metrics_dict.items():
        assert math.isclose(value, REF["coherence"][key], rel_tol=1e-9), (key, value)
    top = [
        [w for w, _ in model.model.get_topic_words(k, top_n=10)] for k in range(model.num_topics)
    ]
    assert top == REF["top_words"]


def test_metrics_within_seed_spread(model: LdaModel) -> None:
    assert math.isclose(model.model.perplexity, REF["perplexity"], rel_tol=0.05)
    assert math.isclose(model.model.ll_per_word, REF["ll_per_word"], rel_tol=0.01)
    assert abs(model.coherence_metrics_dict["c_v"] - REF["coherence"]["c_v"]) <= 0.1
