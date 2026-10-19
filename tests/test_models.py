"""Library behavior that the workflow tests do not reach: priors, corpus inputs, load, masks."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from pydantic import ValidationError

from thematos.datasets import Corpus
from thematos.datasets.corpus import read_dataframe
from thematos.models import LdaModel, WordPrior
from thematos.plots import WordCloud
from thematos.steps import build_corpus

from .conftest import TRAIN, WORD_PRIOR

DOCS = pd.DataFrame(
    {
        "id": [1, 2, 3, 4],
        "text": ["rate hike risk", "rate cut ensure", "bank loan credit", "oil price rise"],
    }
)


def test_wordprior_merges_file_and_data(tmp_path: Path) -> None:
    prior = WordPrior(data_file=str(WORD_PRIOR), prior_data={2: ["Hike", "Rate"], 5: ["OIL"]})
    assert sorted(prior.priors[2]) == ["hike", "rate"]
    assert prior.priors[5] == ["oil"] and len(prior) == 4 and prior
    prior.add(5, "Price")
    assert sorted(prior.priors[5]) == ["oil", "price"]

    saved = WordPrior(data_file=str(tmp_path / "prior.yaml"), prior_data={0: ["a"]})
    saved.save()
    assert WordPrior(data_file=saved.data_file) == saved
    assert repr(saved) == "<WordPrior 1 priors>"
    assert not WordPrior()
    with pytest.raises(ValueError, match="data_file"):
        WordPrior(prior_data={0: ["a"]}).save()


def test_configs_reject_unknown_keys() -> None:
    with pytest.raises(ValidationError):
        LdaModel(output_dir="x", autosave=True)  # a 1.x key


@pytest.mark.parametrize("suffix", [".csv", ".jsonl"])
def test_read_dataframe_formats(tmp_path: Path, suffix: str) -> None:
    path = tmp_path / f"docs{suffix}"
    if suffix == ".csv":
        DOCS.to_csv(path, index=False)
    else:
        DOCS.to_json(path, orient="records", lines=True)
    pd.testing.assert_frame_equal(read_dataframe(path), DOCS)


def test_read_dataframe_rejects_other_formats() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        read_dataframe("docs.xlsx")


def test_corpus_from_text_column() -> None:
    corpus = Corpus(data=DOCS, ngramize=False)
    assert repr(corpus) == "<Corpus>" and not corpus
    assert build_corpus(corpus=corpus) is corpus
    assert repr(corpus) == "<Corpus 4 documents>" and corpus
    assert corpus.doc_ids == [1, 2, 3, 4]
    assert len(corpus.corpus) == 4
    with pytest.raises(ValueError, match="data_file"):
        Corpus().load()


def test_untrained_model_raises(tmp_path: Path) -> None:
    model = LdaModel(output_dir=tmp_path)
    with pytest.raises(ValueError, match="trained"):
        _ = model.model
    with pytest.raises(ValueError, match="trained"):
        _ = model.timestamp
    with pytest.raises(ValueError, match="corpus"):
        model.train()
    assert model.ll_per_words is None


def test_save_load_round_trip(tmp_path: Path) -> None:
    corpus = Corpus(data_file=TRAIN, text_col="tokens")
    trained = LdaModel(
        output_dir=tmp_path, seed=3, model_args={"k": 2}, train_args={"iterations": 10}
    )
    trained.train(corpus).save()

    loaded = LdaModel(output_dir=tmp_path, model_args={"k": 2}).load()
    assert loaded.num_topics == 2
    assert loaded.doc_ids == corpus.doc_ids
    assert loaded.ll_per_words.equals(trained.ll_per_words)
    assert loaded.get_topic_words(0) == trained.get_topic_words(0)

    missing = LdaModel(output_dir=tmp_path, model_args={"k": 9})
    missing._load_model()
    with pytest.raises(ValueError, match="trained"):
        _ = missing.model


def test_wordcloud_mask_from_file(tmp_path: Path) -> None:
    mask_file = tmp_path / "mask.png"
    mask = np.full((60, 120), 255, dtype=np.uint8)
    mask[10:50, 10:110] = 0
    Image.fromarray(mask).save(mask_file)
    out = tmp_path / "wc" / "cloud.png"
    wc = WordCloud(mask=str(mask_file)).generate_from_frequencies(
        {"rate": 3.0, "bank": 2.0, "oil": 1.0}, max_font_size=20, output_file=out
    )
    assert (wc.width, wc.height) == (120, 60)
    assert Image.open(out).size == (120, 60)
