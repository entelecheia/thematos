"""Library behavior that the workflow tests do not reach: priors, corpus inputs, load, masks."""

import json
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
from thematos.plots.wordcloud import save_image_pages
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
    with pytest.raises(FileNotFoundError):
        _ = WordPrior(data_file=str(tmp_path / "missing.yaml")).priors

    # add() before reading keeps the file's other topics
    early = WordPrior(data_file=saved.data_file)
    early.add(1, "b")
    assert early.priors == {0: ["a"], 1: ["b"]}

    # review of #86: add() + save() creates a new file, parent directories included
    fresh = WordPrior(data_file=str(tmp_path / "new-dir" / "prior.yaml"))
    fresh.add(0, "Rate")
    fresh.save()
    assert WordPrior(data_file=fresh.data_file).priors == {0: ["rate"]}


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
    build_corpus(DOCS.head(2), corpus=corpus)  # new data rebuilds the corpus
    assert len(corpus.corpus) == 2 and corpus.doc_ids == [1, 2]
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
    trained.eval_coherence = False
    trained.train(corpus)  # retraining drops the previous coherence scores
    assert trained.coherence_metrics_dict == {}

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


def test_collage_grid_must_hold_a_page(tmp_path: Path) -> None:
    # #87: 3 images, 20 per page, 1x2 grid used to drop the third image silently
    images = [np.zeros((4, 4, 3), dtype=np.uint8)] * 3
    args = {"output_dir": tmp_path, "output_file_format": "p{page_num}.png", "num_cols": 2}
    with pytest.raises(ValueError, match="num_rows"):
        save_image_pages(images, num_images_per_page=20, num_rows=1, **args)
    assert not list(tmp_path.glob("*.png"))
    files = save_image_pages(images, num_images_per_page=2, num_rows=1, **args)
    assert [f.name for f in files] == ["p0.png", "p1.png"]


def test_training_stops_at_requested_iterations(tmp_path: Path) -> None:
    # #88: iterations=11, interval=10 used to train 20 iterations
    corpus = Corpus(data_file=TRAIN, text_col="tokens")
    model = LdaModel(
        output_dir=tmp_path,
        seed=1,
        model_args={"k": 2},
        train_args={"iterations": 11, "interval": 10},
        eval_coherence=False,
    ).train(corpus)
    assert model.model.global_step == 11
    assert model.ll_per_words["iter"].tolist() == [0, 10]


@pytest.mark.parametrize(("model_type", "name"), [("/abs", "x"), ("LDA", "x/../../escaped")])
def test_model_id_stays_inside_output_dir(tmp_path: Path, model_type: str, name: str) -> None:
    # #89: model_id is joined to output_dir, so it must not resolve outside it
    model = LdaModel(output_dir=tmp_path / "out", model_type=model_type, name=name)
    with pytest.raises(ValueError, match="outside output_dir"):
        _ = model.model_file
    ok = LdaModel(output_dir=tmp_path / "out", name="m")
    assert ok.model_file == tmp_path / "out" / "LDA_m_k(10).mdl"


def test_wordprior_save_keeps_json(tmp_path: Path) -> None:
    # #90: save() wrote YAML into a .json data_file
    path = tmp_path / "prior.json"
    WordPrior(data_file=str(path), prior_data={0: ["Rate"], 1: ["oil"]}).save()
    assert json.loads(path.read_text(encoding="utf-8")) == {"0": ["rate"], "1": ["oil"]}
    assert WordPrior(data_file=str(path)).priors == {0: ["rate"], 1: ["oil"]}


def test_artifacts_are_not_written_through_symlinks(tmp_path: Path) -> None:
    # #92: model_id was checked without the suffix, so a symlinked .mdl path was followed
    out, outside = tmp_path / "out", tmp_path / "outside"
    out.mkdir()
    outside.mkdir()
    model = LdaModel(output_dir=out, name="m")
    (out / "LDA_m_k(10).mdl").symlink_to(outside / "stolen.mdl")
    with pytest.raises(ValueError, match="outside output_dir"):
        _ = model.model_file
    (out / "wordclouds").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="outside output_dir"):
        model._contained(Path(model.topic_wordcloud_file_format.format(topic_id=0)))
    (out / "model-summary.jsonl").symlink_to(outside / "summary.jsonl")
    with pytest.raises(ValueError, match="outside output_dir"):
        _ = model.model_summary_file
    assert model.ll_per_words_file == out / "LDA_m_k(10)-ll_per_word.csv"  # plain paths still work
    assert list(outside.iterdir()) == []
