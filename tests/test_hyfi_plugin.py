"""AC2: HyFI 2.0 discovers the thematos config groups and runs the LDA workflows with manifests."""

import json
from pathlib import Path

import hyfi
import pandas as pd
import pytest
from lexikanon.stopwords import Stopwords

from thematos.datasets import Corpus
from thematos.models import LdaModel, WordcloudConfig, WordPrior

from .conftest import DISCARD, TRAIN, hyfi_cli

WORKFLOWS = sorted(
    p.stem for p in (Path(__file__).parents[1] / "src/thematos/conf/workflow").glob("*.yaml")
)
ALLOW = ["thematos.*", "lexikanon.*"]
SMALL = [
    f"dataset.data_file={TRAIN}",
    "dataset.text_col=tokens",
    "model.model_args.k=3",
    "model.train_args.iterations=20",
    "model.seed=7",
    "model.wc_args.dpi=50",
]


def run_in_process(name: str, *overrides: str) -> hyfi.RunManifest:
    manifest = hyfi.run_workflow(hyfi.compose([f"workflow={name}", *overrides]), allow=ALLOW)
    assert manifest.status == "ok", manifest.error
    return manifest


def test_plugins_lists_thematos(workdir: Path) -> None:
    r = hyfi_cli("plugins", "--json", cwd=workdir)
    assert r.returncode == 0, r.stderr
    row = {"name": "thematos", "module": "thematos.conf", "distribution": "thematos"}
    assert {**row, "available": True} in json.loads(r.stdout)


def test_hyfi_run_trains_into_workspace_with_manifest(workdir: Path) -> None:
    r = hyfi_cli(
        "run",
        "workflow=thematos_lda",
        *SMALL,
        "--allow",
        "thematos.*",
        "--allow",
        "lexikanon.*",
        cwd=workdir,
    )
    assert r.returncode == 0, r.stdout + r.stderr

    out = workdir / "workspace" / "topic" / "model"
    model_id = "LDA_model_k(3)"
    for suffix in [".mdl", "-summary.txt", "-ll_per_word.csv", "-ll_per_word.png"]:
        assert (out / f"{model_id}{suffix}").is_file(), suffix
    assert len(list((out / "wordclouds").glob(f"{model_id}-wordcloud_*.png"))) == 3
    assert (out / "wordcloud_collage" / f"{model_id}_wordcloud_00.png").is_file()
    summary = json.loads((out / "model-summary.jsonl").read_text().splitlines()[-1])
    assert summary["model_id"] == model_id and summary["seed"] == 7

    (manifest_file,) = workdir.glob(".hyfi/runs/*/manifest.json")
    manifest = json.loads(manifest_file.read_text())
    assert manifest["status"] == "ok"
    assert [s["name"] for s in manifest["steps"]] == [
        "read_dataframe",
        "build_corpus",
        "train",
        "summarize",
        "plot",
    ]
    assert manifest["config"]["project"]["workspace"] == str((workdir / "workspace").resolve())
    # Interpolated paths are recorded as resolved at compose time, relative to the run's cwd.
    assert (workdir / manifest["config"]["model"]["output_dir"]).resolve() == out.resolve()


@pytest.mark.parametrize("name", WORKFLOWS)
def test_workflow_examples_resolve(workdir: Path, name: str) -> None:
    r = hyfi_cli("run", f"workflow={name}", *SMALL, "--dry-run", "--json", cwd=workdir)
    assert r.returncode == 0, r.stdout + r.stderr
    assert all(step["ok"] for step in json.loads(r.stdout)["steps"])


def test_sweep_and_infer_workflows(workdir: Path) -> None:
    run_in_process("thematos_lda_sweep", *SMALL, "workflow.steps.2.with.k=[2,3]")
    out = workdir / "workspace" / "topic" / "model"
    lines = (out / "model-summary.jsonl").read_text().splitlines()
    assert [json.loads(line)["model_id"] for line in lines] == [
        "LDA_model(0)_k(2)",
        "LDA_model(1)_k(3)",
    ]

    run_in_process(
        "thematos_lda_infer", *SMALL, f"dataset.data_file={DISCARD}", "model.name='model(1)'"
    )
    inferred = pd.read_parquet(
        out / "inferred_topics/LDA_model(1)_k(3)-inferred_doc_topic_dists.parquet"
    )
    assert len(inferred) == len(pd.read_parquet(DISCARD))
    assert list(inferred.columns) == ["id", "topic0", "topic1", "topic2", "log_likelihood"]


def test_groups_instantiate(workdir: Path) -> None:
    corpus = hyfi.instantiate(hyfi.compose_group("dataset", "topic_corpus"), allow=ALLOW)
    assert isinstance(corpus, Corpus) and isinstance(corpus.stopwords, Stopwords)
    model = hyfi.instantiate(hyfi.compose_group("model", "lda"), allow=ALLOW)
    assert isinstance(model, LdaModel)
    assert isinstance(model.wordprior, WordPrior) and isinstance(model.wc_args, WordcloudConfig)
    assert model.output_dir.resolve() == (workdir / "workspace" / "topic" / "model").resolve()


def test_config_defaults_match_class_defaults(workdir: Path) -> None:
    """A model built from the config group equals one built from class defaults."""
    model = hyfi.instantiate(hyfi.compose_group("model", "lda"), allow=ALLOW)
    plain = LdaModel(output_dir=model.output_dir)
    assert model.model_dump(exclude={"corpus"}) == plain.model_dump(exclude={"corpus"})
    corpus = hyfi.instantiate(hyfi.compose_group("dataset", "topic_corpus"), allow=ALLOW)
    assert corpus.model_dump(exclude={"stopwords"}) == Corpus().model_dump(exclude={"stopwords"})
