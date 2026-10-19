"""AC1: the library imports and trains a small LDA model without hyfi (or Hydra/OmegaConf)."""

import re
import subprocess
import sys
from pathlib import Path

from .conftest import TRAIN

SRC = Path(__file__).parents[1] / "src" / "thematos"

BLOCKED_SCRIPT = """
import sys
from pathlib import Path

BLOCKED = {"hyfi", "hydra", "omegaconf"}


class Block:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ImportError(f"blocked for this test: {name}")


sys.meta_path.insert(0, Block())

import thematos
import thematos.plots
import thematos.steps
from thematos.datasets import Corpus
from thematos.models import LdaModel

data_file, output_dir = sys.argv[1:3]
corpus = Corpus(data_file=data_file, text_col="tokens")
model = LdaModel(output_dir=output_dir, seed=1, model_args={"k": 3}, train_args={"iterations": 20})
model.train(corpus).save().summarize()
assert model.num_topics == 3 and model.model.perplexity > 0
assert (Path(output_dir) / f"{model.model_id}.mdl").is_file()
loaded = sorted(m for m in sys.modules if m.split(".")[0] in BLOCKED)
assert not loaded, loaded
print("ok")
"""


def test_trains_without_hyfi(tmp_path: Path) -> None:
    r = subprocess.run(
        [sys.executable, "-c", BLOCKED_SCRIPT, str(TRAIN), str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "ok"


def test_no_hyfi_import_in_library_code() -> None:
    pattern = re.compile(r"^\s*(import|from)\s+(hyfi|hydra|omegaconf)\b", re.M)
    offenders = [
        str(p.relative_to(SRC))
        for p in SRC.rglob("*.py")
        if pattern.search(p.read_text(encoding="utf-8"))
    ]
    assert offenders == []
