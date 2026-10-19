import os
import subprocess
import sys
from pathlib import Path

import pytest

ASSETS = Path(__file__).parent / "assets"
TRAIN = ASSETS / "sample_data" / "train.parquet"  # 464 documents, column `tokens`
DISCARD = ASSETS / "sample_data" / "discard.parquet"  # 100 documents, same columns
WORD_PRIOR = ASSETS / "words" / "word_prior.yaml"


@pytest.fixture
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An empty cwd without HYFI_* env, so ./conf, workspace/, and .hyfi/runs stay out of the checkout."""
    monkeypatch.chdir(tmp_path)
    for key in list(os.environ):
        if key.startswith("HYFI_"):
            monkeypatch.delenv(key)
    return tmp_path


def hyfi_cli(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "hyfi", *args], cwd=cwd, capture_output=True, text=True
    )
