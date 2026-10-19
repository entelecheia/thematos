import logging
from collections.abc import ItemsView, Iterator
from pathlib import Path

import yaml
from pydantic import PrivateAttr

from thematos._base import Model

logger = logging.getLogger(__name__)

PriorType = dict[int, list[str]]


class WordPrior(Model):
    """Seed words per topic, from ``prior_data`` and/or a YAML or JSON ``data_file``."""

    lowercase: bool = True
    prior_data: PriorType | None = None
    data_file: str | None = None
    min_prior_weight: float = 0.01
    max_prior_weight: float = 1.0
    verbose: bool = False

    _priors_: PriorType = PrivateAttr(default_factory=dict)

    @property
    def priors(self) -> PriorType:
        if not self._priors_:
            self.load()
        return self._priors_

    def add(self, id_: int, prior: str | list[str]) -> None:
        """Add a prior to the word prior"""
        if isinstance(prior, str):
            prior = [prior]
        prior = [w.lower() for w in prior] if self.lowercase else prior
        self._priors_[id_] = list(set(self._priors_.get(id_, []) + prior))

    def load(self) -> None:
        """Load words from file and ``prior_data``"""
        priors: dict = {}
        if self.data_file and Path(self.data_file).is_file():
            priors = yaml.safe_load(Path(self.data_file).read_text(encoding="utf-8")) or {}
            if self.verbose:
                logger.info("Loaded %d words from %s", len(priors), self.data_file)
        elif self.data_file:
            logger.warning("Word prior file not found: %s", self.data_file)
        if self.prior_data:
            priors.update(self.prior_data)
        self._priors_ = {
            int(id_): list({w.lower() if self.lowercase else w for w in prior})
            for id_, prior in priors.items()
        }
        if self.verbose:
            logger.info("Loaded %d priors", len(self._priors_))

    def save(self) -> None:
        """Save words to ``data_file``"""
        if not self.priors:
            logger.warning("No priors to save")
            return
        if not self.data_file:
            raise ValueError("WordPrior.save needs `data_file`")
        Path(self.data_file).write_text(
            yaml.safe_dump(self.priors, allow_unicode=True), encoding="utf-8"
        )
        if self.verbose:
            logger.info("Saved %d words to %s", len(self.priors), self.data_file)

    def items(self) -> ItemsView[int, list[str]]:
        return self.priors.items()

    def __iter__(self) -> Iterator[int]:  # type: ignore[override]
        return iter(self.priors)

    def __len__(self) -> int:
        return len(self.priors)

    def __repr__(self) -> str:
        return f"<WordPrior {len(self.priors)} priors>"

    __str__ = __repr__

    def __bool__(self) -> bool:
        return bool(self.priors)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, WordPrior) and self.priors == other.priors

    __hash__ = None  # type: ignore[assignment]
