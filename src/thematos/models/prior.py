import logging
from typing import Dict, List, Optional, Union

from hyfi import HyFI
from hyfi.composer import BaseModel

logger = logging.getLogger(__name__)

PriorType = Dict[int, List[str]]


class WordPrior(BaseModel):
    _config_group_: str = "/words"
    _config_name_: str = "wordprior"

    lowercase: bool = True
    prior_data: Optional[PriorType] = None
    data_file: Optional[str] = None
    min_prior_weight: float = 0.01
    max_prior_weight: float = 1.0
    verbose: bool = False

    _priors_: PriorType = {}

    @property
    def priors(self) -> List[str]:
        if not self._priors_:
            self.load()
        return self._priors_

    def add(self, id_: int, prior: Union[str, List[str]]):
        """Add a prior to the word prior"""
        if isinstance(prior, str):
            prior = [prior]
        prior = [w.lower() for w in prior] if self.lowercase else prior
        self._priors_[id_] = list(set(self._priors_.get(id_, []) + prior))

    def load(self):
        """Load words from file or function"""
        if self.data_file:
            self._priors_ = HyFI.to_dict(HyFI.load(self.data_file))
            if self.verbose:
                logger.info(
                    "Loaded %d words from %s",
                    len(self._priors_),
                    self.data_file,
                )
        else:
            self._priors_ = {}

        if self.prior_data:
            for id_, prior in self.prior_data.items():
                self._priors_[id_] = prior

        self._priors_ = {
            int(id_): list({w.lower() if self.lowercase else w for w in prior})
            for id_, prior in self._priors_.items()
        }
        if self.verbose:
            logger.info("Loaded %d priors", len(self._priors_))

    def save(self):
        """Save words to file"""
        if not self.priors:
            logger.warning("No priors to save")
            return
        HyFI.save(self.priors, self.data_file)
        if self.verbose:
            logger.info("Saved %d words to %s", len(self.priors), self.data_file)

    def items(self):
        return self.priors.items()

    def __iter__(self):
        return iter(self.priors)

    def __len__(self):
        return len(self.priors)

    def __repr__(self):
        return f"<WordPrior {len(self.priors)} priors>"

    def __str__(self):
        return f"<WordPrior {len(self.priors)} priors>"

    def __bool__(self):
        return bool(self.priors)

    def __eq__(self, other: "WordPrior"):
        return self.priors == other.priors

    def __ne__(self, other: "WordPrior"):
        return self.priors != other.priors
