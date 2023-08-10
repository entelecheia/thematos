from typing import Dict, List, Optional

from hyfi import HyFI
from hyfi.composer import BaseModel

logger = HyFI.getLogger(__name__)


class WordPrior(BaseModel):
    _config_group_: str = "/words"
    _config_name_: str = "wordprior"

    name: str = "wordprior"
    lowercase: bool = False
    words_list: Optional[Dict[str, float]] = None
    words_path: Optional[str] = None
    verbose: bool = False

    _loaded_: bool = False
    _words_: Dict[str, float] = {}

    @property
    def words(self) -> List[str]:
        if self._loaded_:
            return self._words_
        self.load()
        return self._words_

    def load(self):
        """Load words from file or function"""
        if self.words_path:
            self._words_ = HyFI.to_dict(HyFI.load(self.words_path))
            if self.verbose:
                logger.info(
                    "Loaded %d words from %s",
                    len(self._words_),
                    self.words_path,
                )
        else:
            self._words_ = {}

        if self.words_list:
            for word, prior in self.words_list.items():
                self._words_[word] = prior

        if self.lowercase:
            self._words_ = {word.lower(): prior for word, prior in self._words_.items()}
        if self.verbose:
            logger.info("Loaded %d words", len(self._words_))
        self._loaded_ = True

    def save(self):
        """Save words to file"""
        if not self._loaded_:
            return
        HyFI.save(self._words_, self.words_path)
        if self.verbose:
            logger.info("Saved %d words to %s", len(self._words_), self.words_path)

    def items(self):
        return self.words.items()

    def __iter__(self):
        return iter(self.words)

    def __len__(self):
        return len(self.words)

    def __contains__(self, word):
        return self.words.__contains__(word)

    def __getitem__(self, word):
        return self.words.__getitem__(word)

    def __repr__(self):
        return f"<WordPrior {len(self.words)} words>"

    def __str__(self):
        return f"<WordPrior {len(self.words)} words>"

    def __bool__(self):
        return bool(self.words)

    def __eq__(self, other: "WordPrior"):
        return self.words == other.words

    def __ne__(self, other: "WordPrior"):
        return self.words != other.words
