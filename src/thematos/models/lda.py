import logging
from typing import Optional

import tomotopy as tp
from hyfi import HyFI
from tqdm.auto import tqdm

from .base import TopicModel
from .config import LdaConfig

logger = logging.getLogger(__name__)


class LdaModel(TopicModel):
    _config_name_ = "lda"

    model_type: str = "LDA"
    model_args: LdaConfig = LdaConfig()

    # internal attributes
    _model_: Optional[tp.LDAModel] = None

    @property
    def model(self) -> tp.LDAModel:
        if self._model_ is None:
            self._model_ = tp.LDAModel(
                corpus=self.tp_corpus,
                seed=self.seed,
                **self.model_args_dict,
            )
        return self._model_

    def _train(self, model: tp.LDAModel) -> None:
        train_args = self.train_args
        model.burn_in = train_args.burn_in
        model.train(0)
        logger.info("Number of docs: %s", len(model.docs))
        logger.info("Vocab size: %s", model.num_vocabs)
        logger.info("Number of words: %s", model.num_words)
        logger.info("Removed top words: %s", model.removed_top_words)
        logger.info(
            "Training model by iterating over the corpus %s times, %s iterations at a time with %s workers",
            train_args.iterations,
            train_args.interval,
            self.batch.num_workers,
        )

        ll_per_words = []
        for i in tqdm(range(0, train_args.iterations, train_args.interval)):
            model.train(
                iter=train_args.interval,
                workers=self.batch.num_workers,
            )
            logger.info("Iteration: %s\tLog-likelihood: %s", i, model.ll_per_word)
            ll_per_words.append((i, model.ll_per_word))
        self._ll_per_words_ = ll_per_words
        if self.verbose:
            model.summary()
        self._model_ = model

    def _load_model(self):
        model_path = self.model_file
        if HyFI.is_file(model_path):
            self._model_ = tp.LDAModel.load(model_path)
            logger.info("Model loaded from %s", model_path)
        else:
            self._model_ = None
            logger.warning("Model file %s does not exist", model_path)
