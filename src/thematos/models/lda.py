import logging

import tomotopy as tp

from .base import TopicModel

logger = logging.getLogger(__name__)


class LdaModel(TopicModel):
    model_type: str = "LDA"

    def _new_model(self) -> tp.LDAModel:
        return tp.LDAModel(corpus=self.tp_corpus, seed=self.seed, **self.model_args_dict)

    def _train(self, model: tp.LDAModel) -> None:
        train_args = self.train_args
        model.burn_in = train_args.burn_in
        model.train(0, workers=self.num_workers)
        logger.info("Number of docs: %s", len(model.docs))
        logger.info("Vocab size: %s", model.num_vocabs)
        logger.info("Number of words: %s", model.num_words)
        logger.info("Removed top words: %s", model.removed_top_words)
        logger.info(
            "Training model by iterating over the corpus %s times, %s iterations at a time with %s workers",
            train_args.iterations,
            train_args.interval,
            self.num_workers,
        )
        ll_per_words = []
        for i in range(0, train_args.iterations, train_args.interval):
            model.train(train_args.interval, workers=self.num_workers)
            logger.info("Iteration: %s\tLog-likelihood: %s", i, model.ll_per_word)
            ll_per_words.append((i, model.ll_per_word))
        self._ll_per_words_ = ll_per_words
        if self.verbose:
            model.summary(**self.train_summary_args.model_dump())

    def _load_model(self) -> None:
        model_path = self.model_file
        if model_path.is_file():
            self._model_ = tp.LDAModel.load(str(model_path))
            logger.info("Model loaded from %s", model_path)
        else:
            self._model_ = None
            logger.warning("Model file %s does not exist", model_path)
