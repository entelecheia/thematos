import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

from hyfi import HyFI
from hyfi.runner import BaseRunner
from tqdm.auto import tqdm

from thematos.datasets import Corpus
from thematos.models import LdaModel

from .config import InferConfig, LdaRunConfig, TopicRunnerResult

logger = logging.getLogger(__name__)


class TopicRunner(BaseRunner):
    _config_group_ = "/runner"
    _config_name_ = "topic"

    task_name: str = "topic"
    batch_name: str = "runner"
    model: LdaModel = LdaModel()
    run_args: LdaRunConfig = LdaRunConfig()
    corpus_to_infer: Corpus = Corpus()
    infer_args: InferConfig = InferConfig()

    num_workers: int = 0
    verbose: bool = False

    calls: Optional[List[Union[str, Dict]]] = ["train"]

    _summaries_: Optional[TopicRunnerResult] = None

    def load_model(
        self,
        batch_name: Optional[str] = None,
        batch_num: Optional[int] = None,
        filepath: Optional[Union[str, Path]] = None,
        **config_kwargs,
    ) -> None:
        self.model.load(
            batch_name=batch_name,
            batch_num=batch_num,
            filepath=filepath,
            **config_kwargs,
        )

    def infer(self) -> None:
        if self.verbose:
            logger.info("Running inference with args: %s", self.infer_args)
        self.load_model(filepath=self.infer_args.model_config_file)
        self.model.infer(
            corpus=self.corpus_to_infer,
            output_file=self.infer_args.output_file,
            iterations=self.infer_args.iterations,
            tolerance=self.infer_args.tolerance,
            num_workers=self.infer_args.num_workers,
            together=self.infer_args.together,
        )

    def train(self) -> None:
        self._summaries_ = TopicRunnerResult(
            runner_task_name=self.task_name,
            runner_batch_name=self.batch_name,
            runner_batch_num=self.batch_num,
            runner_batch_id=self.batch_id,
        )
        self.model.batch.num_workers = self.num_workers
        for args in tqdm(self.run_args.iter_configs(), total=self.run_args.total_runs):
            if self.verbose:
                logger.info("Running with args: %s", args)
            self.model.update_model_args(**args)
            self.model.train()
            self._summaries_.add_model_summary(
                overrides=args,
                summary=self.model.model_summary_dict,
            )
            self.model.set_batch_num(self.model.batch_num + 1)
        self.save()

    def save(self) -> None:
        self.save_result_summary()
        self.save_config()

    @property
    def result_summary_file(self) -> Path:
        summary_file = f"{self.batch_id}_summaries.json"
        return self.task_dir / summary_file

    @property
    def summaries(self) -> Dict:
        return self._summaries_.model_dump() if self._summaries_ else {}

    def save_result_summary(self) -> None:
        if self._summaries_:
            HyFI.save_json(self.summaries, self.result_summary_file)
            logger.info("Saved summaries to %s", self.result_summary_file)
        else:
            logger.warning("No summaries to save")
