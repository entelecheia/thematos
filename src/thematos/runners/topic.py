from pathlib import Path
from typing import Dict, Optional

from hyfi import HyFI
from hyfi.task import BatchTaskConfig
from tqdm.auto import tqdm

from thematos.models import LdaModel

from .config import LdaRunConfig, TopicRunnerResult

logger = HyFI.getLogger(__name__)


class TopicRunner(BatchTaskConfig):
    _config_group_ = "/runner"
    _config_name_ = "topic"

    task_name: str = "topic"
    batch_name: str = "runner"
    model: LdaModel = LdaModel()
    run_args: LdaRunConfig = LdaRunConfig()

    num_workers: int = 0
    verbose: bool = False

    _summaries_: Optional[TopicRunnerResult] = None

    def __call__(self):
        self.run()

    def run(self) -> None:
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
        self.save_result_summary()

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
