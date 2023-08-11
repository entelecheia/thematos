from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from hyfi import HyFI
from hyfi.task import BatchTaskConfig
from tqdm.auto import tqdm

from thematos.models import LdaModel

from .config import LdaRunConfig

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

    _summaries_: List[Dict] = []

    def __call__(self):
        self.run()

    @property
    def summary_file(self) -> Path:
        summary_file = f"{self.model_name}_summaries.csv"
        return self.output_dir / summary_file

    @property
    def summaries(self) -> List[Dict]:
        if self._summaries_:
            return self._summaries_

    def run(self) -> None:
        self._summaries_ = []
        for args in tqdm(self.run_args.iter_configs(), total=self.run_args.total_runs):
            print(args)
        #     self.model.config.update(config)
        #     self.model.run()
        #     self._summaries_.append(self.model.summary)
        # self.save_summaries()
