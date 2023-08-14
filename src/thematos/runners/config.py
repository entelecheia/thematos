import itertools
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from hyfi.composer import BaseModel

logger = logging.getLogger(__name__)


class LdaRunConfig(BaseModel):
    _config_group_ = "/runner/config"
    _config_name_ = "lda"

    k: List[int] = [10, 20]
    alpha: List[float] = [0.1]
    eta: List[float] = [0.01]

    @property
    def total_runs(self) -> int:
        return len(self.k) * len(self.alpha) * len(self.eta)

    def iter_configs(self) -> Dict[str, Any]:
        for k, alpha, eta in itertools.product(self.k, self.alpha, self.eta):
            yield dict(k=k, alpha=alpha, eta=eta)


class TopicRunnerResult(BaseModel):
    runner_task_name: str = "topic"
    runner_batch_name: str = "runner"
    runner_batch_num: int = 0
    runner_batch_id: str = ""
    overrides: List[Dict] = []
    model_summary: List[Dict] = []

    def add_model_summary(
        self,
        overrides: Dict,
        summary: Dict,
    ):
        self.overrides.append(overrides)
        self.model_summary.append(summary)


class InferConfig(BaseModel):
    _config_group_ = "/runner/config"
    _config_name_ = "infer_topics"

    model_config_file: Optional[Union[str, Path]] = None
    output_file: Optional[Union[str, Path]] = None
    iterations: int = 100
    tolerance: float = -1
    num_workers: int = 0
    together: bool = False
