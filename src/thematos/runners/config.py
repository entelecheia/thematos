import itertools
from typing import Any, Dict, List

from hyfi import HyFI
from hyfi.composer import BaseModel

logger = HyFI.getLogger(__name__)


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
