from typing import Optional

from hyfi.composer import BaseModel

from .types import IDF


class LdaConfig(BaseModel):
    _config_group_ = "/model/config"
    _config_name_ = "lda"

    tw: int = int(IDF.value)
    min_cf: int = 5
    min_df: int = 0
    rm_top: int = 0
    k: Optional[int] = None
    alpha: float = 0.1
    eta: float = 0.01


class TrainConfig(BaseModel):
    _config_group_ = "/model/train"
    _config_name_ = "topic"

    burn_in: int = 0
    interval: int = 10
    iterations: int = 100
