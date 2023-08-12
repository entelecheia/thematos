from typing import List, Optional, Tuple

from hyfi.composer import BaseModel, Field


class NgramConfig(BaseModel):
    _config_group_ = "/ngrams"
    _config_name_ = "tp_ngrams"

    min_cf: int = Field(
        20, description="Minimum collection frequency of n-grams to be extracted."
    )
    min_df: int = Field(
        10, description="Minimum document frequency of n-grams to be extracted."
    )
    max_len: int = Field(5, description="Maximum length of n-grams to be extracted.")
    max_cand: int = Field(
        1000, description="Maximum number of n-grams to be extracted."
    )
    min_score: Optional[float] = Field(
        None, description="Minium PMI score of n-grams to be extracted."
    )
    normalized: bool = Field(
        False, description="Whether to normalize the PMI score of n-grams."
    )
    workers: int = Field(
        0, description="Number of workers to use for n-gram extraction."
    )
