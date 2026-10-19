from pydantic import Field

from thematos._base import Model


class NgramConfig(Model):
    min_cf: int = Field(20, description="Minimum collection frequency of n-grams to be extracted.")
    min_df: int = Field(10, description="Minimum document frequency of n-grams to be extracted.")
    max_len: int = Field(5, description="Maximum length of n-grams to be extracted.")
    max_cand: int = Field(5000, description="Maximum number of n-grams to be extracted.")
    min_score: float | None = Field(0.5, description="Minium PMI score of n-grams to be extracted.")
    normalized: bool = Field(True, description="Whether to normalize the PMI score of n-grams.")
    workers: int = Field(0, description="Number of workers to use for n-gram extraction.")
    delimiter: str = Field("_", description="Delimiter to use to concatenate n-grams.")
