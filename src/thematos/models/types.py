from typing import Optional

import tomotopy as tp
from hyfi.composer import BaseModel


class CoherenceMetrics(BaseModel):
    u_mass: Optional[float] = None
    c_uci: Optional[float] = None
    c_npmi: Optional[float] = None
    c_v: Optional[float] = None


class ModelSummary(BaseModel):
    timestamp: str = None
    model_file: str = None
    model_id: str = None
    model_type: str = None
    sample_ratio: float = None
    num_docs: int = None
    num_words: int = None
    total_vocabs: int = None
    used_vocabs: int = None
    iterations: int = None
    interval: int = None
    burn_in: int = None
    ll_per_word: float = None
    tw: str = None
    min_cf: int = None
    min_df: int = None
    rm_top: int = None
    k: int = None
    k1: float = None
    k2: float = None
    alpha: float = None
    eta: float = None
    seed: int = None
    perplexity: float = None
    u_mass: float = None
    c_uci: float = None
    c_npmi: float = None
    c_v: float = None


IDF = tp.TermWeight.IDF
ONE = tp.TermWeight.ONE
PMI = tp.TermWeight.PMI
