from typing import Any

import tomotopy as tp
from pydantic import BaseModel


class CoherenceMetrics(BaseModel):
    u_mass: float | None = None
    c_uci: float | None = None
    c_npmi: float | None = None
    c_v: float | None = None


class ModelSummary(BaseModel):
    timestamp: str
    model_id: str
    model_type: str
    num_docs: int
    num_words: int
    num_total_vocabs: int | None = None
    num_used_vocabs: int
    seed: int | None = None
    model_args: dict[str, Any]
    train_args: dict[str, Any]
    perplexity: float
    coherence: dict[str, float | None] | None = None


IDF = tp.TermWeight.IDF
ONE = tp.TermWeight.ONE
PMI = tp.TermWeight.PMI
