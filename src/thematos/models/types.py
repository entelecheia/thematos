from typing import Any, Dict, Optional

import tomotopy as tp
from hyfi.composer import BaseModel


class CoherenceMetrics(BaseModel):
    u_mass: Optional[float] = None
    c_uci: Optional[float] = None
    c_npmi: Optional[float] = None
    c_v: Optional[float] = None


class ModelSummary(BaseModel):
    timestamp: str = None
    model_id: str = None
    model_type: str = None
    num_docs: int = None
    num_words: int = None
    num_total_vocabs: Optional[int] = None
    num_used_vocabs: int = None
    seed: Optional[int] = None
    model_args: Dict[str, Any] = None
    train_args: Dict[str, Any] = None
    perplexity: float = None
    coherence: Optional[Dict[str, float]] = None


IDF = tp.TermWeight.IDF
ONE = tp.TermWeight.ONE
PMI = tp.TermWeight.PMI
