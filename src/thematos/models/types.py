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
    total_vocabs: int = None
    used_vocabs: int = None
    train_config: Dict[str, Any] = None
    ll_per_word: float = None
    perplexity: float = None
    coherence: Dict[str, float] = None


IDF = tp.TermWeight.IDF
ONE = tp.TermWeight.ONE
PMI = tp.TermWeight.PMI
