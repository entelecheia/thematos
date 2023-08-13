import sys
from typing import Any, List, Optional, Tuple

from hyfi.composer import BaseModel, Field

from thematos.plots import WordCloud

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


class TrainSummaryConfig(BaseModel):
    _config_group_ = "/model/summary"
    _config_name_ = "topic_train"

    initial_hp: bool = Field(
        True, description="whether to show the initial parameters at model creation"
    )
    params: bool = Field(
        True, description="whether to show the current parameters of the model"
    )
    topic_word_top_n: int = Field(
        10, description="the number of words by topic to display"
    )
    flush: bool = Field(False, description="whether to forcibly flush the stream")


class WordcloudConfig(BaseModel):
    _config_group_ = "/model/plot"
    _config_name_ = "wordcloud"

    wc: WordCloud = WordCloud()
    top_n: int = 100
    make_collage: bool = True
    num_images_per_page: int = 5
    num_cols: int = 5
    num_rows: Optional[int] = None
    output_file_format: str = "wordcloud_p{page_num:02d}.png"
    titles: Optional[List[str]] = None
    title_fontsize: int = 20
    title_color: str = "green"
    figsize: Optional[Tuple[int, int]] = None
    width_multiple: float = 4
    height_multiple: float = 2
    dpi: int = 300
    mask_dir: Optional[str] = None
    fontpath: Optional[str] = None
    save: bool = True
