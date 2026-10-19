from pydantic import Field

from thematos._base import Model
from thematos.plots import WordCloud

from .types import IDF


class LdaConfig(Model):
    """Keyword arguments for ``tomotopy.LDAModel``."""

    tw: int = int(IDF.value)
    min_cf: int = 5
    min_df: int = 0
    rm_top: int = 0
    k: int = 10
    alpha: float = 0.1
    eta: float = 0.01


class TrainConfig(Model):
    burn_in: int = 0
    interval: int = 10
    iterations: int = 100


class TrainSummaryConfig(Model):
    """Keyword arguments for ``tomotopy.LDAModel.summary``."""

    initial_hp: bool = Field(
        True, description="whether to show the initial parameters at model creation"
    )
    params: bool = Field(True, description="whether to show the current parameters of the model")
    topic_word_top_n: int = Field(10, description="the number of words by topic to display")
    flush: bool = Field(False, description="whether to forcibly flush the stream")


class WordcloudConfig(Model):
    wc: WordCloud = Field(default_factory=WordCloud)
    top_n: int = 100
    make_collage: bool = True
    num_images_per_page: int = 20
    num_cols: int = 5
    num_rows: int | None = None
    titles: list[str] | None = None
    title_fontsize: int = 14
    title_color: str = "green"
    figsize: tuple[float, float] | None = None
    width_multiple: float = 4
    height_multiple: float = 2
    dpi: int = 300
