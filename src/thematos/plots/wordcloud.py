import logging
import math
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import wordcloud
from matplotlib.figure import Figure
from PIL import Image
from pydantic import PrivateAttr

from thematos._base import Model

logger = logging.getLogger(__name__)


class WordCloud(Model):
    """Keyword arguments for ``wordcloud.WordCloud``; ``mask`` may be an image path."""

    font_path: str | None = None
    width: int = 400
    height: int = 200
    prefer_horizontal: float = 0.9
    mask: str | np.ndarray | None = None
    contour_width: float = 0
    contour_color: str = "steelblue"
    scale: float = 1
    min_font_size: int = 4
    max_font_size: int | None = None
    font_step: int = 1
    max_words: int = 200
    stopwords: set[str] | None = None
    background_color: str = "black"
    mode: str = "RGB"
    relative_scaling: float | str = "auto"
    color_func: Callable | None = None
    regexp: str | None = None
    collocations: bool = True
    colormap: str = "PuBu"
    normalize_plurals: bool = True
    repeat: bool = False
    include_numbers: bool = False
    min_word_length: int = 0
    collocation_threshold: int = 30

    _wc_: wordcloud.WordCloud | None = PrivateAttr(None)

    @property
    def wc(self) -> wordcloud.WordCloud | None:
        return self._wc_

    def generate_from_frequencies(
        self,
        frequencies: dict[str, float],
        max_font_size: int | None = None,
        output_file: str | Path | None = None,
        verbose: bool = False,
    ) -> wordcloud.WordCloud:
        kwargs = dict(self)
        if max_font_size:
            kwargs["max_font_size"] = max_font_size
        if isinstance(self.mask, str):
            if verbose:
                logger.info("Loading mask from %s", self.mask)
            mask = np.array(Image.open(self.mask))
            kwargs.update(mask=mask, width=mask.shape[1], height=mask.shape[0])

        self._wc_ = wordcloud.WordCloud(**kwargs)
        self._wc_.generate_from_frequencies(frequencies)
        if output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            self._wc_.to_file(str(output_file))
            if verbose:
                logger.debug("Saved wordcloud to %s", output_file)
        return self._wc_


def save_image_pages(
    images: Sequence[wordcloud.WordCloud | np.ndarray],
    output_dir: str | Path,
    output_file_format: str,
    num_images_per_page: int,
    num_cols: int,
    num_rows: int | None = None,
    titles: Sequence[str] | None = None,
    title_fontsize: int = 10,
    title_color: str = "black",
    figsize: tuple[float, float] | None = None,
    width_multiple: float = 4,
    height_multiple: float = 2,
    dpi: int = 100,
) -> list[Path]:
    """Lay images out in grids of ``num_images_per_page`` and save one file per page.

    Raises ``ValueError`` when an explicit ``num_rows`` x ``num_cols`` grid cannot hold a page.
    """
    page_size = min(num_images_per_page, len(images))
    if num_rows and num_rows * num_cols < page_size:
        raise ValueError(
            f"num_rows ({num_rows}) x num_cols ({num_cols}) cannot hold {page_size} images per "
            "page; raise num_rows, leave it unset, or lower num_images_per_page"
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for page_num, start in enumerate(range(0, len(images), num_images_per_page)):
        page = images[start : start + num_images_per_page]
        rows = num_rows or math.ceil(len(page) / num_cols)
        size = figsize or (num_cols * width_multiple, rows * height_multiple)
        fig = Figure(figsize=size)  # no pyplot: leaves the caller's backend alone
        axes = fig.subplots(rows, num_cols, squeeze=False)
        for i, ax in enumerate(axes.flat):
            ax.axis("off")
            if i >= len(page):
                ax.set_visible(False)
                continue
            ax.imshow(page[i])
            if titles and start + i < len(titles):
                ax.set_title(titles[start + i], fontsize=title_fontsize, color=title_color)
        fig.tight_layout()
        output_file = output_dir / output_file_format.format(page_num=page_num)
        fig.savefig(output_file, dpi=dpi)
        files.append(output_file)
    return files
