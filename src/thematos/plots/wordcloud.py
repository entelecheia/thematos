import logging
from pathlib import Path
from typing import Callable, Dict, Optional, Set, Tuple, Union

import numpy as np
import wordcloud
from hyfi import HyFI
from hyfi.composer import BaseModel

logger = logging.getLogger(__name__)


class WordCloud(BaseModel):
    _config_group_ = "/plot"
    _config_name_ = "wordcloud"

    font_path: Optional[str] = None
    width: int = 400
    height: int = 200
    prefer_horizontal: float = 0.9
    mask: Optional[Union[str, np.ndarray]] = None
    contour_width: float = 0
    contour_color: str = "steelblue"
    scale: float = 1
    min_font_size: int = 4
    max_font_size: Optional[int] = None
    font_step: int = 1
    max_words: int = 200
    stopwords: Optional[Set] = None
    background_color: str = "black"
    mode: str = "RGB"
    relative_scaling: Union[float, str] = "auto"
    color_func: Optional[Callable] = None
    regexp: Optional[str] = None
    collocations: bool = True
    colormap: str = "PuBu"
    normalize_plurals: bool = True
    repeat: bool = False
    include_numbers: bool = False
    min_word_length: int = 0
    collocation_threshold: int = 30

    _wc_: Optional[wordcloud.WordCloud] = None

    @property
    def wc(self) -> Optional[wordcloud.WordCloud]:
        return self._wc_

    def generate_from_frequencies(
        self,
        frequencies: Dict[str, float],
        max_font_size: Optional[int] = None,
        output_file: Optional[str] = None,
        lang: str = "en",
        verbose: bool = False,
    ) -> wordcloud.WordCloud:
        _, self.font_path = HyFI.get_plot_font(
            fontpath=self.font_path,
            lang=lang,
        )
        if max_font_size:
            self.max_font_size = max_font_size

        if isinstance(self.mask, str) and HyFI.is_file(self.mask):
            if verbose:
                logger.info("Loading mask from %s", self.mask)
            self.mask = HyFI.load_image_as_ndarray(self.mask)
            self.width = self.mask.shape[1]
            self.height = self.mask.shape[0]

        self._wc_ = wordcloud.WordCloud(**self.kwargs)
        self._wc_.generate_from_frequencies(frequencies)
        if output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            self._wc_.to_file(output_file)
            if verbose:
                logger.debug("Saved wordcloud to %s", output_file)
        return self._wc_
