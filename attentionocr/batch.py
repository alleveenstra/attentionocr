from typing import Generator, Tuple
import logging

from .vectorizer import Vectorizer


LOG = logging.getLogger(__file__)


class BatchGenerator:

    def __init__(self, vectorizer: Vectorizer, batch_size: int = 64):
        self._vectorizer = vectorizer
        self._batch_size = batch_size

    def flow_from_datasource(self, datasource: Generator[Tuple[str, str], None, None], is_training: bool = True):
        batch = []
        while True:
            filename, text = next(datasource)
            try:
                image = self._vectorizer.load_image(filename)
                batch.append([image, text])
            except Exception as e:
                LOG.warning(e)
            if len(batch) == self._batch_size:
                images, texts = zip(*batch)
                yield self._vectorizer.transform(images, texts, is_training)
                batch = []
