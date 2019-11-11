from typing import List, Generator

from .vectorizer import Vectorizer


class BatchGenerator:

    def __init__(self, vectorizer: Vectorizer, batch_size: int = 64):
        self._vectorizer = vectorizer
        self._batch_size = batch_size

    def chunks(self, l: List, n: int) -> Generator[List, None, None]:
        for i in range(0, len(l), n):
            yield l[i:i + n]

    def flow_from_dataset(self, dataset: list, is_training: bool = True):
        current_idx = 0
        batches = list(self.chunks(dataset, self._batch_size))
        while True:
            if current_idx >= len(batches):
                current_idx = 0
            batch = batches[current_idx]
            images, texts = zip(*batch)
            current_idx += 1
            yield self._vectorizer.transform(images, texts, is_training)
