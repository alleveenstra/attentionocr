import os
import random
from glob import glob


class FlatDirectoryDataSource:

    def __init__(self, glob_pattern: str, max_items: int = None):
        self._examples = glob(glob_pattern)
        random.shuffle(self._examples)
        if max_items is not None:
            self._examples = self._examples[:max_items]
        self._current_index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._current_index >= len(self._examples):
            raise StopIteration()
        filename = self._examples[self._current_index]
        text = os.path.basename(filename).split('.')[0]  # text is the filename
        self._current_index += 1
        return filename, text


class CSVDataSource:

    def __init__(self, directory: str, filename: str, max_items: int = None, sep: str = ';'):
        self._examples = []
        with open(os.path.join(directory, filename), 'r') as fp:
            line = fp.readline()
            if sep in line:
                file, txt = line.split(sep=sep, maxsplit=1)
                file = os.path.abspath(os.path.join(directory, file))
                txt = txt.strip()
                if os.path.isfile(file):
                    self._examples.append((file, txt))
        random.shuffle(self._examples)
        if max_items is not None:
            self._examples = self._examples[:max_items]
        self._current_index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._current_index >= len(self._examples):
            raise StopIteration()
        example = self._examples[self._current_index]
        self._current_index += 1
        return example
