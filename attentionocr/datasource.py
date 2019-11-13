import os
import random
from glob import glob
from typing import Optional


class FlatDirectoryDataSource:

    def __init__(self, glob_pattern: str, max_items: Optional[int] = None, looping: bool = True):
        self._examples = glob(glob_pattern)
        random.shuffle(self._examples)
        if max_items is not None:
            self._examples = self._examples[:max_items]
        self._current_index = 0
        self._looping = looping

    def __iter__(self):
        self._current_index = 0
        return self

    def __next__(self):
        if self._current_index >= len(self._examples):
            if not self._looping:
                raise StopIteration()
            self._current_index = 0
        filename = self._examples[self._current_index]
        text = os.path.basename(filename).split('.')[0]  # text is the filename
        self._current_index += 1
        return filename, text


class CSVDataSource:

    def __init__(self, directory: str, filename: str, max_items: Optional[int] = None, sep: str = ';', looping: bool = True):
        self._examples = []
        with open(os.path.join(directory, filename), 'r') as fp:
            for line in fp.readlines():
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
        self._looping = looping

    def __iter__(self):
        self._current_index = 0
        return self

    def __next__(self):
        if self._current_index >= len(self._examples):
            if not self._looping:
                raise StopIteration()
            self._current_index = 0
        example = self._examples[self._current_index]
        self._current_index += 1
        return example
