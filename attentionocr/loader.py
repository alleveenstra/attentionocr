from glob import glob
import random


class FlatDirectoryIterator:

    def __init__(self, glob_pattern: str, max_items: int = None):
        self.filenames = glob(glob_pattern)
        random.shuffle(self.filenames)
        if max_items is not None:
            self.filenames = self.filenames[:max_items]
        self.current_index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_index >= len(self.filenames):
            raise StopIteration()
        filename = self.filenames[self.current_index]
        text = filename.split('/')[-1].split('.')[0]  # text is the filename
        self.current_index += 1
        return filename, text
