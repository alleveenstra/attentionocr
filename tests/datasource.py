import os
import unittest

from attentionocr import CSVDataSource, FlatDirectoryDataSource


class DataSourceTest(unittest.TestCase):

    def test_csv(self):
        source = CSVDataSource('.', 'sample.txt')
        for filename, label in source:
            print(filename, label)
            assert os.path.isfile(filename)
            assert label in ['test_50x16', 'test_100x32', 'test_288x32', 'test_600x100']

    def test_flat(self):
        source = FlatDirectoryDataSource('test_*.png')
        for filename, label in source:
            print(filename, label)
            assert os.path.isfile(filename)
            assert label in ['test_50x16', 'test_100x32', 'test_288x32', 'test_600x100']
