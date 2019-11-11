import os
import unittest

from attentionocr import CSVDataSource, FlatDirectoryDataSource


class DataSourceTest(unittest.TestCase):

    def test_csv(self):
        source = CSVDataSource('.', 'sample.txt')
        self.check_source(source)

    def test_flat(self):
        source = FlatDirectoryDataSource('test_*.png')
        self.check_source(source)

    def check_source(self, source):
        counter = 0
        for filename, label in source:
            counter += 1
            self.assertTrue(os.path.isfile(filename))
            self.assertIn(label, ['test_50x16', 'test_100x32', 'test_288x32', 'test_600x100'])
        self.assertEqual(4, counter)
