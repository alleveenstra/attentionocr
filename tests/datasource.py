import unittest

from attentionocr import CSVDataSource, FlatDirectoryDataSource


class VectorizerTest(unittest.TestCase):

    def test_csv(self):
        source = CSVDataSource('.', 'sample.txt')
        nxt = next(source)
        assert nxt[1] in ['test_50x16', 'test_100x32', 'test_288x32', 'test_600x100']
        print(nxt)

    def test_flat(self):
        source = FlatDirectoryDataSource('test_*.png')
        nxt = next(source)
        assert nxt[1] in ['test_50x16', 'test_100x32', 'test_288x32', 'test_600x100']
        print(nxt)
