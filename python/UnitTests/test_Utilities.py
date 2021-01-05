from unittest import TestCase
import Utilities as Ut
import numpy as np


class UtilitiesTest(TestCase):

    def test_get_mean(self):
        seq1 = (Ut.get_mean([1]),
                Ut.get_mean([1, 2, 3]),
                Ut.get_mean([9, 10]))
        seq2 = (1, 2, 9.5)
        self.assertSequenceEqual(seq1, seq2)

    def test_get_mean_nonnumeric(self):
        self.assertEqual(None, Ut.get_mean(['a', 'b', 1, 2, 3]))

    def test_get_mean_none(self):
        self.assertSequenceEqual((Ut.get_mean([]), Ut.get_mean(None)),
                                 (None, None))

    def test_get_std_dev(self):
        mu, sigma = 0, 0.1  # mean and standard deviation
        s = np.random.normal(mu, sigma, 1000)
        self.assertAlmostEqual(np.std(s), Ut.get_std_dev(s))

    def test_get_std_dev_none(self):
        self.assertSequenceEqual((Ut.get_std_dev([]), Ut.get_std_dev(None)),
                                 (None, None))

    def test_get_std_dev_nonnumeric(self):
        self.assertEqual(None, Ut.get_mean(['a', 'b', 1, 2, 3]))

    def test_get_sum(self):
        mu, sigma = 0, 0.1  # mean and standard deviation
        s = np.random.normal(mu, sigma, 1000)
        self.assertAlmostEqual(np.sum(s), Ut.get_sum(s))

    def test_get_sum_none(self):
        self.assertSequenceEqual((Ut.get_std_dev([]), Ut.get_std_dev(None)),
                                 (None, None))

    def test_get_sum_nonnumeric(self):
        self.assertEqual(None, Ut.get_mean(['a', 'b', 1, 2, 3]))