import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np

from dynamo.alpha_diversity_unit_root_tests import AlphaDiversityUnitRootTests


class TestAlphaDiversityUnitRootTests(unittest.TestCase):

    def setUp(self):
        self.example_df = pd.DataFrame({
            'value': np.random.rand(100)
        })

    @patch('dynamo.alpha_diversity_unit_root_tests.acorr_ljungbox')
    def test_autocorrelation_presence_autocorrelated(self, mock_acorr_ljungbox):
        mock_acorr_ljungbox.return_value = pd.DataFrame({'lb_pvalue': np.array([0.01] * 30)})

        with patch('builtins.print') as mocked_print:
            AlphaDiversityUnitRootTests.autocorrelation_presence(self.example_df)
            mocked_print.assert_called_once_with('DataFrame is autocorrelated')

    @patch('dynamo.alpha_diversity_unit_root_tests.acorr_ljungbox')
    def test_autocorrelation_presence_not_autocorrelated(self, mock_acorr_ljungbox):
        mock_acorr_ljungbox.return_value = pd.DataFrame({'lb_pvalue': np.array([0.1] * 30)})

        with patch('builtins.print') as mocked_print:
            AlphaDiversityUnitRootTests.autocorrelation_presence(self.example_df)
            mocked_print.assert_called_once_with('DataFrame is not autocorrelated')

    def test_autocorrelation_presence_invalid_input(self):
        with self.assertRaises(TypeError):
            AlphaDiversityUnitRootTests.autocorrelation_presence("invalid_input")


if __name__ == '__main__':
    unittest.main()
