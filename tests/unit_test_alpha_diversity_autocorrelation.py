import unittest
import pandas as pd
import numpy as np

from dynamo.alpha_diversity_autocorrelation import AlphaDiversityAutocorrelation


class TestAlphaDiversityAutocorrelation(unittest.TestCase):

    def setUp(self):
        self.valid_df = pd.DataFrame({
            'value': np.random.rand(100)
        })
        self.empty_df = pd.DataFrame()
        self.invalid_df = pd.DataFrame({
            'value': ['a', 'b', 'c']
        })

    def test_calculate_and_plot_acf_invalid_ts_type(self):
        with self.assertRaises(TypeError) as context:
            AlphaDiversityAutocorrelation.calculate_and_plot_acf(["invalid_data"], "subject")
        self.assertEqual(str(context.exception), "Expected 'ts' to be a pandas DataFrame, but got list.")

    def test_calculate_and_plot_acf_invalid_subject_type(self):
        with self.assertRaises(TypeError) as context:
            AlphaDiversityAutocorrelation.calculate_and_plot_acf(self.valid_df, 123)
        self.assertEqual(str(context.exception), "Expected 'subject' to be a string, but got int.")

    def test_calculate_and_plot_acf_empty_df(self):
        with self.assertRaises(ValueError) as context:
            AlphaDiversityAutocorrelation.calculate_and_plot_acf(self.empty_df, "subject")
        self.assertEqual(str(context.exception), "Input DataFrame 'ts' is empty.")

    def test_calculate_and_plot_acf_non_numeric_df(self):
        with self.assertRaises(ValueError) as context:
            AlphaDiversityAutocorrelation.calculate_and_plot_acf(self.invalid_df, "subject")
        self.assertEqual(str(context.exception), "Input DataFrame 'ts' must contain numeric data.")


if __name__ == '__main__':
    unittest.main()
