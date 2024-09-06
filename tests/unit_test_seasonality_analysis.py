import unittest
import pandas as pd
import numpy as np

from dynamo.seasonality_analysis import SeasonalityAnalysis


class TestSeasonalityAnalysis(unittest.TestCase):

    def setUp(self):
        self.valid_subjects = ['subject1', 'subject2']
        self.valid_data = [pd.DataFrame({'value': np.random.randn(100)}), pd.DataFrame({'value': np.random.randn(100)})]
        self.analysis = SeasonalityAnalysis(self.valid_subjects, self.valid_data)

    def test_init_invalid_subjects_type(self):
        with self.assertRaises(TypeError):
            SeasonalityAnalysis("invalid_subject", self.valid_data)

    def test_init_invalid_datasets_type(self):
        with self.assertRaises(TypeError):
            SeasonalityAnalysis(self.valid_subjects, "invalid_data")

    def test_remove_trend_invalid_ts_type(self):
        with self.assertRaises(TypeError):
            SeasonalityAnalysis.remove_trend("not_a_series")

    def test_remove_trend_empty_series(self):
        with self.assertRaises(ValueError):
            SeasonalityAnalysis.remove_trend(pd.Series([]))

    def test_plot_fft_invalid_ts_type(self):
        with self.assertRaises(TypeError):
            SeasonalityAnalysis.plot_fft("not_a_series", 3, 'subject')

    def test_plot_fft_invalid_n_modes(self):
        with self.assertRaises(ValueError):
            SeasonalityAnalysis.plot_fft(pd.Series(np.random.randn(100)), -1, 'subject')

    def test_calculate_flatness_scores_invalid_index(self):
        with self.assertRaises(ValueError) as context:
            self.analysis.calculate_flatness_scores('invalid_column')

        self.assertTrue("Column 'invalid_column' does not exist" in str(context.exception))

    def test_calculate_reconstruction_scores(self):
        result = self.analysis.calculate_reconstruction_scores(max_modes=5)
        self.assertTrue(isinstance(result, pd.DataFrame))

    def test_plot_n_modes_vs_coeff_invalid_df(self):
        with self.assertRaises(TypeError):
            SeasonalityAnalysis.plot_n_modes_vs_coeff("not_a_dataframe", {'subject1': 'blue'})

    def test_plot_n_modes_vs_coeff_invalid_cmap(self):
        with self.assertRaises(TypeError):
            SeasonalityAnalysis.plot_n_modes_vs_coeff(pd.DataFrame({'n_modes': [1, 2], 'coeff': [0.5, 0.6]}), "not_a_dict")

if __name__ == '__main__':
    unittest.main()
