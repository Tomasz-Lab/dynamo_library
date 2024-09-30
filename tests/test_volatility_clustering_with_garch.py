import unittest
import pandas as pd
import numpy as np

from dynamo.volatility_clustering_with_garch import VolatilityClusteringWithGARCH


class TestVolatilityClusteringWithGARCH(unittest.TestCase):

    def setUp(self):
        self.valid_df = pd.DataFrame({
            'value': np.random.randn(100)
        })
        self.valid_df['returns'] = self.valid_df.pct_change(1) * 100
        self.empty_df = pd.DataFrame()
        self.invalid_df = pd.DataFrame({
            'value': ['a', 'b', 'c']
        })

    def test_fit_arch_model_invalid_df_type(self):
        with self.assertRaises(TypeError) as context:
            VolatilityClusteringWithGARCH.fit_arch_model(["invalid_data"], "subject")
        self.assertEqual(str(context.exception), "Expected 'df' to be a pandas DataFrame, but got list.")

    def test_fit_arch_model_invalid_subject_type(self):
        with self.assertRaises(TypeError) as context:
            VolatilityClusteringWithGARCH.fit_arch_model(self.valid_df, 123)
        self.assertEqual(str(context.exception), "Expected 'subject' to be a string, but got int.")

    def test_plot_conditional_volatility_invalid_datasets_type(self):
        with self.assertRaises(TypeError) as context:
            VolatilityClusteringWithGARCH.plot_conditional_volatility("invalid_data", 4)
        self.assertEqual(str(context.exception), "Expected 'datasets' to be a list, but got str.")

    def test_plot_conditional_volatility_invalid_dataframe_in_list(self):
        with self.assertRaises(TypeError) as context:
            VolatilityClusteringWithGARCH.plot_conditional_volatility([self.valid_df, "invalid_df"], 2)
        self.assertEqual(str(context.exception), "All elements in 'datasets' must be pandas DataFrames.")


if __name__ == '__main__':
    unittest.main()
