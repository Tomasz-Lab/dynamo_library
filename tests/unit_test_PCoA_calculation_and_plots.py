from unittest import TestCase

import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch


from dynamo.PCoA_calculation_and_plots import PCoACalculationAndPlots


class TestPCoACalculationAndPlots(TestCase):
    def setUp(self):
        self.data_correct = pd.DataFrame({
            'Sample1': [1, 2, 3],
            'Sample2': [2, 3, 4],
            'Sample3': [4, 5, 6]
        })
        self.data_incorrect_type = [[1, 2, 3], [2, 3, 4], [4, 5, 6]]
        self.data_empty = pd.DataFrame()
        self.data_non_numeric = pd.DataFrame({'A': ['a', 'b', 'c'], 'B': ['d', 'e', 'f']})
        self.subject_column = np.array(['A', 'B', 'C'])

    def test_create_normalized_aitchinson_distance_matrix_correct_input(self):
        clr_data = self.data_correct.apply(lambda x: np.log(x + 1))  # Mock CLR transformation

        result = PCoACalculationAndPlots.create_normalized_aitchinson_distance_matrix(clr_data.values)
        self.assertIsInstance(result, pd.DataFrame)

    def test_create_normalized_aitchinson_distance_matrix_incorrect_input_type(self):
        with self.assertRaises(ValueError):
            PCoACalculationAndPlots.create_normalized_aitchinson_distance_matrix(self.data_incorrect_type)

    def test_create_normalized_aitchinson_distance_matrix_empty_input(self):
        with self.assertRaises(ValueError):
            PCoACalculationAndPlots.create_normalized_aitchinson_distance_matrix(self.data_empty)

    def test_create_normalized_aitchinson_distance_matrix_non_numeric_data(self):
        with self.assertRaises(ValueError):
            PCoACalculationAndPlots.create_normalized_aitchinson_distance_matrix(self.data_non_numeric)

    def test_run_pca_correct_input(self):
        clr_data = self.data_correct.apply(lambda x: np.log(x + 1))
        distance_matrix = PCoACalculationAndPlots.create_normalized_aitchinson_distance_matrix(clr_data.values)

        result, variance = PCoACalculationAndPlots.run_pca(distance_matrix, n_components=2)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(variance), 2)



    def test_run_pca_incorrect_input_type(self):
        with self.assertRaises(ValueError):
            PCoACalculationAndPlots.run_pca(self.data_incorrect_type, n_components=2)

    def test_run_pca_empty_input(self):
        with self.assertRaises(ValueError):
            PCoACalculationAndPlots.run_pca(self.data_empty, n_components=2)

    def test_run_pca_incorrect_n_components(self):
        clr_data = self.data_correct.apply(lambda x: np.log(x + 1))
        distance_matrix = PCoACalculationAndPlots.create_normalized_aitchinson_distance_matrix(clr_data.values)

        with self.assertRaises(ValueError):
            PCoACalculationAndPlots.run_pca(distance_matrix, n_components=0)

        with self.assertRaises(ValueError):
            PCoACalculationAndPlots.run_pca(distance_matrix, n_components=5)

    def test_visualize_pca_incorrect_input(self):
        with self.assertRaises(ValueError):
            PCoACalculationAndPlots.visualize_pca(self.data_incorrect_type, np.array([0.5, 0.5]))

    def test_pcoa_analyze_correct_input(self):
        with patch.object(PCoACalculationAndPlots, 'visualize_pca', return_value=None) as mock_visualize:
            PCoACalculationAndPlots.pcoa_analyze(self.data_correct, n_components=2, subject_column=None,
                                                 individual=True)
            mock_visualize.assert_called_once()


if __name__ == '__main__':
    unittest.main()
