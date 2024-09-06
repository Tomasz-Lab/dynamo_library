import unittest
from unittest import TestCase
import pandas as pd
from unittest.mock import patch
from matplotlib import pyplot as plt

from dynamo.visualize_taxonomy import VisualizeTaxonomy


class TestVisualizeTaxonomy(TestCase):

    def setUp(self):
        # Dane wejściowe do testów
        self.data_correct = pd.DataFrame({
            'Feature ID': ['F1', 'F2', 'F3'],
            'Taxon': ['Bacteria; Firmicutes; Bacilli; Lactobacillales; Lactobacillaceae; Lactobacillus; L. acidophilus',
                      'Bacteria; Firmicutes; Bacilli; Lactobacillales; Streptococcaceae; Streptococcus; S. thermophilus',
                      'Bacteria; Proteobacteria; Gammaproteobacteria; Enterobacterales; Enterobacteriaceae; Escherichia; E. coli']
        })

        self.counts_correct = pd.DataFrame({
            'F1': [10, 15, 5],
            'F2': [20, 25, 15],
            'F3': [30, 35, 25]
        })

        self.data_non_numeric = pd.DataFrame({
            'Feature ID': ['F1', 'F2', 'F3'],
            'Taxon': ['Bacteria; Firmicutes; Bacilli; Lactobacillales; Lactobacillaceae; Lactobacillus; L. acidophilus',
                      'Bacteria; Firmicutes; Bacilli; Lactobacillales; Streptococcaceae; Streptococcus; S. thermophilus',
                      'Bacteria; Proteobacteria; Gammaproteobacteria; Enterobacterales; Enterobacteriaceae; Escherichia; E. coli']
        })

        self.data_empty = pd.DataFrame(columns=['Feature ID', 'Taxon'])
        self.counts_empty = pd.DataFrame()

        self.data_incorrect_type = "This is not a DataFrame"

    def test_prepare_taxonomy_correct_input(self):
        result = VisualizeTaxonomy.prepare_taxonomy(self.data_correct)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape[0], 3)
        self.assertIn('Phylum', result.columns)

    def test_prepare_taxonomy_incorrect_input_type(self):
        with self.assertRaises(ValueError) as cm:
            VisualizeTaxonomy.prepare_taxonomy(self.data_incorrect_type)
        self.assertEqual(str(cm.exception), "Input must be a pandas DataFrame.")

    def test_prepare_taxonomy_missing_columns(self):
        df_missing_column = pd.DataFrame({'Feature ID': ['F1', 'F2']})
        with self.assertRaises(ValueError) as cm:
            VisualizeTaxonomy.prepare_taxonomy(df_missing_column)
        self.assertEqual(str(cm.exception), "DataFrame must contain 'Taxon' and 'Feature ID' columns.")

    def test_prepare_taxonomy_empty_input(self):
        result = VisualizeTaxonomy.prepare_taxonomy(self.data_empty)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape[0], 0)

    def test_filter_taxonomy_correct_input(self):
        result = VisualizeTaxonomy.filter_taxonomy(self.data_correct, self.counts_correct, 'Genus', 2)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape[1], 3)
        self.assertAlmostEqual(result.sum().sum(), 3, places=1)

    def test_filter_taxonomy_incorrect_input_type(self):
        with self.assertRaises(ValueError) as cm:
            VisualizeTaxonomy.filter_taxonomy(self.data_incorrect_type, self.counts_correct, 'Genus', 2)
        self.assertEqual(str(cm.exception), "Both input_taxonomy_df and counts_df must be pandas DataFrames.")

    def test_filter_taxonomy_empty_input(self):
        result = VisualizeTaxonomy.filter_taxonomy(self.data_empty, self.counts_empty, 'Genus', 2)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(result.empty)

    def test_plot_stacked_taxonomy_correct_input(self):
        filtered_data = VisualizeTaxonomy.filter_taxonomy(self.data_correct, self.counts_correct, 'Genus', 2)
        with patch.object(plt, 'show', return_value=None) as mock_show:
            ax = VisualizeTaxonomy.plot_stacked_taxonomy(filtered_data, 'Genus', 'Subject A')
            self.assertIsInstance(ax, plt.Axes)
            mock_show.assert_called_once()

    def test_plot_stacked_taxonomy_incorrect_input(self):
        with self.assertRaises(ValueError) as cm:
            VisualizeTaxonomy.plot_stacked_taxonomy(self.data_incorrect_type, 'Genus', 'Subject A')
        self.assertEqual(str(cm.exception), "Input df must be a pandas DataFrame.")

    def test_plot_stacked_taxonomy_empty_input(self):
        with self.assertRaises(ValueError) as cm:
            VisualizeTaxonomy.plot_stacked_taxonomy(self.data_empty, 'Genus', 'Subject A')
        self.assertEqual(str(cm.exception), "Input DataFrame must contain numeric data to plot.")


if __name__ == '__main__':
    unittest.main()
