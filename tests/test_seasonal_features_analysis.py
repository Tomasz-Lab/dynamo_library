import pandas as pd
import unittest
from matplotlib import pyplot as plt
from unittest.mock import patch
from dynamo.seasonal_features_analysis import ASVSeasonalityAnalysis

class TestASVSeasonalityAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df_seasonal = pd.DataFrame({
            'subject': ['A', 'A', 'B', 'B'],
            'feature': ['X', 'Y', 'X', 'Y'],
            'dominant_seasonality': [365, 182, 365, 90],
            'dominant_seasonality_adj': [365, 180, 365, 89],
            'dominant_mode_score': [0.6, 0.3, 0.8, 0.2],
            'white_noise_binary': [1, 0, 1, 0],
            'flattness_score': [0.5, 0.2, 0.7, 0.4]
        })

        cls.explained_fft_df = pd.DataFrame({
            'subject': ['A', 'A', 'B', 'B'],
            'n_modes': [6, 6, 6, 6],
            'seasonal_reconstruction_score': [0.6, 0.3, 0.8, 0.2],
            'feature': ['X', 'Y', 'X', 'Y']
        })

        cls.subject_cmap = {'A': 'blue', 'B': 'green'}
        cls.asv_analysis = ASVSeasonalityAnalysis(cls.df_seasonal, cls.explained_fft_df, cls.subject_cmap)

    def test_class_initialization(self):
        self.assertIsInstance(self.asv_analysis.df_seasonal, pd.DataFrame)
        self.assertIsInstance(self.asv_analysis.explained_fft_df, pd.DataFrame)
        self.assertIsInstance(self.asv_analysis.subject_cmap, dict)
        self.assertEqual(len(self.asv_analysis.df_seasonal), len(self.df_seasonal))
        self.assertEqual(len(self.asv_analysis.explained_fft_df), len(self.explained_fft_df))

    def test_create_subplot(self):
        fig, axes, loc = self.asv_analysis._create_subplot()
        self.assertEqual(len(axes), 2)
        self.assertEqual(fig.get_size_inches()[0], 9)
        self.assertEqual(fig.get_size_inches()[1], 7)

    def test_plot_modes_vs_score(self):
        try:
            self.asv_analysis.plot_modes_vs_score()
        except Exception as e:
            self.fail(f"plot_modes_vs_score raised {e} unexpectedly!")
        finally:
            plt.close()

    def test_plot_fourier_seasonalities(self):
        try:
            self.asv_analysis.plot_fourier_seasonalities()
        except Exception as e:
            self.fail(f"plot_fourier_seasonalities raised {e} unexpectedly!")
        finally:
            plt.close()

    def test_plot_dominant_seasonalities_combined(self):
        try:
            self.asv_analysis.plot_dominant_seasonalities_combined()
        except Exception as e:
            self.fail(f"plot_dominant_seasonalities_combined raised {e} unexpectedly!")
        finally:
            plt.close()

    def test_plot_histogram_seasonality(self):
        try:
            self.asv_analysis.plot_histogram_seasonality()
        except Exception as e:
            self.fail(f"plot_histogram_seasonality raised {e} unexpectedly!")
        finally:
            plt.close()

    def test_plot_seasonal_bacteria(self):
        try:
            self.asv_analysis.plot_seasonal_bacteria()
        except Exception as e:
            self.fail(f"plot_seasonal_bacteria raised {e} unexpectedly!")
        finally:
            plt.close()

    def test_plot_seasonal_vs_flatness(self):
        try:
            self.asv_analysis.plot_seasonal_vs_flatness()
        except Exception as e:
            self.fail(f"plot_seasonal_vs_flatness raised {e} unexpectedly!")
        finally:
            plt.close()

    @patch('seaborn.distplot')
    def test_plot_white_noise_behavior(self, mock_distplot):
        try:
            self.asv_analysis.plot_white_noise_behavior()
        except Exception as e:
            self.fail(f"plot_white_noise_behavior raised {e} unexpectedly!")

        self.assertEqual(mock_distplot.call_count, 2)

        plt.close()

if __name__ == '__main__':
    unittest.main()
