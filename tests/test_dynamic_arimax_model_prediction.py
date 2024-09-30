import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dynamo.dynamic_arimax_model_prediction import DynamicARIMAXModelPrediction


class TestDynamicARIMAXModelPrediction(unittest.TestCase):

    def setUp(self):
        self.signal = np.sin(np.linspace(0, 2 * np.pi, 100)) + np.random.normal(0, 0.1, 100)
        self.df = pd.Series(self.signal)

    def test_smooth_data(self):
        smoothed_data = DynamicARIMAXModelPrediction.smooth_data(self.signal, 0.7)
        self.assertEqual(len(smoothed_data), len(self.signal))
        self.assertIsInstance(smoothed_data, np.ndarray)

        with self.assertRaises(ValueError):
            DynamicARIMAXModelPrediction.smooth_data(np.array([]), 0.7)

    def test_detrend_ts(self):
        detrended = DynamicARIMAXModelPrediction.detrend_ts(self.signal)
        self.assertEqual(len(detrended), len(self.signal))
        self.assertIsInstance(detrended, np.ndarray)

        with self.assertRaises(ValueError):
            DynamicARIMAXModelPrediction.detrend_ts(np.array([]))

    def test_fft_decomposition(self):
        fft_output = DynamicARIMAXModelPrediction.fft_decomposition(self.signal)
        self.assertIsInstance(fft_output, pd.DataFrame)
        self.assertIn('amplitude', fft_output.columns)

        with self.assertRaises(ValueError):
            DynamicARIMAXModelPrediction.fft_decomposition(np.array([]))

    def test_create_fourier_dict(self):
        fft_output = DynamicARIMAXModelPrediction.fft_decomposition(self.signal)
        fourier_dict = DynamicARIMAXModelPrediction.create_fourier_dict(fft_output, self.signal, 3)
        self.assertIsInstance(fourier_dict, dict)
        self.assertGreater(len(fourier_dict), 0)

    def test_create_fourier_df(self):
        fft_output = DynamicARIMAXModelPrediction.fft_decomposition(self.signal)
        fourier_dict = DynamicARIMAXModelPrediction.create_fourier_dict(fft_output, self.signal, 3)
        fourier_df = DynamicARIMAXModelPrediction.create_fourier_df(fourier_dict, len(self.signal))
        self.assertIsInstance(fourier_df, pd.DataFrame)
        self.assertIn('FT_All', fourier_df.columns)

    def test_plot_prediction(self):
        y_true = pd.Series(np.random.normal(0, 1, 100))
        y_pred = pd.Series(np.random.normal(0, 1, 100))
        try:
            DynamicARIMAXModelPrediction.plot_prediction(y_true, y_pred, 'test_subject')
        except Exception as e:
            self.fail(f"plot_prediction raised an exception: {e}")

    def test_get_periods(self):
        periods_df = DynamicARIMAXModelPrediction.get_periods(self.signal, 3, 'test_subject')
        self.assertIsInstance(periods_df, pd.DataFrame)
        self.assertIn('period [days]', periods_df.columns)

    def test_get_prediction_scores(self):
        y_true = np.random.normal(0, 1, 100)
        y_pred = np.random.normal(0, 1, 100)
        scores_df = DynamicARIMAXModelPrediction.get_prediction_scores(y_true, y_pred, 'test_subject')
        self.assertIsInstance(scores_df, pd.DataFrame)
        self.assertIn('mape', scores_df.columns)

    def test_plot_box_stripplots(self):

        scores = pd.DataFrame({
            'subject': ['subject1', 'subject1', 'subject2', 'subject2'] * 10,
            'mape': np.random.rand(40),
            'emd': np.random.rand(40)
        })

        fig, axes = DynamicARIMAXModelPrediction.plot_box_stripplots(scores, 'subject', 'mape', 'emd')

        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(axes, np.ndarray)
        self.assertEqual(len(axes), 2)

        plt.close(fig)

    def test_edge_cases(self):
        # Test with empty input
        with self.assertRaises(ValueError):
            DynamicARIMAXModelPrediction.detrend_ts(np.array([]))

        with self.assertRaises(ValueError):
            DynamicARIMAXModelPrediction.smooth_data(np.array([]))

        with self.assertRaises(ValueError):
            DynamicARIMAXModelPrediction.fft_decomposition(np.array([]))


if __name__ == '__main__':
    unittest.main()
