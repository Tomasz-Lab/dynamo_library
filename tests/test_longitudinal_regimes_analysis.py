import unittest
import pandas as pd
import numpy as np
from dynamo.longitudinal_regimes_analysis import LongitudinalRegimesAnalysis

class TestLongitudinalRegimesAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df_characteristics = pd.DataFrame({
            'subject': ['male', 'female', 'donorA', 'donorB', 'male', 'female'],
            'prevalence': [0.05, 0.95, 0.5, 0.7, 0.01, 0.93],
            'white_noise_binary': [1, 0, 0, 0, 0, 0],
            'stationary': [1, 1, 1, 0, 1, 0],
            'feature': ['f1', 'f2', 'f3', 'f4', 'f5', 'f6']
        })

        cls.datasets = [
            pd.DataFrame(np.random.rand(100, 4)),
            pd.DataFrame(np.random.rand(100, 4)),
            pd.DataFrame(np.random.rand(100, 4)),
            pd.DataFrame(np.random.rand(100, 4)),
            pd.DataFrame(np.random.rand(100, 4)),
            pd.DataFrame(np.random.rand(100, 4))
        ]
        cls.subjects = ['male', 'female', 'donorA', 'donorB']

    def test_apply_conditions(self):
        analysis = LongitudinalRegimesAnalysis(self.df_characteristics, self.subjects)
        analysis.define_regimes()
        self.assertIn('regime', analysis.df_characteristics.columns)
        self.assertEqual(analysis.df_characteristics['regime'][0], 'noise')
        self.assertEqual(analysis.df_characteristics['regime'][1], 'stable_prevalent')

    def test_plot_regime_distribution(self):
        analysis = LongitudinalRegimesAnalysis(self.df_characteristics, self.subjects)
        analysis.define_regimes()
        print(analysis)
        try:
            analysis.plot_regime_distribution()
        except Exception as e:
            self.fail(f"plot_regime_distribution raised an exception: {e}")


if __name__ == '__main__':
    unittest.main()
