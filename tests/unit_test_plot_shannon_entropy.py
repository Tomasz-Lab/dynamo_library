import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch

from dynamo.plot_shannon_entropy import DiversityAnalysis


class TestDiversityAnalysis(unittest.TestCase):

    def test_init_invalid_file_paths_type(self):
        with self.assertRaises(TypeError):
            DiversityAnalysis(file_paths="invalid_string", subjects=["subject1", "subject2"])

    def test_init_invalid_subjects_type(self):
        with self.assertRaises(TypeError):
            DiversityAnalysis(file_paths={"subject1": "file.csv"}, subjects="invalid_string")

    def test_init_invalid_file_paths_content(self):
        with self.assertRaises(TypeError):
            DiversityAnalysis(file_paths={1: "file.csv"}, subjects=["subject1"])

    def test_init_invalid_subjects_content(self):
        with self.assertRaises(TypeError):
            DiversityAnalysis(file_paths={"subject1": "file.csv"}, subjects=[123])

    @patch('pandas.read_csv')
    def test_get_trend_invalid_data_type(self, mock_read_csv):

        mock_df = pd.DataFrame({"values": np.random.rand(10)})
        mock_read_csv.return_value = mock_df

        da = DiversityAnalysis(file_paths={"subject1": "file.csv"}, subjects=["subject1"])

        with self.assertRaises(TypeError):
            da.get_trend(data="invalid_string")

    @patch('pandas.read_csv')
    def test_get_trend_invalid_breakpoints_type(self, mock_read_csv):
        mock_df = pd.DataFrame({"values": np.random.rand(10)})
        mock_read_csv.return_value = mock_df

        da = DiversityAnalysis(file_paths={"subject1": "file.csv"}, subjects=["subject1"])
        data = mock_df
        with self.assertRaises(TypeError):
            da.get_trend(data=data, breakpoints="invalid_string")

    @patch('pandas.read_csv')
    def test_get_trend_invalid_breakpoints_content(self, mock_read_csv):
        mock_df = pd.DataFrame({"values": np.random.rand(10)})
        mock_read_csv.return_value = mock_df

        da = DiversityAnalysis(file_paths={"subject1": "file.csv"}, subjects=["subject1"])
        data = mock_df
        with self.assertRaises(TypeError):
            da.get_trend(data=data, breakpoints=[1, "invalid"])


if __name__ == '__main__':
    unittest.main()
