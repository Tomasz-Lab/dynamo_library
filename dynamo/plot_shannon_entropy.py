import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from typing import List, Dict, Tuple, Optional


class DiversityAnalysis:
    def __init__(self, file_paths: Dict[str, str], subjects: List[str]):
        self.dataframes = {}
        for subject in subjects:
            file_path = file_paths[subject]
            if file_path.endswith('.csv'):
                self.dataframes[subject] = pd.read_csv(file_path)
            elif file_path.endswith('.tsv'):
                self.dataframes[subject] = pd.read_csv(file_path, sep='\t', index_col=0)
        self.subjects = subjects

    def get_trend(self, data: pd.DataFrame, breakpoints: Optional[List[int]] = None) -> Tuple[List[np.ndarray], List[List[float]]]:
        if breakpoints is None:
            breakpoints = []

        segments = []
        prev_point = 0
        for point in breakpoints:
            segments.append(data.iloc[prev_point:point])
            prev_point = point
        segments.append(data.iloc[prev_point:])

        trends = []
        trend_coeffs = []

        for segment in segments:
            X = segment.index.values.reshape(len(segment), 1).astype(float)
            X = sm.add_constant(X)
            y = segment.values.astype(float)
            model = sm.OLS(y, X).fit()
            trend = model.predict()
            trend_coeff = np.round(model.params[1], 3)
            trend_pvalue = np.round(model.pvalues[1], 3)
            trends.append(trend)
            trend_coeffs.append([trend_coeff, trend_pvalue])

        return trends, trend_coeffs

    def plot_trends(self, breakpoints: Dict[str, List[int]], highlights: Dict[str, List[Tuple[int, int]]], title: str = "Diversity index") -> None:
        num_subjects = len(self.subjects)
        fig, axes = plt.subplots(num_subjects, 1, figsize=(10, 5 * num_subjects))

        for i, subject in enumerate(self.subjects):
            data = self.dataframes[subject]
            trends, trend_coeffs = self.get_trend(data, breakpoints.get(subject, None))

            sns.scatterplot(x=data.index.values, y=data.iloc[:, 0].values, s=20, color='k', ax=axes[i])
            sns.lineplot(x=data.index.values, y=data.iloc[:, 0].values, lw=.8, color='k', ax=axes[i])

            start = 0
            for j, trend in enumerate(trends):
                end = start + len(trend)
                axes[i].plot(np.arange(start, end), trend, 'r', lw=2,
                             label=f'trend coefficient: {trend_coeffs[j][0]}, pval: {trend_coeffs[j][1]}')
                start = end

            for highlight in highlights.get(subject, []):
                axes[i].axvspan(highlight[0], highlight[1], alpha=0.2, color='yellow')

            axes[i].set_xlabel('days')
            axes[i].set_ylabel("")
            axes[i].legend(edgecolor='w')
            axes[i].set_title(subject)

        fig.text(0.5, -0.05, 'time point [days]', ha='center', fontsize=12)
        fig.text(-0.05, 0.5, title,  va='center', rotation='vertical', fontsize=12)
        plt.tight_layout()
        plt.show()

