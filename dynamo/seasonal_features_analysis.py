from typing import Dict, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

class ASVSeasonalityAnalysis:
    def __init__(self, df_seasonal: pd.DataFrame, explained_fft_df: pd.DataFrame, subject_cmap: Dict[str, str]):
        self.df_seasonal = df_seasonal
        self.explained_fft_df = explained_fft_df
        self.subject_cmap = subject_cmap
        self.box_kwargs = {'color': 'white', 'linewidth': 0.7, 'fliersize': 1}
        self.s_kwargs = {'s': 1, 'color': 'k', 'alpha': 0.5}

    def _create_subplot(self, fig_size: Tuple[int, int] = (9, 7), loc=None):
        if loc is None:
            loc = [[0, 0], [0, 1], [1, 0], [1, 1]]
        fig, axes = plt.subplots(2, 2, figsize=fig_size)
        return fig, axes, loc

    def _plot_line_and_box(self, ax, data, y_col: str, title: str):
        """Helper function to handle boxplot and stripplot with a line."""
        data_grouped = data.groupby(['n_modes', 'subject']).median(numeric_only=True).reset_index()

        sns.boxplot(x=data.n_modes, y=data[y_col], ax=ax, **self.box_kwargs)
        sns.stripplot(x=data.n_modes, y=data[y_col], ax=ax, **self.s_kwargs)
        ax.plot(data_grouped.index.values, data_grouped[y_col].values, 'k-', lw=2)
        ax.axhline(0.5, color='r', linestyle='-.')
        ax.set_title(title)
        ax.set_ylim([-0.01, 1])

    def plot_modes_vs_score(self):
        fig, axes, loc = self._create_subplot()

        for subject, ax in zip(self.subject_cmap.keys(), axes.flatten()):
            data = self.explained_fft_df[self.explained_fft_df['subject'] == subject]
            self._plot_line_and_box(ax, data, 'seasonal_reconstruction_score', subject)
            ax.set_xlabel('')
            ax.set_ylabel('')

        fig.text(0.5, 0.04, 'n Fourier modes', ha='center', fontsize=12)
        fig.text(0.04, 0.5, 'seasonal reconstruction score', va='center', rotation='vertical', fontsize=12)

    def plot_fourier_seasonalities(self):
        df = self.explained_fft_df.groupby(['n_modes', 'subject']).mean(numeric_only=True).reset_index()

        plt.figure(figsize=(9, 7))
        sns.lineplot(data=self.explained_fft_df, x='n_modes', y='seasonal_reconstruction_score', hue='subject', lw=5,
                     err_style='bars', palette=self.subject_cmap)
        sns.scatterplot(data=df, x='n_modes', y='seasonal_reconstruction_score', hue='subject', s=200,
                        palette=self.subject_cmap, legend=False)

        plt.axvline(6, linestyle='-.', color='k')
        plt.ylim([0.2, 0.5])
        plt.xlabel('number of Fourier seasonalities', fontsize=16)
        plt.ylabel('seasonal reconstruction score', fontsize=16)
        plt.legend(edgecolor='k', ncol=4, fontsize=14)
        plt.grid(axis='x')

    def plot_dominant_seasonalities_combined(self):
        fig, axes, loc = self._create_subplot(fig_size=(10, 10))

        for subject, l in zip(self.subject_cmap.keys(), loc):
            data = self.df_seasonal[self.df_seasonal['subject'] == subject]
            sns.scatterplot(x=data['dominant_seasonality'], y=data['dominant_mode_score'], s=10, color='k',
                            edgecolor='w', linewidth=0.1, ax=axes[l[0], l[1]])
            axes[l[0], l[1]].axhline(0.5, color='r', linestyle='-.')
            axes[l[0], l[1]].set_xlabel('')
            axes[l[0], l[1]].set_ylabel('')
            axes[l[0], l[1]].set_title(subject)
            axes[l[0], l[1]].set_ylim([-0.01, 1])

        fig.text(0.5, 0.04, 'dominant seasonality period [days]', ha='center', fontsize=14)
        fig.text(0.04, 0.5, 'seasonal reconstruction score', va='center', rotation='vertical', fontsize=14)

        plt.figure(figsize=(8, 7))
        sns.scatterplot(data=self.df_seasonal, x='dominant_seasonality', y='dominant_mode_score',
                        hue='subject', palette=self.subject_cmap, s=30, edgecolor='k')
        plt.axhline(0.5, linestyle='-.', color='red')
        plt.legend(edgecolor='k', ncol=4, fontsize=14)
        plt.xlabel('dominant Fourier seasonality', fontsize=16)
        plt.ylabel('seasonal reconstruction score', fontsize=16)
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.tick_params(axis='both', which='minor', labelsize=14)

    def plot_histogram_seasonality(self):
        fig, axes, loc = self._create_subplot(fig_size=(10, 10))

        for subject, l in zip(self.subject_cmap.keys(), loc):
            data = self.df_seasonal[
                (self.df_seasonal['subject'] == subject) & (self.df_seasonal['dominant_seasonality_adj'] > 0)]
            sns.histplot(data['dominant_seasonality_adj'], ax=axes[l[0], l[1]], color='grey', bins=20)
            axes[l[0], l[1]].set_xlabel('')
            axes[l[0], l[1]].set_ylabel('')
            axes[l[0], l[1]].set_title(subject, fontsize=16)
            axes[l[0], l[1]].grid(axis='x')
            axes[l[0], l[1]].tick_params(axis='both', which='major', labelsize=14)

        fig.text(0.5, 0.04, 'adjusted dominant seasonality period [days]', ha='center', fontsize=16)
        fig.text(0.04, 0.5, 'count', va='center', rotation='vertical', fontsize=16)

    def plot_seasonal_bacteria(self):
        modes_saturation = self.explained_fft_df[self.explained_fft_df['n_modes'] == 6]
        modes_saturation['seasonal'] = np.where(modes_saturation['seasonal_reconstruction_score'] > 0.5, 1, 0)
        modes_saturation['bacteria'] = np.where(modes_saturation['n_modes'] == 6, 1, 0)

        seasonal_bacteria_df = modes_saturation.groupby(by=['subject']).sum()
        seasonal_bacteria_df = seasonal_bacteria_df[['seasonal', 'bacteria']].reindex(['male', 'female', 'donorA', 'donorB'])

        plt.figure(figsize=(10, 10))
        sns.barplot(x=seasonal_bacteria_df.index, y=seasonal_bacteria_df['bacteria'], color='grey', edgecolor='w', lw=2, label='non seasonal')
        sns.barplot(x=seasonal_bacteria_df.index, y=seasonal_bacteria_df['seasonal'], color='#faae7b', edgecolor='w', lw=2, label='seasonal')
        plt.xlabel('subject', fontsize=16)
        plt.ylabel('number of taxa', fontsize=16)
        plt.legend(edgecolor='k', ncol=2, fontsize=14)
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.tick_params(axis='both', which='minor', labelsize=14)

    def plot_seasonal_vs_flatness(self):
        df = pd.merge(self.explained_fft_df, self.df_seasonal[['feature', 'subject', 'flattness_score']],
                      on=['feature', 'subject'])

        plt.figure(figsize=(7, 5))
        sns.scatterplot(x=df['flattness_score'], y=df['seasonal_reconstruction_score'], color='k', s=5, edgecolor='k')
        plt.axhline(0.4, linestyle='-.', color='r')
        plt.axvline(0.4, linestyle='-.', color='r')
        plt.xlabel('flatness score', fontsize=14)
        plt.ylabel('seasonal reconstruction score for 6 modes', fontsize=14)

    def plot_white_noise_behavior(self):
        modes_saturation = self.explained_fft_df[self.explained_fft_df['n_modes'] == 6]
        modes_saturation['seasonal'] = np.where(modes_saturation['seasonal_reconstruction_score'] > 0.5, 1, 0)
        modes_saturation['bacteria'] = np.where(modes_saturation['n_modes'] == 6, 1, 0)

        seasonal_features_df = modes_saturation[['feature', 'seasonal', 'subject', 'seasonal_reconstruction_score']]
        df = pd.merge(seasonal_features_df, self.df_seasonal, on=['feature', 'subject'])

        plt.figure(figsize=(7, 5))

        ax = sns.distplot(x=df[df.white_noise_binary == 1]['seasonal_reconstruction_score'],
                          fit_kws={"color": "#0077b6"}, kde=False,
                          fit=stats.gamma, hist=None, label="1")

        ax = sns.distplot(x=df[df.white_noise_binary == 0]['seasonal_reconstruction_score'],
                          fit_kws={"color": "#ffd166"}, kde=False,
                          fit=stats.gamma, hist=None, label="0")

        # Get the two lines from the axes to generate shading
        l1 = ax.lines[0]
        l2 = ax.lines[1]

        # Get the xy data from the lines so that we can shade
        x1 = l1.get_xydata()[:, 0]
        y1 = l1.get_xydata()[:, 1]
        x2 = l2.get_xydata()[:, 0]
        y2 = l2.get_xydata()[:, 1]
        ax.fill_between(x1, y1, color="#0077b6", alpha=0.5)
        ax.fill_between(x2, y2, color="#ffd166", alpha=0.5)
        plt.grid(axis='x')
        plt.xlabel('seasonal reconstruction score for 6 modes', fontsize=16)
        plt.ylabel('density', fontsize=16)

        plt.legend(title='white noise behavior', edgecolor='k', ncol=2, fontsize=14)
        plt.rcParams['legend.title_fontsize'] = 'large'

        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.tick_params(axis='both', which='minor', labelsize=12)
