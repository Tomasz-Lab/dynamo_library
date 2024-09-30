import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from collections import Counter


class LongitudinalCharacteristics:
    def __init__(self, datasets: dict, subjects: list):
        self.datasets = datasets
        self.subjects = subjects
        self.subject_cmap = {'male': '#d36135', 'female': '#ffb400', 'donorA': '#227c9d', 'donorB': '#7fb069'}
        self.noise_cmap = {0: '#0077b6', 1: '#ffd166'}
        self.configure_plots()

    @staticmethod
    def configure_plots() -> None:
        plt.rcParams['figure.dpi'] = 100
        sns.set_style('whitegrid')
        plt.rcParams["axes.edgecolor"] = "black"
        plt.rcParams["axes.linewidth"] = 1
        plt.rcParams["axes.grid.axis"] = "y"
        plt.rcParams["axes.grid"] = True
        plt.rc('legend', fontsize=10, title_fontsize=10, edgecolor='k')

    @staticmethod
    def filter_dataset(data: pd.DataFrame, threshold: int = 150) -> pd.DataFrame:
        df = data.iloc[:threshold]
        df_sum = df.sum().reset_index().sort_values(by=[0])
        keep_features = df_sum[df_sum[0] != 0]['index'].values
        data_filtered = df[keep_features]
        return data_filtered

    @staticmethod
    def plot_white_noise_tests(df: pd.DataFrame) -> None:
        plt.figure(figsize=(5, 3))
        sns.boxplot(x=df.acf_noise, y=df.flattness_score, color='white', fliersize=0, linewidth=.9, width=.4)
        sns.stripplot(x=df.acf_noise, y=df.flattness_score, hue=df.ljung_box_noise, palette='coolwarm', s=2, alpha=.5)
        plt.legend(edgecolor='k', title='Ljung Box\np value', ncol=1, bbox_to_anchor=(1, 1), markerscale=3)
        plt.xlabel('Autocorrelation absence', fontsize=12)
        plt.ylabel('Flatness score', fontsize=12)
        plt.show()

    # scatterplot
    @staticmethod
    def plot_white_noise(
            df: pd.DataFrame,
            x_vars: list[str],
            y_vars: list[str],
            xlabel: str,
            ylabel: list[str],
            fig_title: str = '',
            marker: str = 's',
            alpha: float = 0.7,
            point_size: int = 20,
            palette: dict = None,
            vertical_line_x: float = None,
            plot_type: str = 'double'
    ) -> None:
        if plot_type == 'double':
            fig, ax = plt.subplots(1, 2, figsize=(10, 3), sharex=True)

            sns.scatterplot(x=df[x_vars[0]],
                            y=df[y_vars[0]],
                            marker=marker,
                            alpha=alpha,
                            s=point_size,
                            palette=palette,
                            ax=ax[0],
                            legend=False)

            sns.scatterplot(x=df[x_vars[1]],
                            y=df[y_vars[1]],
                            marker=marker,
                            alpha=alpha,
                            s=point_size,
                            palette=palette,
                            ax=ax[1],
                            legend=False)

            ax[0].set_ylabel(ylabel[0], fontsize=12)
            ax[1].set_ylabel(ylabel[1], fontsize=12)

            if vertical_line_x is not None:
                ax[0].axvline(vertical_line_x, linestyle=':', color='r')
                ax[1].axvline(vertical_line_x, linestyle=':', color='r')

            fig.text(0.5, -0.1, xlabel, ha='center', fontsize=12)

        elif plot_type == 'single':
            plt.figure(figsize=(7, 5))
            sns.scatterplot(y=df[y_vars[0]],
                            x=df[x_vars[0]],
                            alpha=alpha,
                            s=point_size,
                            palette=palette,
                            marker=marker)

            plt.ylabel(ylabel[0], fontsize=14)
            plt.xlabel(xlabel, fontsize=14)

            if vertical_line_x is not None:
                plt.axvline(vertical_line_x, linestyle='-.', color='r')

            plt.tick_params(axis='both', which='major', labelsize=12)
            plt.tick_params(axis='both', which='minor', labelsize=12)

        if fig_title:
            plt.suptitle(fig_title, fontsize=14)

        plt.show()

    @staticmethod
    def plot_boxplot(
            df: pd.DataFrame,
            x_var: str,
            y_var: str,
            hue_var: str = None,
            xlabel: str = '',
            ylabel: str = '',
            fig_title: str = '',
            figsize: tuple = (7, 5),
            palette: list = None,
            stripplot: bool = False,
            stripplot_params: dict = None,
            legend_params: dict = None,
            fliersize: int = 1,
            width: float = 0.5,
            linewidth: float = 0.7
    ) -> None:

        plt.figure(figsize=figsize)

        sns.boxplot(
            x=x_var,
            y=y_var,
            data=df,
            hue=hue_var,
            palette=palette,
            fliersize=fliersize,
            linewidth=linewidth,
            width=width
        )

        if stripplot and hue_var:
            sns.stripplot(
                x=x_var,
                y=y_var,
                data=df,
                hue=hue_var,
                palette=palette,
                dodge=False,
                **(stripplot_params if stripplot_params else {})
            )
            plt.legend(
                **(legend_params if legend_params else {})
            )

        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)

        if fig_title:
            plt.title(fig_title, fontsize=14)

        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.tick_params(axis='both', which='minor', labelsize=12)

        plt.show()

    def aggregate_white_noise_data(self, df: pd.DataFrame, noise_column='white_noise_binary'):

        WHITE_NOISE_DF = pd.DataFrame()

        for subject in self.subjects:
            d = Counter(df[df['subject'] == subject][noise_column])
            res_df = pd.DataFrame.from_dict(d, orient='index').reset_index()
            res_df.columns = ['white_noise_behavior', 'count']
            res_df = res_df.set_index('white_noise_behavior').T
            res_df['subject'] = subject
            WHITE_NOISE_DF = pd.concat([WHITE_NOISE_DF, res_df])

        WHITE_NOISE_DF = WHITE_NOISE_DF.set_index('subject')
        WHITE_NOISE_DF['total'] = WHITE_NOISE_DF.sum(axis=1)
        WHITE_NOISE_DF = WHITE_NOISE_DF.reset_index()

        return WHITE_NOISE_DF

    @staticmethod
    def plot_white_noise_per_person(df: pd.DataFrame, palette=['#0077b6', '#ffd166'], xlabel='subject', ylabel='number of taxa',
                                    title='White Noise per Person'):
        plt.figure(figsize=(9, 5))

        sns.barplot(data=df, x='subject', y='total', color=palette[0], edgecolor='w', lw=2, errorbar=None, width=0.7)

        sns.barplot(data=df, x='subject', y=df[1], color=palette[1], edgecolor='w', lw=2, errorbar=None, width=0.7)

        signal_patch = mpatches.Patch(color=palette[0], label='signal stationary')
        wn_patch = mpatches.Patch(color=palette[1], label='white noise')

        plt.legend(handles=[wn_patch, signal_patch], ncol=3, edgecolor='k', fontsize=14)
        plt.ylabel(ylabel, fontsize=16)
        plt.xlabel(xlabel, fontsize=16)
        plt.title(title, fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.tick_params(axis='both', which='minor', labelsize=14)
        plt.show()

    @staticmethod
    def plot_histogram(df: pd.DataFrame, subjects, variable, bins=10, color='#cad2c5', edgecolor='k', lw=0.6, figsize=(10, 7)):
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=figsize)

        for i, subject in enumerate(subjects):
            row, col = divmod(i, 2)
            ax[row, col].hist(df[df['subject'] == subject][variable].values, color=color, edgecolor=edgecolor, lw=lw,
                              bins=bins)
            ax[row, col].set_title(subject, size=14)
            ax[row, col].tick_params(axis='both', which='major', labelsize=14)
            ax[row, col].tick_params(axis='both', which='minor', labelsize=14)

        plt.tight_layout()
        plt.show()

    def prepare_stationarity_data(self, df: pd.DataFrame) -> pd.DataFrame:

        df['stationarity'] = np.where((df['ADF_stat'] == 1) | (df['KPSS_stat'] == 1), 'non-stationary', 'stationary')
        df['stationarity'] = np.where(df['ADF_stat'] == 'undefinded', 'undefined', df['stationarity'])
        df = df[df['stationarity'] != 'undefined']

        STATIONARY_DF = pd.DataFrame()

        for subject in self.subjects:
            d = Counter(df[df['subject'] == subject].stationary)
            res_df = pd.DataFrame.from_dict(d, orient='index').reset_index()
            res_df.columns = ['behaviour', 'count']
            res_df = res_df.set_index('behaviour').T
            res_df['subject'] = subject
            STATIONARY_DF = pd.concat([STATIONARY_DF, res_df])

        STATIONARY_DF = STATIONARY_DF.set_index('subject')
        STATIONARY_DF['total'] = STATIONARY_DF.sum(axis=1)
        STATIONARY_DF.columns = ['stationary', 'non_stationary', 'total']

        return STATIONARY_DF

    @staticmethod
    def plot_stationarity_data(STATIONARY_DF: pd.DataFrame, figsize: tuple = (7, 7),
                               palette: list = ['#87b38d', '#ed6a5a']) -> None:

        plt.figure(figsize=figsize)
        sns.barplot(data=STATIONARY_DF, x=STATIONARY_DF.index, y=STATIONARY_DF.total, color=palette[0], edgecolor='w',
                    lw=2)
        sns.barplot(data=STATIONARY_DF, x=STATIONARY_DF.index, y=STATIONARY_DF['stationary'], color=palette[1],
                    edgecolor='w', lw=2)

        signal_patch = mpatches.Patch(color=palette[1], label='stationary')
        wn_patch = mpatches.Patch(color=palette[0], label='non-stationary')

        plt.legend(handles=[signal_patch, wn_patch], edgecolor='k', ncol=2, fontsize=14)
        plt.ylabel('number of taxa', fontsize=16)
        plt.xlabel('subject', fontsize=16)
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.tick_params(axis='both', which='minor', labelsize=14)
        plt.show()

    @staticmethod
    def plot_features_heatmap(df: pd.DataFrame) -> None:
        df_corr = df[['std', 'mean', 'prevalence', 'PC1_loading', 'PC2_loading', 'flattness_score',
                           'dominant_seasonality_adj', 'seasonal', 'trend', 'stationary', 'non_stationary',
                           'white_noise_binary', 'lag_1_corr', 'lag_2_corr', 'lag_3_corr']].astype(float)

        df_corr.columns = ['Standard Deviation', 'Mean', 'Prevalence', 'PC1 Loading', 'PC2 Loading',
                           'Flatness Score', 'Dominant Seasonality Adj', 'Seasonal', 'Trend',
                           'Stationarity', 'Non Stationarity', 'White Noise Behavior',
                           '1st Lag ACF', '2nd Lag ACF', '3rd Lag ACF']

        corr_matrix = df_corr.corr(method='spearman').round(2)
        mask_con_corr = corr_matrix[(corr_matrix >= 0.1) | (corr_matrix <= -0.1)]
        matrix = np.triu(mask_con_corr)

        plt.figure(figsize=(20, 15))
        ax = sns.heatmap(corr_matrix, annot=True, edgecolor='black', lw=1.3, fmt='.1g', cmap='coolwarm', mask=matrix,
                         vmin=-1, vmax=1,
                         cbar_kws={"shrink": .6, "orientation": "horizontal", "location": "top"},
                         annot_kws={"size": 16})
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=20)
        plt.show()
