import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from typing import List


class LongitudinalRegimesAnalysis:
    def __init__(self, df_characteristics: pd.DataFrame, subjects: List[str], colors=None):

        if colors is None:
            colors = ['#8d99ae', '#0f4c5c', '#fcca46', '#a1c181', '#e36414', '#457b9d']
        self.df_characteristics = df_characteristics
        self.subjects = subjects
        self.colors = colors
        self.regimes = ['noise', 'rare', 'stable_temporal', 'unstable_temporal', 'unstable_prevalent',
                        'stable_prevalent']

    @staticmethod
    def filter_dataset(data: pd.DataFrame, threshold: int = 150) -> pd.DataFrame:
        df = data.iloc[:threshold]
        df_sum = df.sum().reset_index().sort_values(by=0)
        keep_features = df_sum[df_sum[0] != 0]['index'].values
        return df[keep_features]

    def define_regimes(self):

        def conditions(s):
            if (s['prevalence'] < .1) and (s['white_noise_binary'] == 1):
                return 'noise'
            elif (s['prevalence'] < .1) and (s['white_noise_binary'] == 0):
                return 'rare'
            elif (s['prevalence'] > .9) and (s['stationary'] == 1) and (s['white_noise_binary'] == 0):
                return 'stable_prevalent'
            elif (s['prevalence'] > .9) and (s['stationary'] == 0) and (s['white_noise_binary'] == 0):
                return 'unstable_prevalent'
            elif (s['prevalence'] < .9) and (s['prevalence'] > .1) and (s['stationary'] == 1) and (
                    s['white_noise_binary'] == 0):
                return 'stable_temporal'
            elif (s['prevalence'] < .9) and (s['prevalence'] > .1) and (s['stationary'] == 0) and (
                    s['white_noise_binary'] == 0):
                return 'unstable_temporal'

        self.df_characteristics['regime'] = self.df_characteristics.apply(conditions, axis=1)

    def plot_regime_distribution(self):
        regime_df = pd.DataFrame()
        for i, subject in enumerate(self.subjects):
            d = Counter(self.df_characteristics[self.df_characteristics['subject'] == subject]['regime'])
            res_df = pd.DataFrame.from_dict(d, orient='index').reset_index()
            res_df.columns = ['regime', 'count']
            res_df = res_df.set_index('regime').T
            res_df['subject'] = subject
            regime_df = pd.concat([regime_df, res_df], axis=0)

        regime_df = regime_df[
            ['noise', 'rare', 'stable_temporal', 'unstable_temporal', 'unstable_prevalent', 'stable_prevalent',
             'subject']].fillna(0)
        colors = ['#8d99ae', '#0f4c5c', '#fcca46', '#a1c181', '#e36414', '#457b9d']

        ax = regime_df.iloc[:, :6].plot(kind='bar', stacked=True, figsize=(17, 7), width=1, edgecolor='k', lw=0.8,
                                        color=colors)
        ax.set_xticklabels(self.subjects, rotation=0, fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.set_xlabel('subject', fontsize=18)
        ax.set_ylabel('ASV', fontsize=18)
        plt.legend(loc='upper right', bbox_to_anchor=(1, 1.12), fontsize=16, ncol=6)
        plt.grid(axis='x')
        plt.tight_layout()
        plt.show()

    def plot_regimes_over_time(self, df: pd.DataFrame, ftable: pd.DataFrame, subject: str, legend: bool = False):

        df = df
        ftable = ftable.reset_index().rename({'index': 'feature'}, axis=1)

        merged_df = pd.merge(ftable, df[['feature', 'regime']], on='feature')
        merged_df = merged_df.drop('feature', axis=1).groupby(by='regime').sum()
        merged_df_t = merged_df.T
        rel_ab_df = merged_df_t.div(merged_df_t.sum(axis=1), axis=0)
        rel_ab_df = rel_ab_df[self.regimes]

        rel_ab_df_rolling = rel_ab_df.transform(lambda x: x.rolling(window=3).mean()).dropna().reset_index(drop=True)

        ax = rel_ab_df_rolling.plot(kind='bar', stacked=True, width=1, edgecolor='k', lw=0, figsize=(20, 3),
                                    color=self.colors)

        if legend == True:

            plt.legend(bbox_to_anchor=(1, 1.3), loc='upper right', title='REGIME', ncol=6, edgecolor='k', fontsize=14)
            plt.rcParams['legend.title_fontsize'] = 'xx-large'

        elif legend == False:
            ax.get_legend().remove()

        num_ticks = rel_ab_df_rolling.shape[0]
        tick_positions = np.arange(0, num_ticks, 100)
        plt.xticks(tick_positions, [str(i) for i in range(0, num_ticks)][::100])
        plt.tick_params(axis='both', which='major', labelsize=18, labelrotation=0)
        plt.tick_params(axis='both', which='minor', labelsize=18, labelrotation=90)
        plt.ylim([0, 1])
        plt.tight_layout()
        plt.show()

    def loading_regimes_boxplot(self):
        self.df_characteristics['loading'] = self.df_characteristics['PC1_loading'] + self.df_characteristics['PC2_loading']
        sns.set_style('whitegrid')
        plt.figure(figsize=(18, 7))

        sns.boxplot(data=self.df_characteristics, x='regime', y='loading', hue='regime', palette=self.colors, legend=False, fliersize=0,
                    boxprops=dict(edgecolor='k'),
                    capprops=dict(color='k'),
                    whiskerprops=dict(color='k'),
                    medianprops=dict(color='k'))

        sns.stripplot(data=self.df_characteristics, x='regime', y='loading', hue='regime', palette=self.colors, s=10, legend=False)

        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.tick_params(axis='both', which='minor', labelsize=16)
        plt.xlabel('Regime', fontsize=16)
        plt.ylabel('Loading', fontsize=16)
        plt.title('Boxplot of Loading by Regime', fontsize=18)
        plt.grid(axis='x')
        plt.tight_layout()
        plt.show()
