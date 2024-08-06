import pandas as pd
import numpy as np
from arch import arch_model
import matplotlib.pyplot as plt
import seaborn as sns


class VolatilityClusteringWithGARCH:

    @staticmethod
    def fit_arch_model(df: pd.DataFrame, subject: str) -> pd.DataFrame:
        df['returns'] = df.pct_change(1) * 100

        returns = df['returns'][1:]
        model = arch_model(returns,
                           vol='GARCH',
                           mean='zero', lags=[1],
                           p=1,
                           q=1).fit(update_freq=20)
        volatility = model.conditional_volatility.dropna().values.ravel()
        estimated_mean = model.forecast(start=1, horizon=1).mean.values

        # gte conditional volatility
        volatility = model.conditional_volatility  # .dropna().values.ravel()
        estimated_mean = model.forecast(horizon=1).mean.values
        volatility_df = pd.DataFrame(volatility.dropna().values.ravel(), columns=['conditional_volatility'])
        volatility_df['conditional_volatility_2'] = np.sqrt(volatility_df['conditional_volatility'])
        volatility_df['subject'] = subject

        return volatility_df

    @staticmethod
    def plot_conditional_volatility(datasets: list[pd.DataFrame], nrows: int = 4, figsize: tuple = (5, 8),
                                    line_color: str = 'darkblue', line_width: float = 0.7) -> tuple:
        sns.set_style('whitegrid')
        fig, axes = plt.subplots(nrows, 1, figsize=figsize)

        for i, dataset in enumerate(datasets):
            axes[i].plot(dataset['conditional_volatility_2'], '-', color=line_color, lw=line_width,
                         label=dataset['subject'].iloc[0])
            axes[i].set_xlim([min(dataset.index), max(dataset.index)])
            axes[i].set_ylim([min(dataset['conditional_volatility_2']), max(dataset['conditional_volatility_2'])])
            axes[i].legend(edgecolor='w')

        fig.text(0.5, -0.1, 'time point [days]', ha='center', fontsize=12)
        fig.text(-0.1, 0.5, 'conditional volatility of Shannon diversity index', va='center', rotation='vertical',
                 fontsize=12)
        fig.subplots_adjust(bottom=0.2)
        fig.tight_layout()

        return fig, axes
