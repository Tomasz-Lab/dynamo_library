import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from scipy import signal


class SeasonalityAnalysis:

    def __init__(self, subjects: list[str], datasets: list[pd.DataFrame]):
        self.subjects = subjects
        self.datasets = datasets

    @staticmethod
    def remove_trend(ts: pd.Series) -> np.ndarray:
        lr = LinearRegression()
        X = ts.index.values.reshape(len(ts), 1)
        lr.fit(X, ts.values)
        trend = lr.predict(X)

        feature_detrended = ts.values - trend

        return feature_detrended

    def calculate_flatness_scores(self, index_name: str) -> list[str]:
        flatness_scores = []
        for i, dataset in enumerate(self.datasets):
            ts = dataset[index_name]
            detrended_ts = self.remove_trend(ts).astype(float)
            # Calculate the power spectral density (PSD) using Welch's method
            f, Pxx = signal.welch(detrended_ts, nperseg=len(detrended_ts) // 2)

            # Calculate the spectral flatness
            spectral_flatness = np.exp(np.mean(np.log(Pxx))) / np.mean(Pxx)

            flatness_scores.append(f"{self.subjects[i]}: {spectral_flatness}")
        return flatness_scores


    @staticmethod
    def plot_fft(ts: pd.Series, n_modes: int, subject: str, plot: bool = False) -> tuple[float, pd.DataFrame]:

        # Smooth using rolling mean
        rolling_ts = ts.rolling(window=7).mean().dropna()
        train_detrend = SeasonalityAnalysis.remove_trend(rolling_ts)

        # Ensure variability in detrended data
        if np.var(train_detrend) == 0:
            print("Detrended data is constant.")
            return np.nan, None

        # FFT calculation
        x = train_detrend.reshape(len(train_detrend), )
        dt = 1
        n = len(x)
        fhat = np.fft.fft(x, n)
        psd = fhat * np.conj(fhat) / n
        freq = (1 / (dt * n)) * np.arange(n)
        freq[freq == 0] = np.nan  # Avoid division by zero
        period = 1 / freq

        idxs_half = np.arange(1, np.floor(n / 2), dtype=np.int32)
        train_fft_df = pd.DataFrame(
            list(zip(psd[idxs_half], np.real(psd[idxs_half]), period[idxs_half], freq[idxs_half])),
            columns=['pds', 'pds_real', 'period [days]', 'freq [1/day]'])
        train_fft_df = train_fft_df.sort_values(by=['pds_real'], ascending=False)
        train_fft_df = train_fft_df[
            (train_fft_df['period [days]'] < len(ts) // 2) & (train_fft_df['period [days]'] > 2)]

        # Filter signal only using dominant mode

        # Check if there are enough modes to filter
        if n_modes > len(train_fft_df):
            n_modes = len(train_fft_df)

        threshold = train_fft_df['pds_real'].values[0:n_modes]
        psd_idxs = np.isin(psd.real, threshold)
        psd_clean = psd * psd_idxs  # zero out all the unnecessary powers
        fhat_clean = psd_idxs * fhat  # used to retrieve the signal

        # Inverse FFT
        signal_filtered = np.fft.ifft(fhat_clean)

        # Take the real part of the inverse FFT result
        signal_filtered_real = signal_filtered.real

        # Ensure the filtered signal has variability
        if np.var(signal_filtered_real) == 0:
            print("Filtered signal is constant.")
            return np.nan, None

        score = np.round(stats.spearmanr(signal_filtered_real, train_detrend)[0], 2)

        if plot:
            # Plot spectrogram and ifft

            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 3))
            markerline, stemlines, baseline = ax[0].stem(train_fft_df['period [days]'], train_fft_df['pds_real'],
                                                         linefmt='black')
            markerline.set_markerfacecolor('black')
            markerline.set_markersize(7)
            markerline.set_markeredgewidth(0)
            stemlines.set_linewidth(.9)
            baseline.set_linewidth(3)
            plt.setp(baseline, 'color', 'black')

            ax[0].set_xlabel('period [day]', fontsize=14)
            ax[0].set_ylabel('amplitude [PDS]', fontsize=14)
            ax[0].grid(axis='y', linestyle=':')
            ax[0].grid(axis='x', linestyle=':', color='white')
            ax[0].set_ylim([0, train_fft_df['pds_real'].max() + 0.25])
            ax[0].set_title(subject)

            plt.plot(train_detrend, 'k-', label='raw', lw=1.5)  # , marker='o', markersize=4)
            plt.plot(signal_filtered_real, 'red', lw=3, label='smoothed')
            ax[1].set_xlabel('days', fontsize=14)
            ax[1].set_ylabel('Shannon diversity index', fontsize=14)
            ax[1].set_title(f'seasonal reconstruction score: {score}', fontsize=11)
            ax[1].legend(loc="upper right", ncol=2, fancybox=True, edgecolor='k', fontsize=10)
            plt.rcParams['legend.title_fontsize'] = 'small'

            ax[1].grid(axis='y', linestyle=':')
            ax[1].grid(axis='x', linestyle=':', color='white')
            ax[1].set_title(subject)
            plt.tight_layout()
            plt.show()

        return score, train_fft_df.head(7)

    def calculate_reconstruction_scores(self, max_modes: int = 11) -> pd.DataFrame:
        DF = []
        for ts, s in zip(self.datasets, self.subjects):

            RHO = []
            for n in range(1, max_modes):
                try:
                    rho = self.plot_fft(ts, n, s, False)[0]
                    RHO.append(rho)
                except:
                    pass
            df = pd.DataFrame(RHO, columns=['coeff'])
            df['n_modes'] = range(1, max_modes)
            df['subject'] = s

            DF.append(df)
        return pd.concat(DF)

    @staticmethod
    def plot_n_modes_vs_coeff(df: pd.DataFrame, cmap: dict, output_file: str = 'nmodes_vs_corr.png'):
        fig, ax1 = plt.subplots(figsize=(10, 5), sharex=True, sharey=True)

        sns.lineplot(data=df, x='n_modes', y='coeff', hue='subject', palette=cmap, ax=ax1, lw=4, legend=False)

        ax2 = ax1.twinx()
        sns.scatterplot(data=df, x='n_modes', y='coeff', hue='subject', legend=False, palette=cmap, s=180,
                        edgecolor='k', ax=ax2)

        patches = [mpatches.Patch(color=color, label=label) for label, color in cmap.items()]
        plt.legend(handles=patches, fancybox=True, edgecolor='white', facecolor="white", ncol=4, fontsize=16,
                   loc='upper right', bbox_to_anchor=(1, 1.2))

        ax1.set_xlabel('number of modes', fontsize=16)
        ax1.set_ylabel('seasonal reconstruction score', fontsize=16)
        ax2.set_xlabel('')
        ax2.set_ylabel('')

        ax1.grid(axis='y')
        ax1.tick_params(axis='both', which='major', labelsize=15)
        ax1.tick_params(axis='both', which='minor', labelsize=15)
        ax2.tick_params(axis='both', which='minor', labelsize=0)
        ax2.tick_params(axis='both', which='major', labelsize=0)

        plt.rcParams["axes.edgecolor"] = "black"
        plt.rcParams["axes.linewidth"] = .5

        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.show()
