import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from skbio.stats import composition
from scipy import signal
import librosa.feature
import warnings
from statsmodels.tools.sm_exceptions import InterpolationWarning
from scipy import stats
import librosa

warnings.simplefilter('ignore', InterpolationWarning)


class LongitudinalFeatureSelector:
    def __init__(self, datasets: dict, subjects: list):
        self.datasets = datasets
        self.subjects = subjects

    def filter_dataset(self, threshold: int = 150):
        for subject in self.subjects:
            df = self.datasets[subject].iloc[:threshold]
            df_sum = df.sum().reset_index().sort_values(by=0)
            keep_features = df_sum[df_sum[0] != 0]['index'].values
            self.datasets[subject] = df[keep_features]

    def calculate_mean_std(self):
        mean_df = pd.DataFrame()
        std_df = pd.DataFrame()

        for subject in self.subjects:
            dataset = self.datasets[subject]
            mean_tmp = dataset.mean().reset_index().rename({'index': 'feature', 0: 'mean'}, axis=1)
            mean_tmp['subject'] = subject
            mean_df = pd.concat([mean_df, mean_tmp])

            std_tmp = dataset.std().reset_index().rename({'index': 'feature', 0: 'std'}, axis=1)
            std_tmp['subject'] = subject
            std_df = pd.concat([std_df, std_tmp])

        return mean_df, std_df

    @staticmethod
    def remove_trend(ts: pd.Series) -> np.ndarray:
        lr = LinearRegression()
        X = ts.index.values.reshape(len(ts), 1)
        lr.fit(X, ts.values)
        trend = lr.predict(X)
        return ts.values - trend

    @staticmethod
    def test_for_white_noise(ts: pd.Series) -> float:
        detrended_ts = LongitudinalFeatureSelector.remove_trend(ts)
        ljung_box_results = sm.stats.acorr_ljungbox(detrended_ts, lags=len(detrended_ts) // 2)
        return ljung_box_results['lb_pvalue'].mean()

    def run_ljung_box_test(self) -> pd.DataFrame:
        ljung_box_df = []

        for subject in self.subjects:
            dataset = self.datasets[subject]
            results = [{'feature': col, 'ljung_box_noise': self.test_for_white_noise(dataset[col])} for col in
                       dataset.columns]
            result_df = pd.DataFrame(results)
            result_df['subject'] = subject
            ljung_box_df.append(result_df)

        return pd.concat(ljung_box_df).reset_index(drop=True)

    def test_acf_noise(self) -> pd.DataFrame:
        acf_noise_df = []

        for subject in self.subjects:
            dataset = self.datasets[subject]
            subject_acf_noise = []

            for feature in dataset.columns:
                ts = dataset[feature]
                acf_vals, acf_ci, _, _ = sm.tsa.acf(ts, nlags=len(ts) // 2, fft=False, alpha=0.05, qstat=True)
                centered_acf_ci = acf_ci - np.stack([acf_vals, acf_vals], axis=1)
                acf_df = pd.DataFrame({'acf': np.abs(acf_vals[1:]), 'ci': np.abs(centered_acf_ci[1:, 0])})
                acf_noise = 1 if acf_df[acf_df['acf'] > acf_df['ci']].empty else 0
                subject_acf_noise.append({'feature': feature, 'acf_noise': acf_noise})

            acf_noise_result_df = pd.DataFrame(subject_acf_noise)
            acf_noise_result_df['subject'] = subject
            acf_noise_df.append(acf_noise_result_df)

        return pd.concat(acf_noise_df).reset_index(drop=True)

    def calculate_flatness(self) -> pd.DataFrame:
        flatness_df = []

        for subject in self.subjects:
            dataset = self.datasets[subject]
            subject_flatness = []

            for col in dataset.columns:
                ts = dataset[col]
                detrended_ts = self.remove_trend(ts).astype(float)

                nperseg = min(256, len(detrended_ts))

                _, _, Sxx = signal.spectrogram(detrended_ts, nperseg=nperseg)
                flatness = librosa.feature.spectral_flatness(S=Sxx, n_fft=len(ts))
                subject_flatness.append({'feature': col, 'flatness_score': flatness[0][0]})

            flatness_result_df = pd.DataFrame(subject_flatness)
            flatness_result_df['subject'] = subject
            flatness_df.append(flatness_result_df)

        return pd.concat(flatness_df).reset_index(drop=True)

    @staticmethod
    def noise_flag_df(df: pd.DataFrame) -> int:
        if (df['flatness_score'] >= 0.4) and (df['ljung_box_noise'] > 0.05):
            return 1
        else:
            return 0

    def analyse_pca_loadings(self) -> pd.DataFrame:
        pca_df = []

        for subject in self.subjects:
            dataset = self.datasets[subject]
            X = composition.clr(dataset + 1)
            pca = PCA(n_components=2)
            pca.fit(X)
            loadings_df = pd.DataFrame(pca.components_, columns=dataset.columns, index=['PC1', 'PC2']).T
            loadings_df['PC1_loading'] = np.abs(loadings_df['PC1'])
            loadings_df['PC2_loading'] = np.abs(loadings_df['PC2'])
            loadings_df['subject'] = subject
            pca_df.append(loadings_df.reset_index().rename({'index': 'feature'}, axis=1))

        return pd.concat(pca_df).reset_index(drop=True)

    def test_stationarity(self) -> pd.DataFrame:
        stationarity_df = []

        for subject in self.subjects:
            dataset = self.datasets[subject]
            subject_stationarity = []

            for col in dataset.columns:
                ts = dataset[col]
                try:
                    detrend_ts = LongitudinalFeatureSelector.remove_trend(ts)

                    if (ts <= 0).any():
                        ts_log = np.log1p(ts - ts.min() + 1)
                    else:
                        ts_log = np.log1p(ts)

                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=RuntimeWarning)

                        result_ADF = adfuller(ts, maxlag=30)

                        result_KPSS = kpss(ts_log, nlags=30, regression='ct')

                    subject_stationarity.append({'feature': col,
                                                 'ADF_pvalue': result_ADF[1],
                                                 'KPSS_pvalue': result_KPSS[1]})
                except:
                    subject_stationarity.append({'feature': col,
                                                 'ADF_pvalue': None,
                                                 'KPSS_pvalue': None})

            stationarity_result_df = pd.DataFrame.from_dict(subject_stationarity)
            stationarity_result_df['ADF_stat'] = np.where(stationarity_result_df.ADF_pvalue < 0.05, 0,
                                                          1)  # 1 if non stationary
            stationarity_result_df['KPSS_stat'] = np.where(stationarity_result_df.KPSS_pvalue < 0.05, 1,
                                                           0)  # 1 if non stationary
            stationarity_result_df['subject'] = subject
            stationarity_df.append(stationarity_result_df)

        return pd.concat(stationarity_df).reset_index(drop=True)

    def stationarity_flag(self, row: pd.Series) -> str:
        if pd.isnull(row['ADF_pvalue']) or pd.isnull(row['KPSS_pvalue']):
            return None
        elif (row['ADF_stat'] == 1) and (row['KPSS_stat'] == 1):
            return 'non-stationary'
        elif (row['ADF_stat'] == 0) and (row['KPSS_stat'] == 0):
            return 'stationary'
        elif (row['ADF_stat'] == 1) and (row['KPSS_stat'] == 0):
            return 'trend-stationary'
        elif (row['ADF_stat'] == 0) and (row['KPSS_stat'] == 1):
            return 'diff-stationary'

    @staticmethod
    def explain_ts_with_fft(ts, n_modes):

        ts = ts.iloc[:150]
        rolling_ts = ts.rolling(window=3).mean().dropna().values
        x = rolling_ts.reshape(len(rolling_ts), )

        n_modes = n_modes
        dt = 1
        n = len(x)
        fhat = np.fft.fft(x, n)

        psd = fhat * np.conj(fhat) / n
        freq = (1 / (dt * n)) * np.arange(n)
        idxs_half = np.arange(1, np.floor(n / 2), dtype=np.int32)
        period = 1 / freq

        train_fft_df = pd.DataFrame(
            list(zip(psd[idxs_half], np.real(psd[idxs_half]), period[idxs_half], freq[idxs_half])),
            columns=['pds', 'pds_real', 'period [days]', 'freq [1/day]'])
        train_fft_df = train_fft_df.sort_values(by=['pds_real'], ascending=False)
        train_fft_df = train_fft_df[
            (train_fft_df['period [days]'] < len(ts) // 2) & (train_fft_df['period [days]'] > 2)]

        threshold = train_fft_df['pds_real'].values[0:n_modes]
        psd_idxs = np.isin(psd, threshold)
        psd_clean = psd * psd_idxs
        fhat_clean = psd_idxs * fhat

        signal_filtered = np.fft.ifft(fhat_clean)
        score = np.round(stats.spearmanr(signal_filtered, rolling_ts)[0], 2)

        return score, train_fft_df.iloc[:n_modes]

    def calculate_fft(self, dataset):

        df = dataset.copy()
        fft_results = []
        for col in df.columns:
            for i in range(1, 11):
                corr, res = self.explain_ts_with_fft(df[col], i)
                fft_results.append({'feature': col,
                                    'seasonal_reconstruction_score': corr,
                                    'n_modes': i,
                                    'seasonality': res['period [days]'].values.tolist()})

        return pd.DataFrame(fft_results)

    @staticmethod
    def find_seasonality_saturation(treshold, subject, explained_fft_df):

        subject_df = explained_fft_df[explained_fft_df['subject'] == subject]

        results = []
        for feature in subject_df.feature.unique():
            df = subject_df[subject_df['feature'] == feature]
            corr_df = df[df['seasonal_reconstruction_score'] >= treshold]

            if corr_df.shape[0] == 0:
                results.append({
                    'feature': feature,
                    'max_mode': 0,
                    'seasonal_reconstruction_score': 0
                })

            elif corr_df.shape[0] > 0:
                results.append({
                    'feature': feature,
                    'max_mode': corr_df.iloc[0].n_modes,
                    'seasonal_reconstruction_score': corr_df.iloc[0]['seasonal_reconstruction_score']
                })

        results_df = pd.DataFrame.from_dict(results)
        results_df['subject'] = subject

        return results_df

    @staticmethod
    def find_trend(ts: pd.Series):

        lr = LinearRegression()
        X = ts.index.values.reshape(len(ts), 1)
        lr.fit((X), ts.values)
        trend = lr.predict(X)

        feature_detrended = ts.values - trend

        return lr.coef_[0]

    @staticmethod
    def analyse_trend_in_bacteria(df, subject):

        trend_results = []
        for col in df.columns:
            trend = LongitudinalFeatureSelector.find_trend(df[col])
            trend_results.append({'feature': col,
                                  'trend': trend})

        trend_results_df = pd.DataFrame.from_dict(trend_results)
        trend_results_df['subject'] = subject

        return trend_results_df

    @staticmethod
    def get_prevalence(df):

        prevalence_df = df.copy()
        prevalence_df[prevalence_df >= 1] = 1
        prevalence_prc_df = prevalence_df.sum() / len(df)

        return prevalence_prc_df

    @staticmethod
    def analyse_autocorrelation(df, subject):
        ACF_DF = pd.DataFrame()
        for col in df.columns:
            ts = df[col]
            acf_vals, acf_ci, acf_qstat, acf_pvalues = sm.tsa.stattools.acf(ts, nlags=10, fft=False, alpha=0.05,
                                                                            qstat=True)
            acf_df = pd.DataFrame(list(zip(acf_vals, acf_pvalues)), columns=['acf', 'pval'])
            acf_df['acf_adj'] = np.where(acf_df.pval < 0.05, acf_df['acf'], 0)
            acf_df['lag'] = np.arange(0, 10)
            acf_df['feature'] = col
            acf_df['subject'] = subject
            ACF_DF = pd.concat([ACF_DF, acf_df])
        return ACF_DF


    def run_feature_selection(self):
        ACF_NOISE_DF = self.test_acf_noise()
        LjungBox_df = self.run_ljung_box_test()
        FLATNESS_DF = self.calculate_flatness()
        STATIONARITY_DF = self.test_stationarity()
        PCA_DF = self.analyse_pca_loadings()
        MEAN_DF = self.calculate_mean_std()[0]
        STD_DF = self.calculate_mean_std()[1]

        NOISE_DF = pd.merge(ACF_NOISE_DF, LjungBox_df, on=['feature', 'subject'], how='outer')
        WHITE_NOISE_DF = pd.merge(FLATNESS_DF, NOISE_DF, on=['feature', 'subject'], how='outer')
        WHITE_NOISE_DF = WHITE_NOISE_DF[
            ['feature', 'acf_noise', 'ljung_box_noise', 'flatness_score', 'subject']].copy()
        WHITE_NOISE_DF['white_noise_binary'] = WHITE_NOISE_DF.apply(LongitudinalFeatureSelector.noise_flag_df, axis=1)
        STATIONARITY_DF['stationarity'] = STATIONARITY_DF.apply(self.stationarity_flag, axis=1)

        # 1. Define stationarity type to make it stationary for seasonality analysis
        STATIONARITY_DF['stationary'] = np.where(STATIONARITY_DF['stationarity'] == 'stationary', 1, 0)
        STATIONARITY_DF['non-stationary'] = np.where(STATIONARITY_DF['stationarity'] == 'non-stationary', 1, 0)
        STATIONARITY_DF['trend-stationary'] = np.where(STATIONARITY_DF['stationarity'] == 'trend-stationary', 1, 0)
        STATIONARITY_DF['diff-stationary'] = np.where(STATIONARITY_DF['stationarity'] == 'diff-stationary', 1, 0)

        STATIONARY_TS = []
        for i in range(len(self.subjects)):
            type1 = STATIONARITY_DF[(STATIONARITY_DF['stationarity'] == 'trend-stationary') & (
                    STATIONARITY_DF['subject'] == self.subjects[i])].feature.values
            type2 = STATIONARITY_DF[(STATIONARITY_DF['stationarity'] == 'diff-stationary') & (
                    STATIONARITY_DF['subject'] == self.subjects[i])].feature.values
            type3 = STATIONARITY_DF[(STATIONARITY_DF['stationarity'] == 'non-stationary') & (
                    STATIONARITY_DF['subject'] == self.subjects[i])].feature.values

            data = self.datasets[self.subjects[i]].copy()
            data[type1] = data[type1].apply(lambda x: LongitudinalFeatureSelector.remove_trend(x))
            data[type2] = data[type2].apply(lambda x: x.diff())
            data[type3] = data[type3].apply(lambda x: x.diff())
            data['subject'] = self.subjects[i]

            STATIONARY_TS.append(data)

        # 2. Find how many modes is enough to explain feature
        explained_fft_df = pd.DataFrame()
        for i in range(len(self.subjects)):
            subject = STATIONARY_TS[i].iloc[:, -1].values[0]
            dataset = STATIONARY_TS[i].iloc[:, :-1]
            df = self.calculate_fft(dataset)
            df['subject'] = subject
            explained_fft_df = pd.concat([explained_fft_df, df])

        explained_fft_df = explained_fft_df.reset_index(drop=True)

        # 3. Define seasonal bacteria. We define that an ASV is seasonal if it's seasonal reconstruction score for modes is at least 0.5
        modes_saturation = explained_fft_df[explained_fft_df['n_modes'] == 6]
        modes_saturation['seasonal'] = np.where(modes_saturation['seasonal_reconstruction_score'] > 0.5, 1, 0)
        SEASONAL_BACTERIA_DF = modes_saturation[['feature', 'subject', 'seasonal']]

        # 4. Find how many modes is necesarry to get seasonal recontruction score of 0.5
        SEASONALITY_SATURATION_DF = pd.DataFrame()
        for subject in self.subjects:
            df = self.find_seasonality_saturation(0.4, subject, explained_fft_df)
            SEASONALITY_SATURATION_DF = pd.concat([SEASONALITY_SATURATION_DF, df])

        SEASONALITY_SATURATION_DF.columns = ['feature', 'max_fft_mode', 'max_fft_mode_corr', 'subject']

        # 5. Find dominant mode
        ONE_MODE_DF = explained_fft_df[explained_fft_df['n_modes'] == 1]
        ONE_MODE_DF['seasonality'] = [int(i[0]) for i in ONE_MODE_DF['seasonality'].values]
        ONE_MODE_DF['s1_adj'] = np.where(ONE_MODE_DF['seasonal_reconstruction_score'] >= 0.4,
                                         ONE_MODE_DF['seasonality'], 0)
        ONE_MODE_DF.columns = ['feature', '1st_mode_spearman_corr', 'n', '1st_mode_seasonality', 'subject',
                               '1st_mode_adj']
        # TREND
        TREND_DF = pd.DataFrame()
        for subject in self.subjects:
            df = self.analyse_trend_in_bacteria(self.datasets[subject], subject)
            TREND_DF = pd.concat([TREND_DF, df])

        TREND_DF = TREND_DF.reset_index(drop=True)


        # PREVALENCE
        PREVALENCE_DF = pd.DataFrame()
        for subject in self.subjects:
            prevalence_df = self.get_prevalence(self.datasets[subject])
            prevalence_df = prevalence_df.reset_index().rename({'index': 'feature', 0: 'prc_occurence'}, axis=1)
            prevalence_df['subject'] = subject
            PREVALENCE_DF = pd.concat([PREVALENCE_DF, prevalence_df])

        # AUTOCORRELATION
        AUTOCORR_DF = pd.DataFrame()
        for subject in self.subjects:
            corr_df = self.analyse_autocorrelation(self.datasets[subject], subject)
            AUTOCORR_DF = pd.concat([AUTOCORR_DF, corr_df])
        AUTOCORR_DF = AUTOCORR_DF[AUTOCORR_DF['lag'] != 0]


        # MERGE FEATURES INTO A CHARACTERICTIS TABLE
        # rename certain columns
        PCA_DF = PCA_DF[['feature', 'PC1_loading', 'PC2_loading', 'subject']]
        SEASONALITY_SATURATION_DF.columns = ['feature', 'n_modes', 'seasonal_reconstruction_score', 'subject']
        ONE_MODE_DF.columns = ['feature', 'dominant_mode_score', 'n', 'dominant_seasonality', 'subject',
                               'dominant_seasonality_adj']
        ONE_MODE_DF = ONE_MODE_DF[
            ['feature', 'dominant_seasonality', 'dominant_seasonality_adj', 'dominant_mode_score', 'subject']]
        PREVALENCE_DF = PREVALENCE_DF.rename({'prc_occurence': 'prevalence'}, axis=1)
        AUTOCORR_DF.columns = ['autocorrelation', 'pvalue', 'autocorrelation_sig', 'lag', 'feature', 'subject']
        AUTOCORR_DF = AUTOCORR_DF[['feature', 'lag', 'autocorrelation', 'pvalue', 'autocorrelation_sig', 'subject']]

        # set index to merge later
        MEAN_DF.set_index(['feature', 'subject'], inplace=True)
        STD_DF.set_index(['feature', 'subject'], inplace=True)
        WHITE_NOISE_DF.set_index(['feature', 'subject'], inplace=True)
        PCA_DF.set_index(['feature', 'subject'], inplace=True)
        STATIONARITY_DF.set_index(['feature', 'subject'], inplace=True)
        SEASONALITY_SATURATION_DF.set_index(['feature', 'subject'], inplace=True)
        ONE_MODE_DF.set_index(['feature', 'subject'], inplace=True)
        TREND_DF.set_index(['feature', 'subject'], inplace=True)
        PREVALENCE_DF.set_index(['feature', 'subject'], inplace=True)
        AUTOCORR_DF = AUTOCORR_DF.pivot(index=['feature', 'subject'], columns='lag', values='autocorrelation_sig')
        AUTOCORR_DF.columns = [f'lag_{i}_corr' for i in range(1, 10)]
        SEASONAL_BACTERIA_DF.set_index(['feature', 'subject'], inplace=True)

        # merge
        LONGITUDINAL_CHARACTERISTICS_DF = pd.concat([MEAN_DF, STD_DF, WHITE_NOISE_DF, PCA_DF, STATIONARITY_DF,
                                                     SEASONALITY_SATURATION_DF, ONE_MODE_DF, SEASONAL_BACTERIA_DF,
                                                     TREND_DF, PREVALENCE_DF,
                                                     AUTOCORR_DF], axis=1).reset_index()

        # merge all non stationary variables to one
        LONGITUDINAL_CHARACTERISTICS_DF['non_stationary'] = np.where(
            (LONGITUDINAL_CHARACTERISTICS_DF['non-stationary'] == 1) |
            (LONGITUDINAL_CHARACTERISTICS_DF['trend-stationary'] == 1) |
            (LONGITUDINAL_CHARACTERISTICS_DF['diff-stationary'] == 1),
            1, 0)

        LONGITUDINAL_CHARACTERISTICS_DF = np.round(LONGITUDINAL_CHARACTERISTICS_DF, 3)
        return LONGITUDINAL_CHARACTERISTICS_DF
        #LONGITUDINAL_CHARACTERISTICS_DF.to_csv('./data/ts_charactericstics_tables/LONGITUDINAL_CHARACTERISTICS_DF.csv', index=False)



