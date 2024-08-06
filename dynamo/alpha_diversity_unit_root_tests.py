import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
from sklearn.linear_model import LinearRegression
from statsmodels.stats.diagnostic import acorr_ljungbox


class AlphaDiversityUnitRootTests:
    @staticmethod
    def remove_trend(ts):
        lr = LinearRegression()
        X = ts.index.values.reshape(len(ts), 1)
        lr.fit(X, ts.values)
        trend = lr.predict(X)

        feature_detrended = ts.values - trend

        return feature_detrended

    @staticmethod
    def autocorrelation_presence(ts):
        detrended_ts = AlphaDiversityUnitRootTests.remove_trend(ts)

        # Ljung-Box test for white noise
        ljung_box_results = acorr_ljungbox(detrended_ts, lags=30)
        ljung_box_results_df = ljung_box_results.reset_index()

        if ljung_box_results_df[ljung_box_results_df['lb_pvalue'] > 0.05].shape[0] == 0:
            print('series is autocorrelated')
        elif ljung_box_results_df[ljung_box_results_df['lb_pvalue'] < 0.05].shape[0] == 0:
            print('series is not autocorrelated')

    @staticmethod
    def test_unit_root(ts, subject):
        detrend_ts = AlphaDiversityUnitRootTests.remove_trend(ts)

        result_ADF = adfuller(ts, maxlag=30)
        result_KPSS = kpss(np.log(ts), nlags=30)

        unit_root_df = pd.DataFrame([result_ADF[1], result_KPSS[1]], columns=['pvalue'])
        unit_root_df['test'] = ['ADF', 'KPSS']
        unit_root_df['pvalue'] = np.round(unit_root_df['pvalue'], 3)
        unit_root_df['subject'] = subject

        return unit_root_df