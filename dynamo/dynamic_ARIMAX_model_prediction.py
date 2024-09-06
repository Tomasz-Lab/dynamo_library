import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import SimpleExpSmoothing

from scipy import stats
from scipy import fft
from scipy import signal as sig
from scipy.fft import fft, fftfreq, ifft, rfft, rfftfreq
from cmath import phase
import math


class DynamicARIMAXModelPrediction:

    @staticmethod
    def smooth_data(X: np.ndarray, smoothing_level: float = 0.7) -> np.ndarray:

        '''
        smooth input data with SimpleExpSmoothing
        '''

        model = SimpleExpSmoothing(np.asarray(X), initialization_method='estimated')
        fit = model.fit(smoothing_level=smoothing_level)
        pred = fit.predict(1, X.shape[0])
        return pred

    @staticmethod
    def detrend_ts(ts: np.ndarray) -> np.ndarray:

        '''
        remove trend from variable by first fitting a linear regression model to the data
        to estimate trend and, removing it from the variable
        '''

        X = np.arange(0, len(ts))
        X = X.reshape(len(X), 1)
        y = ts
        model = LinearRegression()
        model.fit(X, y)

        prediction = model.predict(X)
        residuals = y - prediction

        return residuals

    @staticmethod
    def fft_decomposition(signal: np.ndarray) -> pd.DataFrame:

        '''
        analyse sesonal component in data
        '''

        signal = DynamicARIMAXModelPrediction.detrend_ts(signal)
        fft_output = fft(signal)
        amplitude = np.abs(fft_output)
        freq = fftfreq(len(signal), 1)

        mask = freq >= 0
        freq = freq[mask]
        amplitude = amplitude[mask]

        peaks = sig.find_peaks(amplitude[freq >= 0].reshape(len(amplitude), ))[0]
        peak_freq = freq[peaks]
        peak_amplitude = amplitude[peaks]

        # create dataframe with results from FFT
        output = pd.DataFrame()
        output['index'] = peaks
        output['fft'] = fft_output[peaks]
        output['freq (1/day)'] = peak_freq
        output['period [days]'] = 1 / output['freq (1/day)']
        output['phase'] = output.fft.apply(lambda z: phase(z))
        output['amplitude'] = peak_amplitude

        return output

    @staticmethod
    def create_fourier_dict(fft_output: pd.DataFrame, data: np.ndarray, n_modes: int) -> dict:

        output = fft_output[(fft_output['period [days]'] < len(data) // 2)]
        output = output.sort_values('amplitude', ascending=False)
        output['label'] = list(map(lambda n: 'FT_{}'.format(n), range(1, len(output) + 1)))

        # get n modes
        output = output.head(n_modes)
        output = output.set_index('label')
        fourier_terms_dict = output.to_dict('index')

        return fourier_terms_dict

    @staticmethod
    def create_fourier_df(fourier_terms_dict: dict, size: int) -> pd.DataFrame:

        '''
        create a seasonal wave with n Fourier dominant modes
        '''

        data = pd.DataFrame(np.arange(0, size), columns=['time'])

        for key in fourier_terms_dict.keys():
            a = fourier_terms_dict[key]['amplitude']
            w = 2 * math.pi * (fourier_terms_dict[key]['freq (1/day)'])
            p = fourier_terms_dict[key]['phase']
            data[key] = data['time'].apply(lambda t: a * math.cos(w * t + p))

        data['FT_All'] = 0
        for column in list(fourier_terms_dict.keys()):
            data['FT_All'] = data['FT_All'] + data[column]

        return data

    @staticmethod
    def arimax_model(train: np.ndarray, n_modes: int, p: int, q: int, train_fold_size: int, test_fold_size: int,
                     trend: bool = False):

        f1 = DynamicARIMAXModelPrediction.fft_decomposition(train)
        f2 = DynamicARIMAXModelPrediction.create_fourier_dict(f1, n_modes)

        exog_train = DynamicARIMAXModelPrediction.create_fourier_df(f2, train_fold_size)
        exog_test = DynamicARIMAXModelPrediction.create_fourier_df(f2, test_fold_size)

        if trend == False:

            arima_model = ARIMA((train), order=(p, 0, q), exog=exog_train[['FT_All']])
            arima_model_fit = arima_model.fit(method_kwargs={"warn_convergence": False})

            prediction = arima_model_fit.predict(start=train_fold_size, end=((train_fold_size + test_fold_size) - 1),
                                                 exog=exog_test[['FT_All']], dynamic=True)

        elif trend == True:
            arima_model = ARIMA((train), order=(p, 0, q), exog=exog_train[['time', 'FT_All']])
            arima_model_fit = arima_model.fit(method_kwargs={"warn_convergence": False})

            prediction = arima_model_fit.predict(start=train_fold_size, end=((train_fold_size + test_fold_size) - 1),
                                                 exog=exog_test[['time', 'FT_All']], dynamic=True)

        return prediction, f2

    @staticmethod
    def dynamic_model_predict(df: pd.Series, **model_param_dict) -> tuple:
        data = df.astype(float)

        n_modes = model_param_dict['n_modes']
        p = model_param_dict['p']
        d = model_param_dict['d']
        q = model_param_dict['q']

        # split into train and test
        train = data[:80]
        test = data[80:]
        train_fold_size = len(train)
        test_fold_size = len(test)

        # smooth data
        train_smoothed = DynamicARIMAXModelPrediction.smooth_data(train)
        test_smoothed = DynamicARIMAXModelPrediction.smooth_data(test)

        # create Fourier seasonal wave
        seasonal_decomposition_res = DynamicARIMAXModelPrediction.fft_decomposition(train_smoothed)
        seasonal_variable = DynamicARIMAXModelPrediction.create_fourier_dict(seasonal_decomposition_res, train_smoothed,
                                                                             n_modes)
        exog_train = DynamicARIMAXModelPrediction.create_fourier_df(seasonal_variable, len(train_smoothed))
        exog_test = DynamicARIMAXModelPrediction.create_fourier_df(seasonal_variable, len(test_smoothed))

        # smooth y true
        y_true_smoothed = DynamicARIMAXModelPrediction.smooth_data(data.iloc[1:])

        # fit ARIMA model
        arima_model = ARIMA((train_smoothed), order=(p, d, q), exog=(exog_train['FT_All']))
        arima_model_fit = arima_model.fit(method_kwargs={"warn_convergence": False})
        prediction = arima_model_fit.predict(start=0, end=(len(data) - 1), exog=(exog_test['FT_All']), dynamic=True)

        return y_true_smoothed, prediction

    @staticmethod
    def plot_prediction(y_true: np.ndarray, y_pred: np.ndarray, subject: str):
        plt.figure(figsize=(12, 4))
        plt.plot(y_true, linestyle='-.', color='k', lw=1, label='ytrue')
        plt.plot(y_true, 'o', color='k', markersize=4)

        plt.plot(y_pred.values, color='#EA1313', lw=1.2, label='ypred', linestyle='-')
        plt.plot(y_pred.values, 'o', color='#EA1313', markersize=4)

        plt.legend(fancybox=True, ncol=2, edgecolor='k', fontsize=12)

        plt.xlabel('time point [days]', fontsize=14)
        plt.ylabel('Shannon diversity Index', fontsize=14)

        plt.grid(axis='y', linestyle=':', color='lightgrey')

        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.tick_params(axis='both', which='minor', labelsize=12)

        if subject == 'female':
            plt.xlim([0, 130])
        else:
            plt.xlim([0, 200])
        plt.ylim([y_true.min() - 0.05, y_true.max() + 0.05])

        plt.fill_between(x=np.arange(80), y1=y_true.min(), y2=y_true.max(), color='lightgrey', alpha=0.3)
        plt.fill_between(x=np.arange(80, len(y_true)), y1=y_true.min(), y2=y_true.max(), color='#ffca3a', alpha=0.2)

        if subject == 'donorA':
            plt.fill_between(y_pred.index.values[100:122], y1=y_true.min() - 0.1, y2=y_true.max() + 0.05, alpha=.1,
                             color='red')
        elif subject == 'donorB':
            plt.fill_between(y_pred.index.values[150:159], y1=y_true.min() - 0.1, y2=y_true.max() + 0.05, alpha=.1,
                             color='red')
        else:
            pass

        plt.axvline(80, color='k', linestyle='-.', lw=2)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def get_periods(data: np.ndarray, n: int, subject: str) -> pd.DataFrame:
        '''
        get periods of used seasonalities
        '''

        train_smoothed = DynamicARIMAXModelPrediction.smooth_data(data, 0.5)
        f1 = DynamicARIMAXModelPrediction.fft_decomposition(train_smoothed)
        period = DynamicARIMAXModelPrediction.create_fourier_dict(f1, train_smoothed, n)
        periods_df = pd.DataFrame.from_dict(period).T['period [days]'].astype(float).reset_index()
        periods_df['subject'] = subject

        return periods_df

    @staticmethod
    def get_prediction_scores(y_true: np.ndarray, y_pred: np.ndarray, subject: str) -> pd.DataFrame:
        results = []
        for n in range(0, len(y_pred) - 20, 3):
            try:

                ytrue = y_true[n: (20 + n)]
                ypred = y_pred[n: (20 + n)]

                emd = np.round(stats.wasserstein_distance(ytrue, ypred), 2)
                mape = np.round(mean_absolute_percentage_error(ytrue, ypred), 2)

                results.append({'emd': emd,
                                'mape': mape})
            except:
                pass

        results_df = pd.DataFrame.from_dict(results)
        results_df['subject'] = subject

        return results_df



















