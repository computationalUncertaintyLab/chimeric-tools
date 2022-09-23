from datetime import date, timedelta
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import stan
from scipy.stats import norm
from scipy import signal
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.linear_model import Ridge
from statsmodels.tsa.ar_model import AutoReg
from prophet import Prophet
import statsmodels.api as sm
import itertools
from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics



def train_simulated_data(data: pd.DataFrame, models: Union[list, str], target: str):
    """
    From simulated data from the chimeric_tools.Simulation.COVID.simulate() function, you can train our models and output a dataframe with the predictions for each simulation
    and at each time point.

    Parameters
    -----------
    data: pd.DataFrame
        This has to be a data frame that that has been returned from the chimeric_tools.Simulation.COVID.simulate() function.
    models: Union[list, str]
        This is a list of the models you want to use when forecasting the data. If left blank the default is to use all models in this file.
    target: str
        The name of the paremeter you want to plot: cases, deaths, hosps.
    
    Returns
    --------
    data: pd.DataFrame
        This is a data frame with the predictions for each model and time point. The columns are:
            `forcast_date` is the data the forecast was made from/
            `target_end_date` is the date the forecast is for/
            `location` is the location the forecast is for/
            `sim` is the simulation number (used to help differentiate each series)/
            `value` is the prediction for the percentile.
            `quantile` is the percentile that the predictions represents/

    """
    import chimeric_tools.models

    df = pd.DataFrame(
        columns=[
            "forecast_date",
            "target_end_date",
            "location",
            "sim",
            "model",
            "value",
        ]
    )
    for sim in data["sim"].unique():
        sub_data = data.loc[data["sim"] == sim]
        for model_name in models:
            class_ = getattr(chimeric_tools.models, model_name)
            for i in range(30, len(sub_data) + 1):
                splice_data = sub_data[:i]
                model = class_(
                    data=splice_data,
                    target=target,
                    N_tilde=4,
                    location=splice_data["location"].iloc[0],
                    date=max(splice_data["date"]),
                )
                model.fit()
                preds = model.predict()
                preds["model"] = model_name
                preds["sim"] = sim
                df = pd.concat([df, preds])
    return df.sort_values(
        by=["sim", "model", "target_end_date", "forecast_date", "quantile"]
    )




def _format_sample(data: np.ndarray, start_date: date, location: str):
    """
    Formats the samples from a stan model into a useble dataframe all relevent information.

    Parameters
    -----------
    data: np.ndarray
        This is the data from the stan model.
    start_date: date
        This is the date that the forecasts are being made from
    location: str
        This is the location the forecasts are for.
    
    Returns
    --------
    data: pd.DataFrame
        A dataframe of all the samples with all relevent information. The columns are:
            `forcast_date` is the date the forecast was made from/
            `target_end_date` is the date the forecast is for/
            `location` is the location the forecast is for/
            `sample` is the sample number/
            `value` is the prediction for that sample
    """
    data_predictions = {
        "forecast_date": [],
        "target_end_date": [],
        "location": [],
        "sample": [],
        "value": [],
    }

    for N_tilde, samples in enumerate(data):
        for n, sample in enumerate(samples):
            data_predictions["forecast_date"].append(start_date)
            data_predictions["target_end_date"].append(
                start_date + timedelta(weeks=N_tilde + 1)
            )
            data_predictions["location"].append(location)
            data_predictions["sample"].append(n)
            data_predictions["value"].append(sample)
    return pd.DataFrame(data_predictions)

def _format_quantiles(data: pd.DataFrame):
    """
    Formats the samples form the stan model (precessed througth the format_sample function) into a percentiles.

    Parameters
    -----------
    data: pd.DataFrame
        This is the dataframe that is returned from the format_sample function.
    
    Returns
    --------
    data: pd.DataFrame
        A dataframe of all the predictions and their percentiles. The columns are:
            `forcast_date` is the date the forecast was made from/
            `target_end_date` is the date the forecast is for/
            `location` is the location the forecast is for/
            `quantile` is the percentile that the prediction represents/
            `value` is the prediction for that percentile
    """
    def createQuantiles(x):
        quantiles = np.array(
            [
                0.010,
                0.025,
                0.050,
                0.100,
                0.150,
                0.200,
                0.250,
                0.300,
                0.350,
                0.400,
                0.450,
                0.500,
                0.550,
                0.600,
                0.650,
                0.700,
                0.750,
                0.800,
                0.850,
                0.900,
                0.950,
                0.975,
                0.990,
            ]
        )
        quantileValues = np.percentile(x["value"], q=100 * quantiles)
        return pd.DataFrame(
            {"quantile": list(quantiles), "value": list(quantileValues)}
        )

    dataQuantiles = (
        data.groupby(["forecast_date", "target_end_date", "location"])
        .apply(lambda x: createQuantiles(x))
        .reset_index()
        .drop(columns="level_3")
    )

    return dataQuantiles


def quntiles(preds, sigma, N_tilde: int, location: str, date: date):
    """
    Converts mean predictions and variance predictions into invervals. We assume a normal distribution.

    Parameters
    -----------
    preds: np.ndarray
        This is the mean predictions.
    sigma: np.ndarray
        This is the variance predictions.
    N_tilde: int
        This is the number of weeks ahead to forecast.
    location: str
        This is the location the forecast is for.
    date: date
        This is the date the forecasts are made from.
    
    Returns
    --------
    data: pd.DataFrame
        A dataframe of all the predictions and their percentiles. The columns are:
            `forcast_date` is the date the forecast was made from/
            `target_end_date` is the date the forecast is for/
            `location` is the location the forecast is for/
            `quantile` is the percentile that the prediction represents/
            `value` is the prediction for that percentile
    """
    data_predictions = {
        "forecast_date": [],
        "target_end_date": [],
        "location": [],
        "quantile": [],
        "value": [],
    }
    quantiles = np.array(
        [
            0.010,
            0.025,
            0.050,
            0.100,
            0.150,
            0.200,
            0.250,
            0.300,
            0.350,
            0.400,
            0.450,
            0.500,
            0.550,
            0.600,
            0.650,
            0.700,
            0.750,
            0.800,
            0.850,
            0.900,
            0.950,
            0.975,
            0.990,
        ]
    )
    for quantile in quantiles:
        q = norm.ppf(quantile)
        values = preds + q * sigma
        data_predictions["forecast_date"].extend(N_tilde * [date])
        data_predictions["target_end_date"].extend(
            [date + timedelta(weeks=i) for i in range(1, N_tilde + 1)]
        )
        data_predictions["location"].extend(N_tilde * [location])
        data_predictions["quantile"].extend(N_tilde * [quantile])
        data_predictions["value"].extend(values)
    return pd.DataFrame(data_predictions)


def ar2ma(ar, ma, lags = 100):
    """
    Lowkey I have no idea what this does

    Parameters
    -----------
    ar: np.ndarray
        This is the autoregressive coefficients.
    ma: np.ndarray
        This is the moving average coefficients.
    lags: int
        This is the number of lags to use.
    
    Returns
    --------
    data: np.ndarray
        No idea
    """
    impulse = np.zeros(lags)
    impulse[0] = 1.0
    return signal.lfilter(ma, ar, impulse)


def model(data: pd.DataFrame):
    """
    Adds predictions and residuals to the data
    """
    for fip in data["location"].unique():
        sub_data = data.loc[data["location"] == fip]
        m = ARIMA(sub_data)
        data.loc[data["location"] == fip, "preds"] = m.preds
        data.loc[data["location"] == fip, "residuals"] = m.residuals
    return data

def plot_single_predictions(data: pd.DataFrame, preds: pd.DataFrame, target: str):
    """
    Plot the predictions and the 95% confidence interval for a single prediction.
    
    Parameters
    -----------
    data: pd.DataFrame
        The original dataframe of data from `get_data` function.
    preds: pd.DataFrame
        This is a dataframe that is in the same format as a dataframe from `format_quantiles`
    to_plot: str
        The name of the paremeter you want to plot: cases, deaths, hosps.
    
    Returns
    --------
    A plt plot
    """

    dates = preds["target_end_date"].unique()
    low = preds.loc[preds["quantile"] == 0.025, "value"]
    mid = preds.loc[preds["quantile"] == 0.500, "value"]
    high = preds.loc[preds["quantile"] == 0.975, "value"]

    # plt.figure(figsize=(10, 6), dpi=150)
    plt.plot(data["date"], data[target], label="Truth Data")
    plt.plot(dates, mid, label="Predictions", color="black")
    plt.fill_between(
        dates, low, high, color="red", alpha=0.5, label="95% Confidence Interval"
    )
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel(target)
    plt.show()


class ARIMA:
    """
    Just a quick and dirty test model
    """

    def __init__(self, data: pd.DataFrame):
        self.model_name = "ARIMA"
        self.data = np.array(data["value"])
        self.fit = self.fit()
        self.preds = self.predict()
        self.residuals = self.residuals()

    def fit(self):
        return sm.tsa.statespace.SARIMAX(self.data, order=(2, 1, 0)).fit(disp=0)

    def predict(self):
        return self.fit.predict()

    def residuals(self):
        return self.fit.resid


class AR1:
    """
    LinReg model with Stan
    """

    def __init__(self, data: np.ndarray, N_tilde: int, location: str, date: date):
        self.model_name = "StanLinReg"
        self.data = data
        self.location = location
        self.date = date
        self.N = len(self.data)
        self.N_tilde = N_tilde

    def fit(self):
        stan_code = """
            data {
                int<lower=0> N;
                vector[N] y;
                int<lower=0> N_tilde;
            }
            parameters {
                real alpha;
                real beta;
                real<lower=0> sigma;
            }
            model {
                alpha ~ normal(0, 10);
                beta ~ normal(0, 10);
                sigma ~ cauchy(0, 2.5);
                y[2:N] ~ normal(alpha + beta * y[1:(N-1)], sigma);
            }
            generated quantities {
                vector[N_tilde] y_tilde;
                real i = y[N];
                for (n in 1:N_tilde) {
                    y_tilde[n] = normal_rng(alpha + beta * i, sigma);
                    i = y_tilde[n];
                    }
            }
        """

        data = {
            "N": self.N,
            "y": self.data,
            "N_tilde": self.N_tilde,
        }
        model = stan.build(program_code=stan_code, data=data)
        self.fit = model.sample(num_chains=1, num_samples=1 * 10**3)

    def predict(self):
        predictions = self.fit["y_tilde"]  # this is coming from the model object
        predictions = _format_sample(predictions, self.date, self.location)
        predictions = _format_quantiles(predictions)
        return predictions


class AR3:
    """
    Statsmodel AR(3)
    """

    def __init__(
        self, data: np.ndarray, target: str, N_tilde: int, location: str, date: date
    ):
        self.model_name = "StatsModelAR6"
        data.set_index("date", inplace=True)
        self.data = data[target]
        self.location = location
        self.date = date
        self.N = len(data)
        self.N_tilde = N_tilde

    def fit(self):
        model = AutoReg(self.data, lags=3)
        self.fit = model.fit()

    def predict(self):
        predictions = self.fit.get_prediction(
            start=self.N, end=self.N + self.N_tilde - 1
        )
        predictions = quntiles(
            predictions.predicted_mean, predictions.se_mean, self.N_tilde, self.location, self.date
        )
        return predictions


class AR6:
    """
    Statsmodel AR(6)
    """

    def __init__(
        self, data: np.ndarray, target: str, N_tilde: int, location: str, date: date
    ):
        self.model_name = "StatsModelAR6"
        data.set_index("date", inplace=True)
        self.data = data[target]
        self.location = location
        self.date = date
        self.N = len(data)
        self.N_tilde = N_tilde

    def fit(self):
        model = AutoReg(self.data, lags=6)
        self.fit = model.fit()

    def predict(self):
        predictions = self.fit.get_prediction(
            start=self.N, end=self.N + self.N_tilde - 1
        )
        predictions = quntiles(
            predictions.predicted_mean, predictions.se_mean, self.N_tilde, self.location, self.date
        )
        return predictions


class ridge:
    def __init__(self, data: np.ndarray, N_tilde: int, location: str, target: Union[str, list], date: date):
        self.model_name = "RidgeRegression"
        # data.set_index("date", inplace=True)
        self.data = data[target]
        self.N = self.data.shape[0]
        self.location = location
        self.date = date
        self.N_tilde = N_tilde
        self.target = target

    def fit(self):
        forecaster = ForecasterAutoreg(
                    # regressor = RandomForestRegressor(random_state=0),
                    regressor = Ridge(),
                    lags      = 15
                )
        forecaster.fit(y=self.data)

        self.fit = forecaster

    def sigma2(self):
        resid = self.fit.in_sample_residuals
        sumofsq = np.sum(resid**2)
        return 1.0 / self.N * sumofsq

    def predict(self):
        y_pred = self.fit.predict(self.N_tilde)
        ar_params = np.array([1])
        ar_params = np.append(ar_params, self.fit.get_coef()["coef"].to_numpy().T * -1)
        print(ar_params)
        ma = ar2ma(ar_params, np.ones(1), lags = 15)
        print(ma)
        se_mean = np.sqrt(self.sigma2() * np.cumsum(ma ** 2))[:self.N_tilde]
        return quntiles(y_pred, se_mean, self.N_tilde, self.location, self.date)


class prophet:
    """
    Use the package prophet from Facebook to predict the time series, see https://facebook.github.io/prophet/ for more details.
    
    Parameters
    -----------
    data: pd.DataFrame
        The original dataframe of data from `get_data` function.
    target: str
        The target variable to be predicted such as "cases" or "deaths".
    N_tilde: int
        The number of days or weeks to be predicted.
    location: str
        The location to be predicted.
    date: date
        The date of the last data point.
    Returns
    --------
    A model object.
    """
    def __init__(self, data: pd.DataFrame , target: str, N_tilde: int, location: str, date: date):
        self.model_name = "Prophet"
        self.data = data.loc[:, (target, "date")]
        self.data.rename({target: "y", "date": "ds"}, axis=1 ,inplace=True)
        self.N = self.data.shape[0]
        self.location = location
        self.date = date
        self.N_tilde = N_tilde
        self.target = target
        self.best_params = self.get_best_params()
        
    def get_best_params(self):
        import itertools
        import numpy as np
        import pandas as pd
        from prophet import Prophet
    
        from prophet.diagnostics import cross_validation
        from prophet.diagnostics import performance_metrics

        param_grid = {  
            'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
            'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
            'holidays_prior_scale': [0.01, 0.1, 1.0, 10.0],
                }
        # three cutoffs six months apart
        cutoffs = pd.to_datetime(['2020-08-15','2021-02-15', '2021-08-15', '2022-02-15'])
        # Generate all combinations of parameters
        all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
        rmses = []  # Store the RMSEs for each params here

        # Use cross validation to evaluate all parameters
        for params in all_params:
            m = Prophet(**params)  # Fit model with given params
            m.add_country_holidays(country_name='US')
            m = m.fit(self.data)
            df_cv = cross_validation(m, horizon='30 days', parallel="processes")
            # rolling_window specifies the proportion of forecasts to use in each rolling window
            df_p = performance_metrics(df_cv, rolling_window=1)
            rmses.append(df_p['rmse'].values[0])
        best_params = all_params[np.argmin(rmses)]
        return best_params
    
    def fit(self):
        self.model = Prophet(**self.best_params)
        self.model.add_country_holidays(country_name='US')      
        self.model.fit(self.data)

    def predict(self):
        future = self.model.make_future_dataframe(periods=self.N_tilde, freq='W')
        predictions = self.model.predictive_samples(future)["yhat"][-self.N_tilde:]
        predictions = _format_sample(predictions, self.date, self.location)
        predictions = _format_quantiles(predictions)
        return predictions


class xgboost:
    """
    Use the xgboostRegressor model from package xgboost to predict the time series, see https://xgboost.readthedocs.io/en/stable/index.html for more details.
    
    Parameters
    -----------
    data: pd.DataFrame
        The original dataframe of data from `get_data` function.
    target: str
        The target variable to be predicted such as "cases" or "deaths".
    N_tilde: int
        The number of days or weeks to be predicted.
    location: str
        The location to be predicted.
    date: date
        The date of the last data point.
    Returns
    --------
    A model object.
    """
    def __init__(self, data: np.ndarray, N_tilde: int, location: str, target: Union[str, list], date: date):
        self.model_name = "XgboostRegression"
        # data.set_index("date", inplace=True)
        self.data = data[target]
        self.N = self.data.shape[0]
        self.location = location
        self.date = date
        self.N_tilde = N_tilde
        self.target = target

    def fit(self):
        forecaster = ForecasterAutoreg(
                        regressor = XGBRegressor(random_state=123,n_jobs=4,objective='reg:squarederror', booster='gblinear'),
                        lags = 15
                        )
        forecaster.fit(y=self.data)
        self.fit = forecaster

    def sigma2(self):
        resid = self.fit.in_sample_residuals
        sumofsq = np.sum(resid**2)
        return 1.0 / self.N * sumofsq

    def predict(self):
        print(self.fit)
        y_pred = self.fit.predict(self.N_tilde)
        ar_params = np.array([1])
        ar_params = np.append(ar_params, self.fit.get_coef()["coef"].T * -1)
        # print(ar_params)
        ma = ar2ma(ar_params, np.ones(1), lags = 15)
        # print(ma)
        se_mean = np.sqrt(self.sigma2() * np.cumsum(ma ** 2))[:self.N_tilde]
        return quntiles(y_pred, se_mean, self.N_tilde, self.location, self.date)

class lightgbm:
    """
    Use the LGBMRegressor model from package lightgbm to predict the time series, see https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html for more details.
    
    Parameters
    -----------
    data: pd.DataFrame
        The original dataframe of data from `get_data` function.
    target: str
        The target variable to be predicted such as "cases" or "deaths".
    N_tilde: int
        The number of days or weeks to be predicted.
    location: str
        The location to be predicted.
    date: date
        The date of the last data point.
    Returns
    --------
    A model object.
    """
    def __init__(self, data: np.ndarray, N_tilde: int, location: str, target: Union[str, list], date: date):
        self.model_name = "LightgbmRegression"
        # data.set_index("date", inplace=True)
        self.data = data[target]
        self.N = self.data.shape[0]
        self.location = location
        self.date = date
        self.N_tilde = N_tilde
        self.target = target

    def fit(self):
        forecaster = ForecasterAutoreg(
                        regressor = LGBMRegressor(random_state=123,n_jobs = 4),
                        lags = 24
                        )
        forecaster.fit(y=self.data)

        self.fit = forecaster

    def sigma2(self):
        resid = self.fit.in_sample_residuals
        sumofsq = np.sum(resid**2)
        return 1.0 / self.N * sumofsq

    def predict(self):
        y_pred = self.fit.predict(self.N_tilde)
        ar_params = np.array([1])
        ar_params = np.append(ar_params, self.fit.get_coef()["coef"].to_numpy().T * -1)
        # print(ar_params)
        ma = ar2ma(ar_params, np.ones(1), lags = 15)
        # print(ma)
        se_mean = np.sqrt(self.sigma2() * np.cumsum(ma ** 2))[:self.N_tilde]
        return quntiles(y_pred, se_mean, self.N_tilde, self.location, self.date)

class catboost:
    """
    Use the CatboostRegressor model from package catboost to predict the time series, see https://catboost.ai/en/docs/concepts/installation for more details.
    
    Parameters
    -----------
    data: pd.DataFrame
        The original dataframe of data from `get_data` function.
    target: str
        The target variable to be predicted such as "cases" or "deaths".
    N_tilde: int
        The number of days or weeks to be predicted.
    location: str
        The location to be predicted.
    date: date
        The date of the last data point.
    Returns
    --------
    A model object.
    """
    def __init__(self, data: np.ndarray, N_tilde: int, location: str, target: Union[str, list], date: date):
        self.model_name = "CatboostRegression"
        # data.set_index("date", inplace=True)
        self.data = data[target]
        self.N = self.data.shape[0]
        self.location = location
        self.date = date
        self.N_tilde = N_tilde
        self.target = target

    def fit(self):
        forecaster = ForecasterAutoreg(
                        regressor = CatBoostRegressor(random_state=123, silent=True , n_jobs = 4),
                        lags = 15
                        )
        forecaster.fit(y=self.data)

        self.fit = forecaster

    def sigma2(self):
        resid = self.fit.in_sample_residuals
        sumofsq = np.sum(resid**2)
        return 1.0 / self.N * sumofsq

    def predict(self):
        y_pred = self.fit.predict(self.N_tilde)
        ar_params = np.array([1])
        ar_params = np.append(ar_params, self.fit.get_coef()["coef"].to_numpy().T * -1)
        # print(ar_params)
        ma = ar2ma(ar_params, np.ones(1), lags = 15)
        # print(ma)
        se_mean = np.sqrt(self.sigma2() * np.cumsum(ma ** 2))[:self.N_tilde]
        return quntiles(y_pred, se_mean, self.N_tilde, self.location, self.date)