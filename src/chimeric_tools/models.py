import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
import stan
import matplotlib.pyplot as plt
from scipy.stats import norm
from datetime import date, timedelta
from typing import Union

def train_simulated_data(data: pd.DataFrame, models: Union[list, str]):
    """
    Trains simulated data with all models
    """
    import chimeric_tools.models
    df = pd.DataFrame(columns=["forecast_date", "target_end_date", "location", "sim", "model", "value"])
    for sim in data["sim"].unique():
        print("Sim Number: ", sim)
        sub_data = data.loc[data["sim"] == sim]
        for model_name in models:
            print("\tModel Name: ", model_name)
            class_ = getattr(chimeric_tools.models, model_name)
            for i in range(15, len(sub_data)+1):
                splice_data = sub_data[:i]
                print("\t\tLen: ", len(splice_data))
                model = class_(data = splice_data, param = "cases", N_tilde = 4, location = splice_data["location"].iloc[0], date = max(splice_data["date"]))
                model.fit()
                preds = model.predict()
                preds["model"] = model_name
                preds["sim"] = sim

                df = pd.concat([df, preds])
    return df.sort_values(by=["sim", "model", "target_end_date", "forecast_date", "quantile"])




def model(data: pd.DataFrame):
    """
    Adds predictions and residuals to the data
    """
    for fip in data["location"].unique():
        sub_data = data.loc[data["location"] == fip]
        m = ARIMA(sub_data)
        data.loc[data["location"] == fip,"preds"] = m.preds
        data.loc[data["location"] == fip,"residuals"] = m.residuals
    return data


def format_sample(data: np.ndarray, start_date: date, location: str):
    """
    Formats the data from stan model
    """
    data_predictions = {"forecast_date":[], "target_end_date":[], "location":[], "sample":[], "value":[]}

    for N_tilde, samples in enumerate(data):
        for n, sample in enumerate(samples):
            data_predictions["forecast_date"].append(start_date)
            data_predictions["target_end_date"].append(start_date + timedelta(weeks=N_tilde+1))
            data_predictions["location"].append(location)
            data_predictions["sample"].append(n)
            data_predictions["value"].append(sample)
    return pd.DataFrame(data_predictions)


def format_quantiles(data: pd.DataFrame):
    def createQuantiles(x):
            quantiles = np.array([0.010, 0.025, 0.050, 0.100, 0.150, 0.200, 0.250, 0.300, 0.350, 0.400, 0.450, 0.500
                                  ,0.550, 0.600, 0.650, 0.700, 0.750, 0.800, 0.850, 0.900, 0.950, 0.975, 0.990])
            quantileValues = np.percentile( x["value"], q=100*quantiles)     
            return pd.DataFrame({"quantile":list(quantiles),"value":list(quantileValues)})

    dataQuantiles = data.groupby(["forecast_date", "target_end_date", "location"]).apply(lambda x:createQuantiles(x)).reset_index().drop(columns="level_3")
        
    return dataQuantiles

def from_statsmodels_to_quntiles(preds, N_tilde: int,  location: str, date: date):
    """
    Converts the statsmodels predictions to quantiles
    """
    data_predictions = {"forecast_date":[], "target_end_date":[], "location":[], "quantile":[], "value":[]}
    quantiles = np.array([0.010, 0.025, 0.050, 0.100, 0.150, 0.200, 0.250, 0.300, 0.350, 0.400, 0.450, 0.500
                          ,0.550, 0.600, 0.650, 0.700, 0.750, 0.800, 0.850, 0.900, 0.950, 0.975, 0.990])
    for quantile in quantiles:
        q = norm.ppf(quantile)
        values = preds.predicted_mean + q * preds.se_mean
        data_predictions["forecast_date"].extend(N_tilde*[date])
        data_predictions["target_end_date"].extend([date + timedelta(weeks=i) for i in range(1, N_tilde+1)])
        data_predictions["location"].extend(N_tilde*[location])
        data_predictions["quantile"].extend(N_tilde*[quantile])
        data_predictions["value"].extend(values)
    return pd.DataFrame(data_predictions)


def plot_predictions(data: pd.DataFrame, preds: pd.DataFrame, to_plot: str):
    """
    Plots the predictions and residuals
    """
    dates = preds["target_end_date"].unique()
    low = preds.loc[preds["quantile"] == 0.025, "value"]
    mid = preds.loc[preds["quantile"] == 0.500, "value"]
    high = preds.loc[preds["quantile"] == 0.975, "value"]


    plt.figure(figsize=(10,6), dpi=150)
    plt.plot(data["date"], data[to_plot], label="Truth Data")
    plt.plot(dates, mid, label="Predictions", color="black")
    plt.fill_between(dates, low, high, color="red", alpha=0.5, label="95% Confidence Interval")
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel(to_plot)
    plt.show()

    
class ARIMA():
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


class AR1():
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
        
        data = {"N": self.N, 
                "y": self.data,
                "N_tilde": self.N_tilde,
                }
        model = stan.build(program_code=stan_code, data=data)
        self.fit = model.sample(num_chains = 1, num_samples= 1*10**3)


    def predict(self):
        predictions = self.fit["y_tilde"] # this is coming from the model object
        predictions = format_sample(predictions, self.date, self.location)
        predictions = format_quantiles(predictions)
        return predictions


class AR3():
    """
    Stats Model AR(6)
    """
    
    def __init__(self, data: np.ndarray, param: str, N_tilde: int, location: str, date: date):
        self.model_name = "StatsModelAR6"
        data.set_index("date", inplace=True)
        self.data = data[param]
        self.location = location
        self.date = date
        self.N = len(data)
        self.N_tilde = N_tilde

    def fit(self):
        model = AutoReg(self.data, lags=3)
        self.fit = model.fit()

    def predict(self):
        predictions = self.fit.get_prediction(start=self.N, end=self.N + self.N_tilde - 1)
        predictions = from_statsmodels_to_quntiles(predictions, self.N_tilde, self.location, self.date)
        return predictions

class AR6():
    """
    Stats Model AR(6)
    """
    
    def __init__(self, data: np.ndarray, param: str, N_tilde: int, location: str, date: date):
        self.model_name = "StatsModelAR6"
        data.set_index("date", inplace=True)
        self.data = data[param]
        self.location = location
        self.date = date
        self.N = len(data)
        self.N_tilde = N_tilde

    def fit(self):
        model = AutoReg(self.data, lags=6)
        self.fit = model.fit()

    def predict(self):
        predictions = self.fit.get_prediction(start=self.N, end=self.N + self.N_tilde - 1)
        predictions = from_statsmodels_to_quntiles(predictions, self.N_tilde, self.location, self.date)
        return predictions