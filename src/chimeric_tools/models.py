import numpy as np
import pandas as pd
import statsmodels.api as sm
import stan
import matplotlib.pyplot as plt
from datetime import date, timedelta

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
        self.fit = self.fit()

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
        return model.sample(num_chains = 4, num_samples= 1000)


    def predict(self):
        predictions = self.fit["y_tilde"] # this is coming from the model object
        predictions = format_sample(predictions, self.date, self.location)
        predictions = format_quantiles(predictions)
        return predictions


class AR2():
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
        self.fit = self.fit()

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
                real gamma;
                real<lower=0> sigma;
            }
            model {
                alpha ~ normal(0, 10);
                beta ~ normal(0, 10);
                gamma ~ normal(0, 10);
                sigma ~ cauchy(0, 2.5);
                for (n in 3:N)
                    y[n] ~ normal(alpha + beta*y[n-1] + gamma*y[n-2], sigma);
            }
            generated quantities {
                vector[N_tilde] y_tilde;
                vector[2] i = y[N-1:N];
                for (n in 1:N_tilde) {
                    y_tilde[n] = normal_rng(alpha + beta * i[2] + gamma * i[1], sigma);
                    i[1] = i[2];
                    i[2] = y_tilde[n];
                    }
            }
        """
        
        data = {"N": self.N, 
                "y": self.data,
                "N_tilde": self.N_tilde,
                }
        model = stan.build(program_code=stan_code, data=data)
        return model.sample(num_chains = 4, num_samples= 1000)


    def predict(self):
        predictions = self.fit["y_tilde"] # this is coming from the model object
        predictions = format_sample(predictions, self.date, self.location)
        predictions = format_quantiles(predictions)
        return predictions

