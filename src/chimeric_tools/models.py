import numpy as np
import pandas as pd
import statsmodels.api as sm
import stan

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

    def __init__(self, data: np.ndarray, N_tilde: int):
        self.model_name = "StanLinReg"
        self.data = data
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
        return predictions.T


