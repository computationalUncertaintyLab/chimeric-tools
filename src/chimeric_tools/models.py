import numpy as np
import pandas as pd
import statsmodels.api as sm

def model(data: pd.DataFrame):
    """
    Adds predictions and residuals to the data
    """
    for fip in data["location"].unique():
        sub_data = data.loc[data["location"] == fip]
        m = ARIMA(sub_data)
        data.loc[data["location"] == fip,"pred"] = m.preds
        data.loc[data["location"] == fip,"residual"] = m.residuals
    return data


class ARIMA():
    """
    Just a quick and dirty test model
    """

    def __init__(self, data: pd.DataFrame):
        self.data = np.array(data["value"])
        self.model = sm.tsa.statespace.SARIMAX(self.data, order=(2, 1, 0))
        self.result = self.model.fit(disp=0)
        self.preds = self.result.predict()
        self.residuals = self.result.resid
