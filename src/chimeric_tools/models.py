import numpy as np
import pandas as pd
import statsmodels.api as sm

def model(data: pd.DataFrame):
    """
    Adds predictions and residuals to the data
    """
    to_cat = pd.DataFrame(
        columns=["date", "location", "location_name", "value", "pred", "residual"]
    )
    for fip in data["location"].unique():
        sub_data = data.loc[data["location"] == fip]
        print(sub_data.shape)
        m = ARIMA(sub_data)
        sub_data["pred"] = m.preds
        sub_data["residual"] = m.residuals
        print(sub_data)
        to_cat = pd.concat([to_cat, sub_data])
        break
    return to_cat


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
