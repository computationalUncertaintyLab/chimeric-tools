"""
berlin

Simulate COVID data


"""
from typing import (
    Dict,
    Optional,
    Generator,
    Union,
)
from datetime import date
import numpy as np
import pandas as pd
import statsmodels.api as sm
from chimeric_tools.data import CovidData
from arch.bootstrap import CircularBlockBootstrap



class Model(object):
    """
    Just a quick and dirty test model
    """

    def __init__(self, data: pd.DataFrame):
        self.data = np.array(data["value"])
        self.model = sm.tsa.statespace.SARIMAX(self.data, order=(2, 1, 0))
        self.result = self.model.fit(disp=0)
        self.preds = self.result.predict()
        self.residuals = self.result.resid


class COVID(object):
    """
    Covid simulation class
    """

    def __init__(
        self,
        start_date: Union[date, None] = None,
        end_date: Union[date, None] = None,
        geo_values: Union[np.ndarray, Dict[str, float], str, None] = None,
        custom_data: Optional[pd.DataFrame] = None,
        seed: Union[None, int, Generator] = None,
    ) -> None:

        self.start_date = start_date
        self.end_date = end_date
        if isinstance(geo_values, np.ndarray):
            self.geo_values = geo_values
            self.p = None
        elif isinstance(geo_values, dict):
            self.geo_values = np.array(list(geo_values.keys()))
            self.p = np.array(list(geo_values.values()))
        elif isinstance(geo_values, str):
            self.geo_values = np.array([geo_values])
            self.p = None
        elif geo_values is None:
            self.geo_values = None
            self.p = None
        self.data = CovidData(
            start_date=start_date,
            end_date=end_date,
            geo_values=self.geo_values,
            custom_data=custom_data,
        ).data
        if geo_values is None:
            self.geo_values = self.data["location"].unique()

        if isinstance(seed, Generator):
            self.generator = seed
        elif isinstance(seed, (int, np.integer)):
            self.generator = np.random.default_rng(int(seed))
        elif seed is None:
            self.generator = np.random.default_rng()
        else:
            raise TypeError(
                "generator keyword argument must contain a NumPy Generator or "
                "RandomState instance or an integer when used."
            )
    

    def pick_geo_values(self, reps):
        """
        Picks the geo values to use
        """
        indices = self.generator.choice(
            a=len(self.geo_values), size=reps, p=self.p)
        return self.geo_values[indices]

    def simulate(self, reps):
        """
        Simulates the data
        """
        geo_for_sample = self.pick_geo_values(reps)

        for geo_value in geo_for_sample:
            sub_data = self.data[self.data["location"] == geo_value].reset_index()
            bs = CircularBlockBootstrap(5, sub_data["residual"])
            for data in bs.bootstrap(1):
                return data[0][0].reset_index(drop=True) + sub_data["pred"]
