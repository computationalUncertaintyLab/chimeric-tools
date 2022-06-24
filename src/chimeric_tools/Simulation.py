"""
berlin

Simulate COVID data


"""
import os.path
from tkinter import Y
from typing import (
    Dict,
    Generator,
    Union,
)
import numpy as np
import pandas as pd
from arch.typing import ArrayLike1D
from arch.bootstrap import CircularBlockBootstrap
from chimeric_tools import Data
import statsmodels.api as sm




class Model:
    """
    Just a quick and dirty test model
    """

    def __init__(self, data: pd.DataFrame):
        self.data = np.array(data["value"][1:]).reshape(-1,)
        self.model = sm.tsa.statespace.SARIMAX(self.data, order=(2, 1, 0))
        self.result = self.model.fit(disp=0)
        self.preds = self.result.predict()
        self.residuals = self.result.resid

    def get_preds(self):
        """
        Returns the predictions
        """
        return self.preds

    def get_residuals(self):
        """
        Returns the residuals
        """
        return self.residuals


class COVID:
    """
    Covid simulation class
    """

    def __init__(
        self,
        geo_values: Union[np.ndarray, Dict[str, float], int, None],
        seed: Union[None, int, Generator],
    ) -> None:
        if check_for_data(DATA_PATH):
            self.data = load_data(DATA_PATH)
        else:
            self.data = pd.DataFrame(
                columns=["date", "location", "location_name", "value"]
            )

        if isinstance(geo_values, np.ndarray):
            self.geo_values = geo_values
            self.p = None
        elif isinstance(geo_values, dict):
            self.geo_values = np.array(list(geo_values.keys()))
            self.p = np.array(list(geo_values.values()))
        elif isinstance(geo_values, int):
            self.geo_values = np.array([geo_values])
            self.p = None
        elif geo_values is None:
            self.geo_values = self.data["location"].unique()
            self.p = None
        

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
            a=len(self.geo_values), size=reps, p=self.p, dtype=np.int64
        )
        return self.geo_values[indices]

    def simulate(self, reps):
        """
        Simulates the data
        """
        geo_for_sample = self.pick_geo_values(reps)

        for geo_value in geo_for_sample:
            sub_data = self.data[self.data["location"] == geo_value]
