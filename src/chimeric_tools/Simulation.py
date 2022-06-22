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


DATA_PATH = "./covid_data.feather"


def check_for_data(path: str) -> bool:

    """
    Checks if there is covid data
    """

    return os.path.exists(path)


def save_data(data: pd.DataFrame, path: str) -> None:
    """
    Saves the data to a feather file
    """
    data.to_feather(path)


def load_data(path: str) -> pd.DataFrame:
    """
    Loads the data from a feather file
    """
    return pd.read_feather(path)


class MODEL:
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
        self.geo_values = geo_values
        if check_for_data(DATA_PATH):
            self.data = load_data(DATA_PATH)
        else:
            self.data = pd.DataFrame(
                columns=["date", "location", "location_name", "value"]
            )
        missing = self.find_missing_geo_values()
        if missing.size > 0:
            self.update_data(missing)
            save_data(self.data, DATA_PATH)

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

    def find_missing_geo_values(self):
        """
        Finds the missing geo values in the data
        """
        locations = self.data["location"].unique()
        return self.geo_values[~np.isin(self.geo_values, locations)]

    def update_data(self, missing_geo_values):
        """
        Downloads all geo values data that is not in not on file
        """
        mask = np.array([len(_) >= 4 for _ in missing_geo_values])
        county_values = missing_geo_values[mask]
        state_values = missing_geo_values[~mask]

        county = pd.DataFrame(columns=["date", "location", "location_name", "value"])
        state = pd.DataFrame(columns=["date", "location", "location_name", "value"])

        state_values = np.char.upper(state_values)
        if len(county_values) > 0:
            county = Data.get_covid_data(
                geo_type="county",
                geo_values=county_values,
                start_day=None,
                end_day=None,
            )
        if len(state_values) > 0:
            state = Data.get_covid_data(
                geo_type="state", geo_values=state_values, start_day=None, end_day=None
            )

        self.data = pd.concat([self.data, county, state])

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
