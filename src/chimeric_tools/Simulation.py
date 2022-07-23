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
from chimeric_tools.data import CovidData
from arch.bootstrap import CircularBlockBootstrap


class COVID(object):
    """
    Covid simulation class

    Parameters
    ----------
    start_date : date, optional
        The start date of the simulation. Defaults to None.
    end_date : date, optional
        The end date of the simulation. Defaults to None.
    geo_values : Union[np.ndarray, Dict[str, float], str, list, None], optional
        The geo values to use. Defaults to None.
    custom_data : Optional[pd.DataFrame], optional
        The custom data to use. Defaults to None.
    seed : Union[None, int, Generator], optional
        The seed to use. Defaults to None.

    Examples
    --------
    """

    def __init__(
        self,
        start_date: Union[date, None] = None,
        end_date: Union[date, None] = None,
        geo_values: Union[np.ndarray, Dict[str, float], str, list, None] = None,
        custom_data: Optional[pd.DataFrame] = None,
        seed: Union[None, int, Generator] = None,
    ) -> None:

        self.start_date = start_date
        self.end_date = end_date

        # --conver geo_values to correct type
        if isinstance(geo_values, (np.ndarray, list)):
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
        
        # --get covid data from data class
        self.data = CovidData(
            start_date=start_date,
            end_date=end_date,
            geo_values=self.geo_values,
            custom_data=custom_data,
        ).data
        if geo_values is None:
            self.geo_values = self.data["location"].unique()

        # --set seed
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
        Randomly generate geo values with probability p and repeat for reps times
        """
        indices = self.generator.choice(a=len(self.geo_values), size=reps, p=self.p)
        return np.array([self.geo_values[x] for x in indices])

    def simulate(self, reps):
        """
        Simulate reps number of simulations using random geo values and bootstrapped time series
        """
        geo_for_sample = self.pick_geo_values(reps)

        # --for each geo value boostrap the residuals and add to data
        for geo_value in geo_for_sample:
            sub_data = self.data[self.data["location"] == geo_value].reset_index()
            bs = CircularBlockBootstrap(5, sub_data["residual"])
            for data in bs.bootstrap(1):
                sim_data = (data[0][0]).reset_index(drop=True) + sub_data["pred"]
                yield (sim_data, geo_value)
