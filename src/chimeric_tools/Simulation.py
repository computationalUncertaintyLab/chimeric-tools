"""
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
from chimeric_tools.Data import covid_data
from arch.bootstrap import CircularBlockBootstrap
from arch.bootstrap import optimal_block_length as arch_optimal_block_length


def optimal_block_length(x: Union[np.ndarray, pd.Series, pd.DataFrame]):
    """
    This is a wrapper function for `arch.boostrap.optimal_block_length <https://arch.readthedocs.io/en/latest/bootstrap/generated/arch.bootstrap.optimal_block_length.html#id1>`_
    It returns the optimal block length for the given data.
    """
    return arch_optimal_block_length(x ** 2)["circular"]

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

    >>> from chimeric_tools.Simulation import COVID
    >>> import matplotlib.pyplot as plt
    >>> bs = COVID(start_date = "2021-01-01", end_date = "2021-12-31", geo_values = "US", include = ["cases"])
    >>> for data in bs.simulate(100):
    >>>     plt.plot(data[0]["cases"], color= "tab:blue", alpha = 0.1)
    >>> plt.xlabel("Weeks")
    >>> plt.ylabel("Cases")
    >>> plt.title("Covid Simulation of FIP: " + data[1])
    >>> plt.show()
    """

    def __init__(
        self,
        start_date: Union[date, str, None] = None,
        end_date: Union[date, str, None] = None,
        geo_values: Union[np.ndarray, Dict[str, float], str, list, None] = None,
        include: Union[list, None] = None,
        seed: Union[None, int, np.random.Generator] = None,
    ) -> None:

        self.start_date = start_date
        self.end_date = end_date

        # --conver geo_values to correct type
        if isinstance(geo_values, (np.ndarray, list)):
            self.geo_values = geo_values
            self.p = None
        elif isinstance(geo_values, dict):
            if not abs(1 - sum(geo_values.values())) <= 0.001:
                raise ValueError("geo_values must sum to 1")
            self.geo_values = np.array(list(geo_values.keys()))
            self.p = np.array(list(geo_values.values()))
        elif isinstance(geo_values, str):
            self.geo_values = np.array([geo_values])
            self.p = None
        elif geo_values is None:
            self.geo_values = None
            self.p = None
        
        if include is None: 
            self.include = ["cases", "deaths", "hosps"]
        elif isinstance(include, list):
            self.include = include
        else:
            raise Exception("include must be a list or None")

        # --get covid data from data class
        self.data = covid_data(
            start_date=start_date,
            end_date=end_date,
            geo_values=self.geo_values,
            include=include,
        )
        if geo_values is None:
            self.geo_values = self.data["location"].unique()

        # --set seed
        if isinstance(seed, np.random.Generator):
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

    def pick_geo_values(self, reps: int) -> np.ndarray:
        """
        Randomly generate geo values with probability p and repeat for reps times
        """
        indices = self.generator.choice(a=len(self.geo_values), size=reps, p=self.p)
        return np.array([self.geo_values[x] for x in indices])


    def simulate(self, block_length: Union[list, np.ndarray, int, str], reps: int):
        """
        Simulate reps number of simulations using random geo values and bootstrapped time series
        """

        # check block_length args
        auto = False
        if isinstance(block_length, (list, np.integer)):
            if not len(block_length) == len(self.include):
                raise ValueError("block_length must be a list of length the same as include")
        elif isinstance(block_length, int):
            block_length = [block_length] * len(self.include)
        elif isinstance(block_length, str) and block_length == "auto":
            auto = True
        else:
            raise ValueError("block_length does not match the correct type")

        geo_for_sample = self.pick_geo_values(reps)

        # --for each geo value boostrap the residuals and add to data

        sim_num = 0

        bs_data = pd.DataFrame()

        for geo_value in geo_for_sample:
            sub_data = self.data[self.data["location"] == geo_value].reset_index()

            # auto block length
            if auto:
                mask = []
                for i in self.include:
                    mask.append("".join("residuals_" + i))
                block_length = optimal_block_length(sub_data[mask])

            # --make temp dataframe to hold bootstrapped data
            bootstrapped_data = sub_data[["date", "end_date", "location", "location_name", "EW"]].copy()
            bootstrapped_data["sim"] = sim_num


            # --assemble dictionary of bootstrapped data
            for idx, i in enumerate(self.include):
                mask = "".join("residuals_" + i)

                # --bootstrap the data
                bs = CircularBlockBootstrap(int(block_length[idx]), sub_data[mask])
                for data in bs.bootstrap(1):
                    sim_data = data[0][0]

                    preds = "".join("preds_" + i)
                    bootstrapped_data[i] = sim_data.reset_index(drop=True) + sub_data[preds]
            sim_num += 1
            bs_data = pd.concat([bs_data, bootstrapped_data])
        return bs_data.reset_index(drop=True)