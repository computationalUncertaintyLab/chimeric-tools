import os.path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arch.typing import ArrayLike1D
from matplotlib.lines import Line2D

from chimeric_tools import Data

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


class COVID:
    """
    Covid simulation class
    """

    def __init__(self, geo_values: ArrayLike1D) -> None:
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
        self.data = self.data.loc[self.data["location"].isin(geo_values)]

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
        mask = np.array([True if len(_) >= 4 else False for _ in missing_geo_values])
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

    def block_boostrap(self, x, block_size: int, overlap: int = 0) -> list:
        """
        Block Bootstrap

        Parameters
        ----------
        x : array_like
        block_size : int
        overlap : int  when overlap = 0 then this becomes a non-moving block bootstrap
        """
        # initialize the bootstrap array and the indices
        blocks = []
        start = 0
        end = block_size
        # if the block is going to go over the length of the array we stop
        while end <= len(x):
            blocks.append(x[start:end])
            start += block_size - overlap
            end += block_size - overlap
        n_blocks = len(blocks)
        overflow_len = len(x) - start
        d = []
        # randomly select n_blocks blocks from the list of blocks with replacement
        for _ in range(int((len(x) - overflow_len) / block_size) + 1):
            d.append(blocks[np.random.randint(0, n_blocks)])
        # unpacks and chops end to correct length ( fun code :) )
        return [x for xs in d for x in xs][: -(block_size - overflow_len)]

    def plot_sim(
        self,
        actuals,
        pred,
        residuals,
        iterations: int,
        block_size: int = 5,
        overlap: int = 0,
    ) -> None:

        """
        Plots the actuals, predicted, and residuals

        Parameters:
        -----------
        actuals : array_like
        pred : array_like
        residuals : array_like
        iterations : int
        block_size : int
        overlap : int

        """

        fig = plt.figure(figsize=(12, 8), dpi=150)
        for _ in range(0, iterations):
            _d = self.block_boostrap(residuals, block_size, overlap)
            new = pred + _d
            plt.plot(new, color="blue", alpha=0.05)
        plt.plot(actuals, color="black")
        plt.plot(pred, color="red")
        colors = ["black", "red", "blue"]
        lines = [Line2D([0], [0], color=c) for c in colors]
        labels = ["Actual", "Prediction", "NBB"]
        plt.legend(lines, labels)
        plt.xlabel("Model Week")
        plt.ylabel("Cases")
        plt.show()
