import pandas as pd
import numpy as np
from typing import Optional, Union, Iterable
from datetime import date


class COVID():
    def __init__(self) -> None:
        if self.check_for_data():
            self.data = self.load_data()
        else:
            self.data = self.download_data(geo_type = "state", geo_values = "pa", start_day = date(2020,1,22), end_day = date(2022,1,22))
            self.save_data(self.data)

    def check_for_data(self) -> bool:
        import os.path
        return os.path.exists("./covid_data.feather")

    def save_data(self, data: pd.DataFrame) -> None:
        data.reset_index().to_feather("covid_data.feather")

    def load_data(self) -> pd.DataFrame:
        return pd.read_feather("covid_data.feather")
    
    def download_data(self, geo_type: str, geo_values: Union[str, Iterable[str]], start_day: Optional[date], end_day: Optional[date]) -> None:
        from chimeric_tools import Data
        return Data.get_covid_data(geo_type, geo_values, start_day, end_day)


    def block_boostrap(x, block_size: int, overlap: int = 0) -> list:
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
        for i in range(n_blocks+1):
            d.append(blocks[np.random.randint(0, n_blocks)])
        # unpacks and chops end to correct length ( fun code :) )
        return [x for xs in d for x in xs][:-(block_size - overflow_len)]

    def plot_sim(self, actuals, pred, residuals, iterations: int, block_size: int = 5) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        fig = plt.figure(figsize=(12,8), dpi=150)
        for i in range(0,iterations):
            d = self.NBB(residuals, block_size)
            new =  pred + d
            plt.plot(new, color='blue', alpha=0.05)
        plt.plot(actuals, color='black')
        plt.plot(pred, color='red')
        colors = ['black', 'red', 'blue']
        lines = [Line2D([0], [0], color=c) for c in colors]
        labels = ['Actual', 'Prediction', 'NBB']
        plt.legend(lines, labels)
        plt.xlabel("Model Week")
        plt.ylabel("Cases")
        plt.show()

    
