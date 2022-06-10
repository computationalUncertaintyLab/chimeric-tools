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