"""
Classes to keep data up to date

"""
import os
from typing import Optional, Union, Iterable
from datetime import date
import pandas as pd
import numpy as np
import covidcast
from epiweeks import Week


def check_for_data(path: str) -> bool:

    """
    Checks if there is covid data
    """

    return os.path.exists(path)


def save_feather(data: pd.DataFrame, path: str) -> None:
    """
    Saves the data to a feather file
    """
    data.to_feather(path)


def load_feather(path: str) -> pd.DataFrame:
    """
    Loads the data from a feather file
    """
    return pd.read_feather(path)


def get_unique_covid_data(
    geo_type: str,
    geo_values: Union[str, Iterable[str]],
    start_day: Optional[date],
    end_day: Optional[date],
) -> pd.DataFrame:
    """
    Gets covid data from Delphi API

    :param geo_type: the type of the geo value
    :param geo_value: the value of the geo
    :param start_date: the start date of the data
    :param end_date: the end date of the data
    :return: the dataframe of the covid data

    """

    # --checking inputs
    if not (geo_type == "state" or geo_type == "county"):
        raise Exception("geo_type must be 'state' or 'county'")
    if start_day is None:
        start_day = date(2020, 1, 22)

    data = covidcast.signal(
        data_source="jhu-csse",
        signal="confirmed_incidence_num",
        geo_type=geo_type,
        geo_values=geo_values,
        start_day=start_day,
        end_day=end_day,
    )

    if geo_type == "county":
        # --configure df
        df = data[["time_value", "geo_value", "value"]]
        df = df.rename(columns={"geo_value": "location", "time_value": "date"})
        df["location"] = df["location"].astype(int).astype(str)
        df["location_name"] = covidcast.fips_to_name(df["location"])
    else:
        # --configure df
        df = pd.DataFrame()
        df["date"] = data["time_value"]
        #df["location"] = [x[0:2] for x in covidcast.abbr_to_fips(data["geo_value"], ignore_case=True)]
        df["location"] = data["geo_value"].apply(lambda x: x.upper())
        df["location_name"] = covidcast.abbr_to_name(
            data["geo_value"], ignore_case=True
        )
        df["value"] = data["value"]
    return df


def get_time_locked_dat() {
    
}
def daily_to_weekly(data):
    """
    Converts the daily data to weekly data

    df must be in format [ date, location, location_name,  value]
    """
    unique_dates = data.date.unique()

    fromDate2EW = {"date": [], "start_date": [], "end_date": [], "EW": []}
    for date in unique_dates:
        fromDate2EW["date"].append(date)

        dt = pd.to_datetime(date)
        week = Week.fromdate(dt)

        startdate = week.startdate()
        fromDate2EW["start_date"].append(startdate)

        enddate = week.enddate()
        fromDate2EW["end_date"].append(enddate)

        fromDate2EW["EW"].append(week.cdcformat())
    fromDate2EW = pd.DataFrame(fromDate2EW)

    data = data.merge(fromDate2EW, on=["date"])

    def aggregate(x):
        cases = x.value.sum()

        return pd.Series({"cases": cases})

    weekly_date = data.groupby(
        ["location", "location_name", "start_date", "end_date", "EW"]
    ).apply(aggregate)
    weekly_date.reset_index().to_feather("covid_cases.feather")


__DATA_PATH = "some path"

class CovidData:
    """
    Class to Manage COVID Data
    """

    def __init__(self):
        """
        Initialize the COVID_DATA class

        """
        


    def find_missing_geo_values(self, geo_values):
        """
        Finds the missing geo values in the data
        """
        locations = self.data["location"].unique()
        return geo_values[~np.isin(geo_values, locations)]

    def download_data(self, missing_geo_values):
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
            county = get_unique_covid_data(
                geo_type="county",
                geo_values=county_values,
                start_day=None,
                end_day=None,
            )
        if len(state_values) > 0:
            state = get_unique_covid_data(
                geo_type="state", geo_values=state_values, start_day=None, end_day=None
            )

        self.data = pd.concat([self.data, county, state])