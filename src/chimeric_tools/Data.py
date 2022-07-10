"""
Classes to keep data up to date

"""
import os
from typing import Optional, Union, Iterable
from datetime import date, timedelta
import io
import warnings
import pandas as pd
import numpy as np
import requests
import pkg_resources
import covidcast
from epiweeks import Week
from chimeric_tools.models import model


def check_for_data(path: str) -> bool:
    """
    Checks to see if a data file exists

    Parameters
    ----------
        path : str path to the data file
    """
    return os.path.exists(path)


def load_truths():
    """
    Loads package data
    """
    stream = pkg_resources.resource_stream(__name__, "data/truth-Incident Cases.csv")
    data = pd.read_csv(stream)
    data["location"] = data["location"].astype(str)
    return data

def load_daily_covid():
    """
    Loads the daily covid data
    """
    stream = pkg_resources.resource_stream(__name__, "data/daily_covid.csv.gz")
    data = pd.read_csv(stream, compression="gzip")
    data["location"] = data["location"].astype(str)
    return data

def load_weekly_covid():
    """
    Loads the weekly covid data
    """
    stream = pkg_resources.resource_stream(__name__, "data/weekly_covid.csv.gz")
    data = pd.read_csv(stream, compression="gzip")
    data["location"] = data["location"].astype(str)
    return data


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
        # df["location"] = [x[0:2] for x in covidcast.abbr_to_fips(data["geo_value"], ignore_case=True)]
        df["location"] = data["geo_value"].apply(lambda x: x.upper())
        df["location_name"] = covidcast.abbr_to_name(
            data["geo_value"], ignore_case=True
        )
        df["value"] = data["value"]
    return df


def get_raw_truth_df(url) -> pd.DataFrame:
    """
    Gets raw csv and turns it into dataframe
    """
    url_req = requests.get(url).content
    return pd.read_csv(io.StringIO(url_req.decode("utf-8")))


def daily_to_weekly(data):
    """
    Converts the daily data to weekly data

    df must be in format [ date: date or str, location: str, location_name, str,  value: int or float]
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
    weekly_date = weekly_date.reset_index()
    return weekly_date.rename(columns={"start_date": "date", "cases": "value"})


DATA_URL = "https://raw.githubusercontent.com/computationalUncertaintyLab/chimeric-tools/abe_sim/src/chimeric_tools/data/truth-Incident%20Cases.csv"


class CovidData(object):
    """
    Class to Manage COVID Data
    """

    def __init__(
        self,
        start_date: Union[date, None] = None,
        end_date: Union[date, None] = None,
        geo_values: Union[np.ndarray, list, str, None] = None,
        custom_data: Optional[pd.DataFrame] = None,
    ):
        """
        Initialize the COVID_DATA class

        """

        __DATA_PATH = os.path.dirname(__file__) + "/data/truth-Incident Cases.csv"

        if isinstance(custom_data, pd.DataFrame):
            if custom_data.empty:
                raise Exception("custom_data is empty")
            if not {"date", "location", "value"}.issubset(custom_data.columns):
                raise Exception(
                    "custom_data must have columns 'date', 'location', and 'value'"
                )
            self.data = custom_data
        else:
            if not check_for_data(__DATA_PATH):
                print(
                    "Downloading data...you must have gone out of you way to delete the data in lib :)"
                )
                get_raw_truth_df(DATA_URL).to_csv(__DATA_PATH)
            self.data = load_weekly_covid()

        self.data["date"] = pd.to_datetime(self.data["date"]).dt.date
        self.data["location"] = self.data["location"].astype(str)

        if geo_values is None:
            self.geo_values = self.data["location"].unique()
        elif isinstance(geo_values, list):
            self.geo_values = np.array(geo_values)
        elif isinstance(geo_values, str):
            self.geo_values = np.array([geo_values])
        else:
            self.geo_values = geo_values

        max_date = max(self.data["date"])
        min_date = min(self.data["date"])

        self.start_date = start_date
        self.end_date = end_date

        if self.start_date is None:
            self.start_date = min_date
        if self.end_date is None:
            self.end_date = max_date
        if self.start_date < min_date:
            warnings.warn(
                "start_date is before the earliest date in the data. Now using default start date"
            )
            self.start_date = min_date
        if self.end_date > max_date:
            warnings.warn(
                "end_date is after the latest date in the data. We will downlaod this data for you. It might take some time"
            )
            # --check if the data is already downloaded
            self.file_hash = self.create_file_hash()
            file_path = "./data/" + self.file_hash + ".csv"
            if check_for_data(file_path):
                self.data = pd.read_csv("./data/" + self.file_hash + ".csv")
            else:
                # loc data that we already have
                loc_data = self.download_data(
                    start_date=max_date + timedelta(days=1), end_date=self.end_date
                )
                self.data = pd.concat([self.data, loc_data])
                self.data = model(self.data)

        mask = (
            (self.data["date"] >= self.start_date)
            | (self.data["date"] <= self.end_date)
        ) & (self.data["location"].isin(self.geo_values))
        self.data = self.data.loc[mask]

    def create_file_hash(self):
        """
        Creates a hash of the data
        """
        self.geo_values.sort()
        hash_string = (
            "".join([str(i) for i in self.geo_values])
            + (str(self.start_date))
            + (str(self.end_date))
        )
        return str(hash(hash_string))

    def download_data(self, start_date: Optional[date], end_date: Optional[date]):
        """
        Downloads all geo values data that is not in not on file
        """
        mask = np.array([len(_) >= 4 for _ in self.geo_values])
        county_values = self.geo_values[mask]
        state_values = covidcast.fips_to_abbr(self.geo_values[~mask])

        county = pd.DataFrame(columns=["date", "location", "location_name", "value"])
        state = pd.DataFrame(columns=["date", "location", "location_name", "value"])

        state_values = np.char.upper(state_values)
        if len(county_values) > 0:
            print("Downloading county data")
            county = get_unique_covid_data(
                geo_type="county",
                geo_values=county_values,
                start_day=start_date,
                end_day=end_date,
            )
        if len(state_values) > 0:
            print("Downloading state data")
            state = get_unique_covid_data(
                geo_type="state",
                geo_values=state_values,
                start_day=start_date,
                end_day=end_date,
            )

        return pd.concat([county, state])
