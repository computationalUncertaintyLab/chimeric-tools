"""
Classes to keep data up to date

"""
import os
from typing import Optional, Union, Iterable
from datetime import date, timedelta, datetime
import io
import warnings
import pandas as pd
import numpy as np
import requests
import pkg_resources
import covidcast
from epiweeks import Week

def check_for_data(filename: str) -> bool:
    """
    Check to see if a  file exists

    Parameters
    ----------
        path : str path to the file that is being checked
    """
    return os.path.exists("".join(os.path.dirname(__file__) + "/data/" + filename))


def load_cases_truths():
    """
    Loads raw truth data from CSSE dataset

    Returns
    ----------
        dataframe
    """
    filename = "truth-Incident Cases.csv.gz"

    if not check_for_data(filename):
        download_from_github(filename)
    stream = pkg_resources.resource_stream(__name__, "data/" + filename)
    data = pd.read_csv(stream, compression="gzip")
    data["location"] = data["location"].astype(str)
    return data


def load_deaths_truths():
    """
    Loads raw truth data from CSSE dataset

    Returns
    ----------
        dataframe
    """
    filename = "truth-Incident Deaths.csv.gz"

    if not check_for_data(filename):
        download_from_github(filename)
    stream = pkg_resources.resource_stream(__name__, "data/" + filename)
    data = pd.read_csv(stream, compression="gzip")
    data["location"] = data["location"].astype(str)
    return data


def load_hosps_truths():
    """
    Loads raw truth data from CSSE dataset

    Returns
    ----------
        dataframe
    """
    filename = "truth-Incident Hospitalizations.csv.gz"
    
    if not check_for_data(filename):
        download_from_github(filename)
    stream = pkg_resources.resource_stream(__name__, "data/" + filename)
    data = pd.read_csv(stream, compression="gzip")
    data["location"] = data["location"].astype(str)
    return data


def load_cases_weekly():
    """
    Load weekly data complete with model predictions and residuals

    Returns
    ----------
        dataframe
    """
    filename = "cases_weekly.csv.gz"

    if not check_for_data(filename):
        download_from_github(filename)
    stream = pkg_resources.resource_stream(__name__, "data/" + filename)
    data = pd.read_csv(stream, compression="gzip")
    data["location"] = data["location"].astype(str)
    return data


def load_deaths_weekly():
    """
    Load weekly data complete with model predictions and residuals

    Returns
    ----------
        dataframe
    """
    filename = "deaths_weekly.csv.gz"

    if not check_for_data(filename):
        download_from_github(filename)
    stream = pkg_resources.resource_stream(__name__, "data/" + filename)
    data = pd.read_csv(stream, compression="gzip")
    data["location"] = data["location"].astype(str)
    return data


def load_hosps_weekly():
    """
    Load weekly data complete with model predictions and residuals

    Returns
    ----------
        dataframe
    """
    filename = "hosps_weekly.csv.gz"

    if not check_for_data(filename):
        download_from_github(filename)
    stream = pkg_resources.resource_stream(__name__, "data/" + filename)
    data = pd.read_csv(stream, compression="gzip")
    data["location"] = data["location"].astype(str)
    return data


def daily_to_weekly(data):
    """
    Converts daily data into weekly data by summing all cases for that week.
    Dataframe must be in format [ date: date or str, location: str, location_name, str,  value: int or float]

    Parameters
    ----------

        data: dataframe [ date: date or str, location: str, location_name, str,  value: int or float]

    Returns
    ----------

        Dataframe of weekly cases for each location
    """

    unique_dates = data.date.unique()

    weekly_data = {"date": [], "start_date": [], "end_date": [], "EW": []}

    # --iterate through all unique dates
    for date in unique_dates:
        weekly_data["date"].append(date)

        dt = pd.to_datetime(date)
        week = Week.fromdate(dt)

        startdate = week.startdate()
        weekly_data["start_date"].append(startdate)

        enddate = week.enddate()
        weekly_data["end_date"].append(enddate)

        weekly_data["EW"].append(week.cdcformat())
    weekly_data = pd.DataFrame(weekly_data)

    data = data.merge(weekly_data, on=["date"])

    def aggregate(x):
        cases = x.value.sum()

        return pd.Series({"cases": cases})

    weekly_date = data.groupby(
        ["location", "location_name", "start_date", "end_date", "EW"]
    ).apply(aggregate)
    weekly_date = weekly_date.reset_index()
    return weekly_date.rename(columns={"start_date": "date", "cases": "value"})


def download_from_github(filename) -> None:
    """
    Downloads files from github and saves them to the data folder

    Parameters
    ----------
        url : str
            url to the file to be downloaded
    
    """
    url = "https://github.com/computationalUncertaintyLab/chimeric-tools/raw/main/src/chimeric_tools/data/" + filename
    save_path = "".join(os.path.dirname(__file__) + "/data/" + filename)
    with open(save_path, "wb") as f:
        r = requests.get(url)
        f.write(r.content)


class CovidData(object):
    """
    Class to Manage COVID Data
    """

    def __init__(
        self,
        start_date: Union[date, None] = None,
        end_date: Union[date, None] = None,
        geo_values: Union[np.ndarray, list, str, None] = None,
        include: Union[list, None] = None,
        custom_data: Optional[pd.DataFrame] = None,
    ):
        """
        Initialize the COVID_DATA class

        """

        # --does the df have the right colums
        if isinstance(custom_data, pd.DataFrame):
            if custom_data.empty:
                raise Exception("custom_data is empty")
            if not {"date", "location", "value"}.issubset(custom_data.columns):
                raise Exception(
                    "custom_data must have columns 'date', 'location', and 'value'"
                )
            self.data = custom_data
        else:
            # TODO: add a way to download data from github in bulk and check if any data is missing
            pass


        if include is None: 
            self.include = ["cases", "deaths", "hosps"]
        elif isinstance(include, list):
            self.include = include
        else:
            raise Exception("include must be a list or None")
        
        is_first = True
        for i in self.include:
            if i == "cases":
                if is_first:
                    self.data = load_cases_weekly()
                    is_first = False
                else:
                    self.data = self.data.merge(load_cases_weekly(), on=["date", "location", "location_name", "EW", "end_date"])
            elif i == "deaths":
                if is_first:
                    self.data = load_deaths_weekly()
                    is_first = False
                else:
                    self.data = self.data.merge(load_deaths_weekly(), on=["date", "location", "location_name", "EW", "end_date"])
            elif i == "hosps":
                if is_first:
                    self.data = load_hosps_weekly()
                    is_first = False
                else:
                    self.data = self.data.merge(load_hosps_weekly(), on=["date", "location", "location_name", "EW", "end_date"])
            else:
                raise Exception("include must be 'cases', 'deaths', or 'hospitals'")

        self.data["date"] = pd.to_datetime(self.data["date"]).dt.date
        self.data["end_date"] = pd.to_datetime(self.data["end_date"]).dt.date
        self.data["location"] = self.data["location"].astype(str)

        # --sets to geo_values to right type
        if geo_values is None:
            self.geo_values = self.data["location"].unique()
        elif isinstance(geo_values, list):
            self.geo_values = np.array(geo_values)
        elif isinstance(geo_values, str):
            self.geo_values = np.array([geo_values])
        elif isinstance(geo_values, np.ndarray):
            self.geo_values = geo_values
        else:
            raise Exception("geo_values must be a list, string, or numpy array")

        # --get current dates
        max_date = max(self.data["date"])
        min_date = min(self.data["date"])


        # --set start and end dates
        if start_date is None:
            self.start_date = min_date
        elif isinstance(start_date, str):
            self.start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
        elif isinstance(start_date, date):
            self.start_date = start_date
        if end_date is None:
            self.end_date = max_date
        elif isinstance(end_date, str):
            self.end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
        elif isinstance(end_date, date):
            self.end_date = end_date

        # --correct start and end dates if they are out of range
        if self.start_date < min_date:
            warnings.warn(
                "start_date is before the earliest date in the data. Now using default start date"
            )
            self.start_date = min_date
        if self.end_date > max_date:
            warnings.warn(
                "end_date is after the latest date in the data. Now using default end date"
            )
            self.end_date = max_date

        # --set the date to the start of the week
        self.start_date = Week.fromdate(self.start_date).startdate()
        self.end_date = Week.fromdate(self.end_date).enddate()
        
        # --loc all data
        mask = (
            (self.data["date"] >= self.start_date)
            & (self.data["date"] <= self.end_date)
        ) & (self.data["location"].isin(self.geo_values))
        self.data = self.data.loc[mask]
