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
from chimeric_tools.models import model


def check_for_data(path: str) -> bool:
    """
    Check to see if a  file exists

    Parameters
    ----------
        path : str path to the file that is being checked
    """
    return os.path.exists(path)


def load_cases_truths():
    """
    Loads raw truth data from CSSE dataset

    Returns
    ----------
        dataframe
    """
    stream = pkg_resources.resource_stream(__name__, "data/truth-Incident Cases.csv.gz")
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
    stream = pkg_resources.resource_stream(__name__, "data/truth-Incident Deaths.csv.gz")
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
    stream = pkg_resources.resource_stream(
        __name__, "data/truth-Incident Hospitalizations.csv.gz"
    )
    data = pd.read_csv(stream, compression="gzip")
    data["location"] = data["location"].astype(str)
    return data


# def load_cases_daily():
#     """
#     Load daily data complete with model predictions and residuals

#     Returns
#     ----------
#         dataframe
#     """
#     stream = pkg_resources.resource_stream(__name__, "data/cases_daily.csv.gz")
#     data = pd.read_csv(stream, compression="gzip")
#     data["location"] = data["location"].astype(str)
#     return data


# def load_deaths_daily():
#     """
#     Load daily data complete with model predictions and residuals

#     Returns
#     ----------
#         dataframe
#     """
#     stream = pkg_resources.resource_stream(__name__, "data/deaths_daily.csv.gz")
#     data = pd.read_csv(stream, compression="gzip")
#     data["location"] = data["location"].astype(str)
#     return data


# def load_hosps_daily():
#     """
#     Load daily data complete with model predictions and residuals

#     Returns
#     ----------
#         dataframe
#     """
#     stream = pkg_resources.resource_stream(__name__, "data/hosps_daily.csv.gz")
#     data = pd.read_csv(stream, compression="gzip")
#     data["location"] = data["location"].astype(str)
#     return data


def load_cases_weekly():
    """
    Load weekly data complete with model predictions and residuals

    Returns
    ----------
        dataframe
    """
    stream = pkg_resources.resource_stream(__name__, "data/cases_weekly.csv.gz")
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
    stream = pkg_resources.resource_stream(__name__, "data/deaths_weekly.csv.gz")
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
    stream = pkg_resources.resource_stream(__name__, "data/hosps_weekly.csv.gz")
    data = pd.read_csv(stream, compression="gzip")
    data["location"] = data["location"].astype(str)
    return data


def get_unique_covid_data(
    geo_type: str,
    geo_values: Union[str, Iterable[str]],
    start_date: Optional[date],
    end_date: Optional[date],
) -> pd.DataFrame:
    """
    Gets covid data from Delphi API

    Parameters
    ----------

    geo_type: the type of the geo value
    geo_value: the value of the geo
    start_date: the start date of the data
    end_date: the end date of the data

    Returns
    ----------
        dataframe of the covid data

    """

    # --checking inputs
    if not (geo_type == "state" or geo_type == "county"):
        raise Exception("geo_type must be 'state' or 'county'")
    if start_date is None:
        start_date = date(2020, 1, 22)

    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d").date()

    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d").date()

    # --call to Delphi API for incident cases
    data = covidcast.signal(
        data_source="jhu-csse",
        signal="confirmed_incidence_num",
        geo_type=geo_type,
        geo_values=geo_values,
        start_day=start_date,
        end_day=end_date,
    )

    if geo_type == "county":
        # --configure df
        df = data[["time_value", "geo_value", "value"]]
        df = df.rename(columns={"geo_value": "location", "time_value": "date"})
        df["location"] = df["location"].astype(int).astype(str)
        # --change fip to name
        df["location_name"] = covidcast.fips_to_name(df["location"])
    else:
        # --configure df
        df = pd.DataFrame()
        df["date"] = data["time_value"]
        # df["location"] = [x[0:2] for x in covidcast.abbr_to_fips(data["geo_value"], ignore_case=True)]
        # --change all values to uppercase
        df["location"] = data["geo_value"].apply(lambda x: x.upper())
        # --change all abbrivations to full name
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
        include: Union[list, None] = None,
        custom_data: Optional[pd.DataFrame] = None,
    ):
        """
        Initialize the COVID_DATA class

        """

        __DATA_PATH = os.path.dirname(__file__) + "/data/d"

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
            if not check_for_data(__DATA_PATH):
                print(
                    "Downloading data...you must have gone out of you way to delete the data in lib :)"
                )
                get_raw_truth_df(DATA_URL).to_csv(__DATA_PATH)


        if include is None: 
            self.data = load_cases_weekly()
            self.data = self.data.merge(load_deaths_weekly(), on=["date", "location", "location_name", "EW", "end_date"])
            self.data = self.data.merge(load_hosps_weekly(), on=["date", "location", "location_name", "EW", "end_date"])
        elif isinstance(include, list):
            is_first = True
            for i in include:
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
                elif i == "hospitals":
                    if is_first:
                        self.data = load_hosps_weekly()
                        is_first = False
                    else:
                        self.data = self.data.merge(load_hosps_weekly(), on=["date", "location", "location_name", "EW", "end_date"])
                else:
                    raise Exception("include must be 'cases', 'deaths', or 'hospitals'")

        self.data["date"] = pd.to_datetime(self.data["date"]).dt.date
        self.data["location"] = self.data["location"].astype(str)

        # --sets to geo_values to right type
        if geo_values is None:
            self.geo_values = self.data["location"].unique()
        elif isinstance(geo_values, list):
            self.geo_values = np.array(geo_values)
        elif isinstance(geo_values, str):
            self.geo_values = np.array([geo_values])
        else:
            self.geo_values = geo_values

        # --get current dates
        max_date = max(self.data["date"])
        min_date = min(self.data["date"])

        if isinstance(start_date, str):
            self.start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
        else:
            self.start_date = start_date
        if isinstance(end_date, str):
            self.end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
        else:
            self.end_date = end_date

        # --set start and end dates
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
                "end_date is after the latest date in the data. We will download this data for you. It might take some time"
            )
            # --check if the data is already downloaded
            self.file_hash = self.create_file_hash()
            file_path = os.path.dirname(__file__) + "/data" + self.file_hash + ".csv"
            if check_for_data(file_path):
                self.data = pd.read_csv(file_path)
            else:
                # loc data that we already have
                loc_data = self.download_data(
                    start_date=max_date + timedelta(days=1), end_date=self.end_date
                )
                self.data = pd.concat([self.data, loc_data])
                print(self.data)
                self.data = model(self.data)

        # --loc all data
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
        # --create hash of all FIPS and start and end dates
        hash_string = (
            "".join([str(i) for i in self.geo_values])
            + (str(self.start_date))
            + (str(self.end_date))
        )
        return hash_string

    def download_data(self, start_date: Optional[date], end_date: Optional[date]):
        """
        Downloads all geo values data that is not in not on file
        """
        # --sort state and county FIPS
        mask = np.array([len(_) >= 4 for _ in self.geo_values])
        county_values = self.geo_values[mask]
        state_values = covidcast.fips_to_abbr(self.geo_values[~mask])

        # --create empty df
        county = pd.DataFrame(columns=["date", "location", "location_name", "value"])
        state = pd.DataFrame(columns=["date", "location", "location_name", "value"])

        # --download county data
        if len(county_values) > 0:
            print("Downloading county data")
            county = get_unique_covid_data(
                geo_type="county",
                geo_values=county_values,
                start_date=start_date,
                end_date=end_date,
            )
        # --download state data
        if len(state_values) > 0:
            print("Downloading state data")
            state_values = np.char.upper(state_values)
            state = get_unique_covid_data(
                geo_type="state",
                geo_values=state_values,
                start_date=start_date,
                end_date=end_date,
            )

        return pd.concat([county, state])
