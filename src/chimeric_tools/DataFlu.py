from datetime import date, datetime
from typing import Union
import warnings
import pandas as pd
import numpy as np
from Data import check_for_data
from epiweeks import Week

CASES_TRUTHS = "truth-Incident FluCasesNew.csv.gz"
CASES_WEEKLY = "flu_cases_weekly.csv.gz"

def _load_flu_weekly():
	data = _load_file(CASES_WEEKLY)
	return _add_start_end_dates(data)

def _load_flu_data():
	data = _load_file(CASES_TRUTHS)
	return _add_start_end_dates(data)
	
def _load_file(filename):
	if check_for_data(filename):
		data = pd.read_csv("data/" + filename, compression="gzip")
		data = data.drop([
			"release_date", 
			"issue", 
			"num_ili", 
			"num_patients", 
			"num_providers", 
			"num_age_0", 
			"num_age_1", 
			"num_age_2", 
			"num_age_3", 
			"num_age_4", 
			"num_age_5",
			"ili"], axis=1)
		return data
	return None


def _add_start_end_dates(df: pd.DataFrame):
	df["year"] = df["epiweek"].astype("int") // 100
	df["week"] = df["epiweek"] - df["year"] * 100
	unique_dates = df["epiweek"].unique()
	weekly_data = {"start_date": [], "end_date": [], "EW": []}
	for date in unique_dates:
		week = Week.fromstring(str(date))

		weekly_data["start_date"].append(week.startdate())
		weekly_data["end_date"].append(week.enddate())
		weekly_data["EW"].append(week.cdcformat())

	weekly_data = pd.DataFrame(weekly_data)

	weekly_data["EW"] = weekly_data["EW"].astype(np.int64)
	weekly_data["start_date"] = pd.to_datetime(weekly_data["start_date"]).dt.date
	weekly_data["end_date"] = pd.to_datetime(weekly_data["end_date"]).dt.date
	
	join = df.merge(weekly_data, left_on="epiweek", right_on="EW")
	# renames region to location for consistency with COVID
	return join.rename(columns={"start_date": "date", "wili": "value"})

def flu_data(
	start_date: Union[date, str, None] = None,
	end_date: Union[date, str, None] = None,
	geo_values: Union[np.ndarray, list, str, None] = None):

	data = _load_flu_weekly()
	
	# --set geo_values to right type
	if geo_values is None:
		geo_values = data["location"].unique()
	elif isinstance(geo_values, list):
		geo_values = np.array(geo_values)
	elif isinstance(geo_values, str):
		geo_values = np.array([geo_values])
	elif isinstance(geo_values, np.ndarray):
		pass
	else:
		raise ValueError("geo_values must be a list, string, or numpy array")

	# --get min and max dates
	max_date = max(data["end_date"])
	min_date = min(data["date"])

	# --set start and end dates
	if start_date is None:
		start_date = min_date
	elif isinstance(start_date, str):
		start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
	elif isinstance(start_date, date):
		pass
	if end_date is None:
		end_date = max_date
	elif isinstance(end_date, str):
		end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
	elif isinstance(end_date, date):
		pass

	# --correct start/end dates if out of range
	if start_date < min_date:
		warnings.warn(
            "start_date is before the earliest date in the data. Now using default start date"
        )
		start_date = min_date
	if end_date > max_date:
		warnings.warn(
            "end_date is after the latest date in the data. Now using default end date"
        )
		end_date = max_date

	# --set the date to the start of the week
	start_date = Week.fromdate(start_date).startdate()
	end_date = Week.fromdate(end_date).enddate()

	mask = (
		(data["date"] >= start_date)
		& (data["end_date"] <= end_date)
	) & (data["location"].isin(geo_values))

	return data.loc[mask].reset_index(drop=True)

def get():
	return _load_flu_data()