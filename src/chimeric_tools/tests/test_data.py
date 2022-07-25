import pytest
import pandas as pd
from chimeric_tools.Data import *



def test_load_truths():
    assert isinstance(load_cases_truths(), pd.DataFrame)
    assert isinstance(load_deaths_truths(), pd.DataFrame)
    assert isinstance(load_hosps_truths(), pd.DataFrame)

# def test_load_daily_covid():
#     assert isinstance(load_cases_daily(), pd.DataFrame)
#     assert isinstance(load_deaths_daily(), pd.DataFrame)
#     assert isinstance(load_hosps_daily(), pd.DataFrame)


def test_load_weekly_covid():
    assert isinstance(load_cases_weekly(), pd.DataFrame)
    assert isinstance(load_deaths_weekly(), pd.DataFrame)
    assert isinstance(load_hosps_weekly(), pd.DataFrame)


def test_get_unique_covid_data():
    data = get_unique_covid_data(geo_type="state", geo_values=['PA'], start_date='2022-01-01', end_date='2022-02-01')
    assert isinstance(data, pd.DataFrame)

def test_get_raw_truths():
    DATA_URL = "https://raw.githubusercontent.com/computationalUncertaintyLab/chimeric-tools/abe_sim/src/chimeric_tools/data/truth-Incident%20Cases.csv"
    data = get_raw_truth_df(DATA_URL)
    assert isinstance(data, pd.DataFrame)

def test_covid_class():
    assert isinstance(CovidData().data, pd.DataFrame)
    assert CovidData(start_date="2022-01-01", end_date="2022-01-02", geo_values="42095").create_file_hash() == '420952022-01-012022-01-02'

    #test date ranges
    assert CovidData(start_date="2019-01-01", end_date="2022-01-02", geo_values="42095").start_date == date(2020,1,19)
    assert CovidData(start_date="2020-01-01", end_date="2023-01-02", geo_values="42095").start_date == date(2020,1,19)
    