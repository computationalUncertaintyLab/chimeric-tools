import pytest
import pandas as pd
from chimeric_tools.Data import get_raw_truth_df, load_truths, load_daily_covid, load_weekly_covid, get_unique_covid_data



def test_load_truths():
    assert isinstance(load_truths(), pd.DataFrame)


def test_load_daily_covid():
    assert isinstance(load_daily_covid(), pd.DataFrame)


def test_load_weekly_covid():
    assert isinstance(load_weekly_covid(), pd.DataFrame)


def test_get_unique_covid_data():
    data = get_unique_covid_data(geo_type="state", geo_values=['PA'], start_date='2022-01-01', end_date='2022-02-01')
    assert isinstance(data, pd.DataFrame)

def test_get_raw_truths():
    DATA_URL = "https://raw.githubusercontent.com/computationalUncertaintyLab/chimeric-tools/abe_sim/src/chimeric_tools/data/truth-Incident%20Cases.csv"
    data = get_raw_truth_df(DATA_URL)
    assert isinstance(data, pd.DataFrame)


def