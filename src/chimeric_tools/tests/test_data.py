import pytest
import pandas as pd
from chimeric_tools.Data import *



def test_load_truths():
    assert isinstance(load_cases_truths(), pd.DataFrame)
    assert isinstance(load_deaths_truths(), pd.DataFrame)
    assert isinstance(load_hosps_truths(), pd.DataFrame)


def test_load_weekly_covid():
    assert isinstance(load_cases_weekly(), pd.DataFrame)
    assert isinstance(load_deaths_weekly(), pd.DataFrame)
    assert isinstance(load_hosps_weekly(), pd.DataFrame)

def test_covid_class():
    assert isinstance(CovidData().data, pd.DataFrame)
    #test date ranges
    assert CovidData(start_date="2019-01-01", end_date="2022-01-02", geo_values="42095").start_date == date(2020,1,19)    