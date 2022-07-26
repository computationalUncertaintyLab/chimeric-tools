import pytest
import pandas as pd
from chimeric_tools.Data import *



def test_load_truths():
    os.remove(os.path.dirname(__file__) + "/../data/truth-Incident Cases.csv.gz")
    os.remove(os.path.dirname(__file__) + "/../data/truth-Incident Deaths.csv.gz")
    os.remove(os.path.dirname(__file__) + "/../data/truth-Incident Hospitalizations.csv.gz")

    assert isinstance(load_cases_truths(), pd.DataFrame)
    assert isinstance(load_deaths_truths(), pd.DataFrame)
    assert isinstance(load_hosps_truths(), pd.DataFrame)


def test_load_weekly_covid():
    os.remove(os.path.dirname(__file__) + "/../data/cases_weekly.csv.gz")
    os.remove(os.path.dirname(__file__) + "/../data/deaths_weekly.csv.gz")
    os.remove(os.path.dirname(__file__) + "/../data/hosps_weekly.csv.gz")

    assert isinstance(load_cases_weekly(), pd.DataFrame)
    assert isinstance(load_deaths_weekly(), pd.DataFrame)
    assert isinstance(load_hosps_weekly(), pd.DataFrame)

def test_weekly():
    assert isinstance(daily_to_weekly(load_cases_truths()), pd.DataFrame)

def test_data():
    assert isinstance(CovidData().data, pd.DataFrame)
    
    
def test_date():
    assert min(CovidData(start_date="2019-01-01", end_date="2022-01-02").data["date"]) == date(2020,1,19)
    #assert max(CovidData(start_date="2019-01-01", end_date="2023-01-02").data["date"]) == date.today() - timedelta(days=1)
    assert min(CovidData(start_date="2022-01-01", end_date="2022-01-30").data["date"]) == Week.fromdate(date(2022,1,1)).startdate()
    assert max(CovidData(start_date="2022-01-01", end_date="2022-01-30").data["end_date"]) == Week.fromdate(date(2022,1,30)).enddate()
    assert min(CovidData(start_date=date(2022,1,1), end_date=date(2022,1,30)).data["date"]) == Week.fromdate(date(2022,1,1)).startdate()
    assert max(CovidData(start_date=date(2022,1,1), end_date=date(2022,1,30)).data["end_date"]) == Week.fromdate(date(2022,1,30)).enddate()

def test_data_sources():
    assert np.isin(["cases", "deaths", "hosps"], CovidData(include = ["cases", "deaths" , "hosps"]).data.columns.to_list()).all()
    assert np.isin(["cases", "deaths"], CovidData(include = ["cases", "deaths"]).data.columns.to_list()).all()
    assert np.isin(["hosps", "cases"], CovidData(include = ["cases", "hosps"]).data.columns.to_list()).all()
    assert np.isin(["deaths", "hosps"], CovidData(include = ["deaths", "hosps"]).data.columns.to_list()).all()
    assert np.isin(["cases"], CovidData(include = ["cases"]).data.columns.to_list()).all()

def test_geo_values():
    assert np.isin(["US"], CovidData(geo_values = "US").data.location.unique()).all()
    assert np.isin(["US", "42"], CovidData(geo_values = np.array(["US", "42"])).data.location.unique()).all()
    assert np.isin(["US", "42", "42095"], CovidData(geo_values = ["US", "42", "42095"]).data.location.unique()).all()

def test_custom_data():
    assert isinstance(CovidData(custom_data = load_cases_truths()).data, pd.DataFrame)