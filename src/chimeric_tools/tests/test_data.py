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
    data = load_cases_truths()
    data = data.loc[data["location"].isin(["US", "42"])]
    assert isinstance(daily_to_weekly(data), pd.DataFrame)

def test_data():
    assert isinstance(covid_data(), pd.DataFrame)
    
    
def test_date():
    assert min(covid_data(start_date="2019-01-01", end_date="2022-01-02")["date"]) == date(2020,1,19)
    #assert max(covid_data(start_date="2019-01-01", end_date="2023-01-02")["date"]) == date.today() - timedelta(days=1)
    assert min(covid_data(start_date="2022-01-01", end_date="2022-01-30")["date"]) == Week.fromdate(date(2022,1,1)).startdate()
    assert max(covid_data(start_date="2022-01-01", end_date="2022-01-30")["end_date"]) == Week.fromdate(date(2022,1,30)).enddate()
    assert min(covid_data(start_date=date(2022,1,1), end_date=date(2022,1,30))["date"]) == Week.fromdate(date(2022,1,1)).startdate()
    assert max(covid_data(start_date=date(2022,1,1), end_date=date(2022,1,30))["end_date"]) == Week.fromdate(date(2022,1,30)).enddate()

def test_data_sources():
    assert np.isin(["cases", "deaths", "hosps"], covid_data(include = ["cases", "deaths" , "hosps"]).columns.to_list()).all()
    assert np.isin(["cases", "deaths"], covid_data(include = ["cases", "deaths"]).columns.to_list()).all()
    assert np.isin(["hosps", "cases"], covid_data(include = ["cases", "hosps"]).columns.to_list()).all()
    assert np.isin(["deaths", "hosps"], covid_data(include = ["deaths", "hosps"]).columns.to_list()).all()
    assert np.isin(["cases"], covid_data(include = ["cases"]).columns.to_list()).all()

def test_geo_values():
    assert np.isin(["US"], covid_data(geo_values = "US").location.unique()).all()
    assert np.isin(["US", "42"], covid_data(geo_values = np.array(["US", "42"])).location.unique()).all()
    assert np.isin(["US", "42", "42095"], covid_data(geo_values = ["US", "42", "42095"]).location.unique()).all()