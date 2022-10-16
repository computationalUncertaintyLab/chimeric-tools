import pytest
import pandas as pd
from chimeric_tools.DataFlu import *

def test_dates():
	# start date tests
	flu = flu_data(start_date=date(2010, 12, 6))
	assert len(flu.loc[flu["date"] < date(2010, 12, 6)]) <= 12
	flu2 = flu_data(start_date="2010-12-06")
	assert flu.equals(flu2)

	# end date tests
	flu = flu_data(end_date=date(2018, 2, 3))
	assert len(flu.loc[flu["end_date"] > date(2018, 2, 3)]) == 0
	flu2 = flu_data(end_date="2018-02-03")
	assert flu.equals(flu2)

	with pytest.warns(Warning):
		flu_data(start_date=date(2010, 5, 5))
		flu_data(end_date=date(2020, 1, 1))
	
def test_geo_values():
	assert np.isin(["nat"], flu_data(geo_values="nat").location.unique()).all()
	assert np.isin(["nat", "hhs1"], flu_data(geo_values=np.array["nat", "hhs1"]).location.unique()).all()
	assert np.isin(["nat", "hhs1", "hhs10"], flu_data(geo_values=["nat", "hhs1", "hhs10"]).location.unique()).all()
	with pytest.raises(ValueError):
		flu_data(geo_values=42)
