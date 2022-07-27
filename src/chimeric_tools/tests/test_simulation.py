import pytest
from chimeric_tools.Simulation import *

def test_data():
    assert isinstance(COVID().data, pd.DataFrame)
