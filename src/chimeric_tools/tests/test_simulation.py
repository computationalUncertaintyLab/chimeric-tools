import pytest
from chimeric_tools.Simulation import *

def test_data():
    assert isinstance(COVID().data, pd.DataFrame)

def test_geo_values():
    assert isinstance(COVID(geo_values="US").data, pd.DataFrame)
    assert isinstance(COVID(geo_values=["US"]).data, pd.DataFrame)
    assert isinstance(COVID(geo_values=np.array(["US", "42"])).data, pd.DataFrame)
    with pytest.raises(ValueError):
        COVID(geo_values={"US":0.1})
    assert isinstance(COVID(geo_values={"US":1}).data, pd.DataFrame)

def test_pick_geo_values():
    assert isinstance(COVID(geo_values="US").pick_geo_values(5), np.ndarray)
    assert isinstance(COVID(geo_values=np.array(["US", "42"])).pick_geo_values(5), np.ndarray)
    assert isinstance(COVID(geo_values={"US":0.3, "42":0.7}).pick_geo_values(1), np.ndarray)

def test_generator():
    assert np.array_equal(COVID(geo_values=["US","42"], seed = 1).pick_geo_values(5), COVID(geo_values=["US","42"], seed = 1).pick_geo_values(5))
    gen = np.random.default_rng(1)
    gen1 = np.random.default_rng(1)
    assert np.array_equal(COVID(geo_values=["US","42"], seed = gen).pick_geo_values(5), COVID(geo_values=["US","42"], seed = gen1).pick_geo_values(5))
    with pytest.raises(TypeError):
        COVID(seed = "1")

def test_simulation():
    bs = COVID()
    length = len(bs.data.loc[bs.data["location"] == "US"])
    for data in bs.simulate(1):
        cases = data[0]["cases"]
        deaths = data[0]["deaths"]
        hosps = data[0]["hosps"]
        assert isinstance(cases, pd.Series)
        assert isinstance(deaths, pd.Series)
        assert isinstance(hosps, pd.Series)
        assert cases.shape == (length,)
        assert deaths.shape == (length,)
        assert hosps.shape == (length,)