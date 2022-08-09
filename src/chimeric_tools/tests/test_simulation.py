from numpy import block
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
    assert isinstance(COVID(geo_values = "US").simulate(block_length = "auto", reps = 2), pd.DataFrame)
    assert isinstance(COVID(geo_values = ["US", "42"], include = ["cases"]).simulate(block_length = [5], reps = 2), pd.DataFrame)
    with pytest.raises(ValueError):
        COVID(geo_values = "US", include = ["cases"]).simulate(block_length = [1,2], reps = 2)
    assert isinstance(COVID(geo_values = "US").simulate(block_length = 5, reps = 2), pd.DataFrame)