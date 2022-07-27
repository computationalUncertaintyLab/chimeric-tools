# chimeric-tools

Chimeric-Tools provides a set of tools to help influenza like illness forecast modeling. 

The main purpose of this package is to provide a score for ensemble models of influenza like illnesses. When comparing different ensemble models between different scientific papers there are vast differences in data and base model’s used to train the ensemble models, thus creating incomparable results.

Current Working Features:
- COVID-19 data for cases, deaths, and hosps
- Simulation of COVID-19 data

Features Coming Soon:
- Baseline forecast models

## Installation

For right now we suggest installing from GitHub because of the rapid development of this project

### pip

Releases are available PyPI and can be installed with `pip`.


```shell
pip install chimeric-tools
```

You can alternatively install the latest version from GitHub

```bash
pip install git+https://github.com/computationalUncertaintyLab/chimeric-tools.git
```


## Documentation

Read the release documentation [here](https://computationaluncertaintylab.github.io/chimeric-tools/).


## Examples

### Covid Data

```python
from chimeric_tools.Data import CovidData
data = CovidData(start_date = "2021-01-01", end_date = "2021-12-31", geo_values = "US", include = ["cases", "deaths", "hosps"]).data
data.head()
```

### Covid Simulation

```python
from chimeric_tools.Simulation import COVID
import matplotlib.pyplot as plt
bs = COVID(start_date = "2021-01-01", end_date = "2021-12-31", geo_values = "US", include = ["cases"])
for data in bs.simulate(100):
    plt.plot(data[0]["cases"], color= "tab:blue", alpha = 0.1)
plt.xlabel("Weeks")
plt.ylabel("Cases")
plt.title("Covid Simulation of FIP: " + data[1])
plt.show()
```



## Contributions

Contributions are welcome. There are opportunities at many levels to contribute:

- Implement new process, e.g., different bootstrap method
- Improve docstrings where unclear or with typos
- Provide examples, preferably in the form of IPython notebooks