# chimeric-tools
[![codecov](https://codecov.io/gh/computationalUncertaintyLab/chimeric-tools/branch/main/graph/badge.svg)](https://codecov.io/gh/computationalUncertaintyLab/chimeric-tools)
[![Covid Data Update](https://github.com/computationalUncertaintyLab/chimeric-tools/actions/workflows/update_covid_data.yml/badge.svg)](https://github.com/computationalUncertaintyLab/chimeric-tools/actions/workflows/update_covid_data.yml)
[![Deploy to GitHub Pages](https://github.com/computationalUncertaintyLab/chimeric-tools/actions/workflows/main.yml/badge.svg)](https://github.com/computationalUncertaintyLab/chimeric-tools/actions/workflows/main.yml)


Chimeric-Tools aims to rigorous test ensemble algorithms, algorithms that map a set of individual forecasts into a single forecast, of the trajectory of an infectious agent by (i) generating simulated outbreaks that depend on past surveillance data, (ii) training a set of individual models to produce forecasts of this simulated outbreak, (iii) applying the proposed ensemble algorithm to aggregate these individual models, and (iv) scoring the ensemble algorithm. 
Past work has tested novel ensemble algorithms on empirical data which makes it difficult to extract statistical properties of the newly proposed ensemble algorithm and compare two ensembles unless they are trained on the same dataset and incorporate the same individual models into their ensemble.
The Chimeric-Tools package allows one to compare the performance of two or more different ensemble algorithms on the same surveillance data and on the same set of individual models. 
Chimeric-tools offers infectious disease modelers a testbed for ensemble algorithms in an effort to more quickly improve ensemble algorithms that generate forecasts of a pathogen.  


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
from chimeric_tools.Data import covid_data
data = covid_data(start_date = "2021-01-01", end_date = "2021-12-31", geo_values = "US", include = ["cases", "deaths", "hosps"])
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
