"""
Script for Github to make predictions on the covid data
"""

import pandas as pd

from chimeric_tools.models import model
from chimeric_tools.data import daily_to_weekly, load_truths

if __name__ == "__main__":
    data = load_truths()
    # to_cat = model(data)
    # to_cat.to_csv("../src/chimeric_tools/data/daily_covid.csv", index=False)

    weekly = daily_to_weekly(data)
    weekly.to_csv(
        "../src/chimeric_tools/data/truh-Incident WeeklyCases.csv", index=False, compression="gzip"
    )
    weekly = model(weekly)
    weekly.to_csv("../src/chimeric_tools/data/weekly_covid.csv", index=False, compression="gzip")
