"""
Script for Github to make predictions on the covid data
"""

import pandas as pd

from chimeric_tools.models import model
from chimeric_tools.Data import load_deaths_truths, daily_to_weekly

if __name__ == "__main__":

    # --load case data
    data = load_deaths_truths()

    # --convert to weekly data and save
    weekly = daily_to_weekly(data)
    weekly.to_csv(
        "../src/chimeric_tools/data/truh-Incident WeeklyDeaths.csv.gz", index=False, compression="gzip"
    )
    weekly = model(weekly)
    weekly.to_csv("../src/chimeric_tools/data/deaths_weekly.csv.gz", index=False, compression="gzip")
