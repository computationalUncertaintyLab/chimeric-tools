"""
Script for Github to make predictions on the covid data
"""

import pandas as pd

from chimeric_tools.models import model

if __name__=="__main__":
    data = pd.read_csv("../src/chimeric_tools/data/truth-Incident Cases.csv")
    to_cat = model(data)
    to_cat.to_csv("../src/chimeric_tools/data/covid.csv", index=False)