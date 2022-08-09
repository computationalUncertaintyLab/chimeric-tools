#!/bin/sh
echo "Unzipping Files"
gzip -d ./src/chimeric_tools/data/truth-Incident\ Cases.csv.gz
gzip -d ./src/chimeric_tools/data/truth-Incident\ Deaths.csv.gz
gzip -d ./src/chimeric_tools/data/truth-Incident\ Hospitalizations.csv.gz

echo "Downloading Files"
Rscript scripts/download_truths.R

echo "Zipping Files"
gzip ./src/chimeric_tools/data/truth-Incident\ Cases.csv
gzip ./src/chimeric_tools/data/truth-Incident\ Deaths.csv
gzip ./src/chimeric_tools/data/truth-Incident\ Hospitalizations.csv

echo "Case Predictions"
python scripts/cases_preds.py

echo "Death Predictions"
python scripts/deaths_preds.py

echo "Hospitalization Predictions"
python scripts/hosps_preds.py
