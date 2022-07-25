#!/bin/sh
echo "Unzipping Files"
gzip -d ../src/chimeric_tools/data/truth-Incident\ Cases.csv.gz
gzip -d ../src/chimeric_tools/data/truth-Incident\ Deaths.csv.gz
gzip -d ../src/chimeric_tools/data/truth-Incident\ Hospitalizations.csv.gz

echo "Downloading Files"
Rscript download_truths.R

echo "Zipping Files"
gzip ../src/chimeric_tools/data/truth-Incident\ Cases.csv
gzip ../src/chimeric_tools/data/truth-Incident\ Deaths.csv
gzip ../src/chimeric_tools/data/truth-Incident\ Hospitalizations.csv

echo "Case Predictions"
python cases_preds.py

echo "Death Predictions"
python deaths_preds.py

echo "Hospitalization Predictions"
python hosps_preds.py
