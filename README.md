# chimeric-tools

# Data folder

## functions

### Preprocess folder
- Function name: formatIndividualPredictions
  - input: a single csv file of individual predictions from metaculus
  - output: a pandas daat frame that is long format
  - info: Wenxuan, this is your code that transforms the csv file to long. 

- Function name: write_individual_formatted
  - input: a pandas data frame in lojng format
  - output: nothing
  - info: Wenxuan, this will output the formatted pandas data frame to a file with the name "WW-YYYY-MM-DD_metaculus_individual_predictions.csv.gz" where WW is the epidemic week, YYYY is the year, mm is month, and dd is day. 

- Function name: (for prof m to add) metaculus_client
  - input: string that links to user credentials
  - output: object to interact with API
  - info: prof m will write this and Wenuxan and Xinze will test how well it does (or does not) work

### spatial folder

- Function Name: fromState2FIPS
  - input: String abbrevation of a state. For example "PA"
  - output: The FIPS value that corresponds to the state (For example the FIPS for PA is 42)
  -  info: I think this URL will be useful https://www.nrcs.usda.gov/wps/portal/nrcs/detail/national/home/?cid=nrcs143_013697

### Time Folder

## functions

- Function name: fromDate2Epiweek
  - input:  a string in the format YYYY-mm-dd
  - output: The Epidemic week that corresponds to YYYY-mm-dd
  - info: We can use the epiweeks package to do this.

- Function name: fromDates2Epiweeks
  - input:  a list of strings in the format YYYY-mm-dd
  - output: The Epidemic weeks that corresponds to YYYY-mm-dd
  - info: We can use the epiweeks package to do this.

- Function name: today
  - input:  nothing
  - output: todays date in YYYY-mm-dd format
  - info: We can use the datetime package to do this.

- Function name: todayEpiWeek
  - input:  nothing
  - output: the epidemic weeks that corresponds to todays date in YYYY-mm-dd format
  - info: We can use the datetime package and epiweeks package to do this.

# Aggregation


