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

# Data
Influenza-like illness data:

Influenza-like illness (ILI) from U.S. Outpatient Influenza-like Illness Surveillance Network (ILINet).

The goal is to store all ILI values per epidemic week from 2009 to present in a compressed data file that allows fast I/O. 
ILI data can be downloaded with a python package that comes with an API(API where Data is stored = https://cmu-delphi.github.io/delphi-epidata/api/README.html). The data we need is in the API under influenza data and is the endpoint "fluview". 
We need to 
1. Download weighted ILI values (wILI) from the API above from 2009 to present. 
2. Format the wILI values into a dataframe with columns
   1. Season
       - The Infleunza season starts in epidemic week 40 of year YYYY and ends in epidemic week 20 of the follow year (YYYY+1). Season is a string value with one year a "/" and a second year (YYYY/YYYY) that depends on epidemic week. For example, If an epidemic week has year 2011 and any of the weeks 40 to 53 then the Season will be 2011/2012. If the week is 1-20 then the Season is 2010/2011.  
   3. Epidemic week 
       - The year and week YYYYWW of the associated wILI value
   5. HHS Region
       - There are 10 HHS regions and one national estimate of wILI. HHS regions are integers from 1-10. We can label the US as the integer 0.
   7. wILI value

Potential Data Storage techniques to explore: feather, parquet, hdf5. These appear to be designed for compressed storage and fast i/o. 

We may need to add a key called "package_data" to setup.py 
https://kiwidamien.github.io/making-a-python-package-vi-including-data-files.html









