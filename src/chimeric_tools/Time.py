# Written by:Wenxuan Ye in 2022/03/17

# Function name: fromDate2Epiweek
# input: a string in the format YYYY-mm-dd
# output: The Epidemic week that corresponds to YYYY-mm-dd
# info: We can use the epiweeks package to do this.
#
import epiweeks
import datetime
def fromDate2Epiweek(DateString):
    Date = DateString.split("-")
    Year = int(Date[0])
    Month = int(Date[1])
    Day = int(Date[2])
    Date = datetime.date(Year,Month,Day)
    Epiweek = epiweeks.Week.fromdate(Date)
    # print()
    return Epiweek._week


# Function name: fromDates2Epiweeks
# input: a list of strings in the format YYYY-mm-dd
# output: The Epidemic weeks that corresponds to YYYY-mm-dd
# info: We can use the epiweeks package to do this.
def fromDates2Epiweeks(datelist):
    Epiweeks = []
    for Date in datelist:
        Epiweeks.append(fromDate2Epiweek(Date))
    return Epiweeks

# Function name: todayEpiWeek

# input: nothing
# output: the epidemic weeks that corresponds to todays date in YYYY-mm-dd format
# info: We can use the datetime package and epiweeks package to do this.
def todayEpiWeek():
    Date = datetime.date.today()
    # print(Date)
    Epiweek = epiweeks.Week.fromdate(Date)
    return Epiweek._week