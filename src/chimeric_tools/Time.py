# Written by:Wenxuan Ye in 2022/03/17

# Function name: fromDate2Epiweek
# input: a string in the format YYYY-mm-dd
# output: The Epidemic week that corresponds to YYYY-mm-dd
# info: We can use the epiweeks package to do this.
#
import epiweeks
import datetime
def fromDate2Epiweek(DateString):
    """
    Return the epidemic week that corresponds to a date in YYYY-mm-dd format

    :param: a string in the format YYYY-mm-dd
    :return: the epiweek
    :rtype: INT
    
    """
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
    """
    Return the epidemic weeks that corresponds to a list of dates in YYYY-mm-dd format

    :param: a list of strings in the format YYYY-mm-dd
    :return: a list of epiweeks
    :rtype: LIST
    """
    Epiweeks = []
    for Date in datelist:
        Epiweeks.append(fromDate2Epiweek(Date))
    return Epiweeks

# Function name: todayEpiWeek

# input: nothing
# output: the epidemic weeks that corresponds to todays date in YYYY-mm-dd format
# info: We can use the datetime package and epiweeks package to do this.
def todayEpiWeek():
    """
    Return the epidemic week that corresponds to today date in YYYY-mm-dd format

    :param: None
    :return: the epiweek
    :rtype: INT
    
    """
    Date = datetime.date.today()
    Epiweek = epiweeks.Week.fromdate(Date)
    return Epiweek._week