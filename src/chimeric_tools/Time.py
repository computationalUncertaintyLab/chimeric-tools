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


def getClosestDay(numericDay):
    """
    The user enters a value corresponding to the days of the week: Sunday (0)-Saturday(6)
    and this function returns the closest date YYYY-mm-dd with this day of the week. 

    :param: numericDay(0-6)
    :return: YYYY-mm-dd
    :rtype: str
    
    # Return the YYYY-MM-DD of the closest Wednesday
    getClosestDay(3)
    
    """
    
    import datetime
    from epiweeks import Week

    from datetime import datetime as dt
    today     = dt.today()

    weekAhead = today
    weekday   = today.weekday()

    while weekday != numericDay:
        weekAhead = weekAhead + datetime.timedelta(days=1)
        weekday = weekAhead.weekday()

    weekBehind = today
    weekday = today.weekday()
    while weekday != numericDay:
        weekBehind = weekBehind - datetime.timedelta(days=1)
        weekday = weekBehind.weekday()

    distance2weekahead  = abs( today - weekAhead)
    distance2weekbehind = abs( today - weekBehind)

    if distance2weekbehind < distance2weekahead:
        forecast_date = weekBehind.strftime("%Y-%m-%d")
        return weekBehind.strftime("%Y-%m-%d") 
    return weekAhead.strftime("%Y-%m-%d") 


