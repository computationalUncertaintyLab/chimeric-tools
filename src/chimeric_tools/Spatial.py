# Fuction name: GetStateAbbreviation
# input: StateName
# output: StateAbbreviation
# info: 
from chimeric_tools.Constant import *
def GetStateAbbreviation(StateName):
    """
    Return the abbreviation of a state name

    :param: StateName
    :return: StateAbbreviation
    :rtype: STRING

    """
    dict = State_Abbreviation_Dict
    if StateName in dict:
        return dict[StateName]
    else:
        return "Not Found"

# Function Name: fromState2FIPS
# input: String abbrevation of a state. For example "PA"
# output: The FIPS value that corresponds to the state (For example the FIPS for PA is 42)
# info:https://www.nrcs.usda.gov/wps/portal/nrcs/detail/national/home/?cid=nrcs143_013697

def fromState2FIPS(StateString):
    """
    Return the FIPS value that corresponds to the state (For example the FIPS for PA is 42)

    :param: String abbrevation of a state. For example "PA"
    :return: The FIPS value that corresponds to the state (For example the FIPS for PA is 42)
    :rtype: INT
    
    """
    dict = State_Abbreviation_to_FIPS_Dict
    if StateString in dict:
        return dict[StateString]
    else:
        return "Not Found"


