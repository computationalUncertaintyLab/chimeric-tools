# Written by:Wenxuan Ye in 2022/05/07

from typing import Optional, Union, Iterable
from datetime import date


def crawl_data_to_feature_specific(startepiweek, endepiweek, regions):
    '''
    This function crawls data using the delphi epidata API from specific regions and stores it in a feature file.

    :param startepiweek: the first epiweek to crawl
    :param endepiweek: the last epiweek to crawl
    :param regions: the list of regions to crawl(e.g. ['nat','hhs1','hhs2','hhs3','hhs4','hhs5','hhs6','hhs7','hhs8','hhs9','hhs10'])

    :return: A dataframe of the data
    '''
    import pandas as pd
    from delphi_epidata import Epidata
    region_dic={
        'nat':0,
        'hhs1':1,
        'hhs2':2,
        'hhs3':3,
        'hhs4':4,
        'hhs5':5,
        'hhs6':6,
        'hhs7':7,
        'hhs8':8,
        'hhs9':9,
        'hhs10':10,

    }

    filename = regions[0]+'_'+str(startepiweek)+'_'+str(endepiweek)+'.feather'
    res = Epidata.fluview(regions, [Epidata.range(startepiweek, endepiweek)])
    epidata = res['epidata']
    df = pd.DataFrame(columns=['Season', 'Epidemic_week', 'HHS_region', 'wILI'])
    for i in epidata:   
        # print(i)
        epiWeek = str(i['epiweek'])
        Year = epiWeek[0:4]
        Week = epiWeek[4:6]
        if int(Week) <= 20:
            preYear = int(Year) - 1
            Season = str(preYear) + "/" + str(Year)
        elif int(Week) >= 40:
            Season = str(Year) + "/" + str(int(Year) + 1)
        else:
            continue
        df = df.append({'Season': Season, 'Epidemic_week': epiWeek, 'HHS_region': region_dic[i['region']], 'wILI': i['wili']}, ignore_index=True)
    df = df.sort_values(by=['Epidemic_week','HHS_region']).reset_index(drop=True)
    # store the data as feather
    df.to_feather(filename)
    return df

def crawl_data_to_feature_all(startepiweek, endepiweek):
    '''
    This function crawls data using the delphi epidata API from all regions and stores it in a feather file.

    :param startepiweek: the first epiweek to crawl
    :param endepiweek: the last epiweek to crawl   
    :return: A dataframe of all the data

    '''
    import pandas as pd
    from delphi_epidata import Epidata
    regions = ['nat','hhs1','hhs2','hhs3','hhs4','hhs5','hhs6','hhs7','hhs8','hhs9','hhs10']
    df = pd.DataFrame(columns=['Season', 'Epidemic_week', 'HHS_region', 'wILI'])
    for i in regions:
        df1 = crawl_data_to_feature_specific(startepiweek, endepiweek, [i])
        df = pd.concat([df, df1], ignore_index=True)
    df = df.sort_values(by=['Epidemic_week','HHS_region']).reset_index(drop=True)
    # store the data as feather
    df.to_feather('all_' + str(startepiweek) + '_' + str(endepiweek) + '.feather')
    return df



def randomly_select_fluseason(probobilility_dic,season_features_path):
    '''
    Randomly select a flu season based on the probability_dic, the value of the probability_dic should be sum to 1.

    :param probobilility_dic: the probability_dic
    :param season_features_path: the path to the season_features
    :return: The selected season DataFrame
    
    '''
    import random
    import numpy as np
    import pandas as pd
    features = pd.read_feather(season_features_path)

    if sum(probobilility_dic.values()) != 1:
        raise ValueError('probobilility_dic should sum to 1')
    else:       
        keys = np.array(list(probobilility_dic.keys()))
        values = np.array(list(probobilility_dic.values()))
        season = np.random.choice(keys, 1,p=values)[0]
        features = features[features['Season'] == season]
        return features

def random_generate_fluseason(startyear,endyear,features,regions=None):
    '''
    Randomly generate a flu season

    :param startyear: the start year of the flu season
    :param endyear: the end year of the flu season
    :param features: the features of the flu season
    :param regions: the list of regions to generate the flu season[0,1]
    :return: the dataframe generated flu season
    
    '''
    # choose the region from features
    import random
    import numpy as np
    import pandas as pd
    if regions is None:
        regions = [0,1,2,3,4,5,6,7,8,9,10]
    features = features[features['HHS_region'].isin(regions)]  
    year33 = [2009,2010,2011,2012,2013,2015,2016,2017,2018,2019,2021,2022,2023,2024]
    year34 = [2014,2020]
    year33_count = []
    year34_count = []
    for i in range(startyear,endyear+1):
        if i in year33:
            year33_count.append(i)
        elif i in year34:
            year34_count.append(i)
        else:
            continue
    year33_prob = len(year33_count)/(len(year33_count)+len(year34_count))
    year34_prob = len(year34_count)/(len(year33_count)+len(year34_count))
    print(year33_prob,year34_prob)
    random_year = np.random.choice([33,34],1,p=[year33_prob,year34_prob])[0]
    def get_season(is33year,yearlist,df):
        teamdf = pd.DataFrame(columns=['HHSRegion','EpidemicWeek','Season','wILI'])
        if is33year:
            epiweeks = np.array(list(range(1,21))+list(range(40,54)))
        else:
            epiweeks = np.array(list(range(1,21))+list(range(40,55)))
        for r in regions:
            df1 = df[df['HHS_region'] == r]
            for epiweek in epiweeks:
                yearweek = []
                for i in yearlist:
                    if epiweek < 10:
                        yearweek.append(str(i)+'0'+str(epiweek))
                    else:
                        yearweek.append(str(i)+str(epiweek))
                df2 = df1[df1['Epidemic_week'].isin(yearweek)]
                # randomly select a row from the dataframe
                if len(df2) == 0:
                    continue
                df2 = df2.reset_index(drop=True)
                
                row_index = random.randint(0,len(df2)-1)
                row = df2.iloc[row_index]
                teamdf = teamdf.append({'HHSRegion':r,'EpidemicWeek':epiweek,'Season':row['Season'],'wILI':row['wILI']},ignore_index=True)
        return teamdf



    if random_year == 33:
        df = get_season(True,year33_count,features)
    elif random_year == 34:
        df = get_season(False,year34_count,features)
    return df
    



def get_covid_data(geo_type: str, geo_values: Union[str, Iterable[str]], start_day: Optional[date], end_day: Optional[date]) -> None:
    '''
    Gets covid data from Delphi API

    :param geo_type: the type of the geo value
    :param geo_value: the value of the geo
    :param start_date: the start date of the data
    :param end_date: the end date of the data
    :return: the dataframe of the covid data
    
    '''

    import pandas as pd
    import covidcast

    # --checking inputs 
    if not(geo_type =='state' or geo_type == 'county'):
        raise Exception("geo_type must be 'state' or 'county'")
    if start_day == None:
        start_day = date(2020,1,22)



    data = covidcast.signal(data_source = "jhu-csse",
                            signal      = "confirmed_incidence_num",
                            geo_type    = geo_type,
                            geo_values  = geo_values,
                            start_day   = start_day,
                            end_day     = end_day)

    df = data[ ["time_value","geo_value","value"]  ]
    df = df.rename(columns={"geo_value":"location", "time_value":"date"})
    return df

def daily_to_weekly(data):
    '''
    Converts the daily data to weekly data
        
    df must be in format [ date, location, location_name,  value]
    '''
    import pandas as pd
    from epiweeks import Week
   
    unique_dates = data.date.unique()

    fromDate2EW = { "date":[], "start_date":[], "end_date":[], "EW":[] }
    for date in unique_dates:
        fromDate2EW["date"].append(date)

        dt = pd.to_datetime(date)
        week = Week.fromdate(dt)

        startdate = week.startdate()
        fromDate2EW["start_date"].append( startdate )

        enddate = week.enddate()
        fromDate2EW["end_date"].append( enddate )

        fromDate2EW["EW"].append( week.cdcformat() )
    fromDate2EW = pd.DataFrame(fromDate2EW)

    data = data.merge(fromDate2EW, on = ["date"])

    def aggregate(x):
        cases =  x.value.sum()

        return pd.Series({"cases":cases})
        
    weekly_date = data.groupby( ["location", "location_name", "start_date", "end_date", "EW"]).apply(aggregate)
    weekly_date.reset_index().to_feather("covid_cases.feather")