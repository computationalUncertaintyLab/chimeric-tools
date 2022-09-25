
import pandas as pd
import numpy as np
import scipy.stats as stat
# group the predictions by multiple columns
def cal_log(mean,val,quantile,true):
    # print(mean,val,quantile,true)
    simg = (val- mean)/quantile
    return np.log(stat.norm.pdf(true,mean,simg))
def log_score(preds_new):
    preds_new["logscore"] = 0
    # convert the negative values to 0
    preds_new["cases"] = preds_new["cases"].apply(lambda x: 0 if x<0 else x)
    preds_new['value'] = preds_new['value'].apply(lambda x: 0 if x<0 else x)
    new_preds = pd.DataFrame()
    for subdf in preds_new.groupby(['sim', 'model','target_end_date','date']):
        # subdf is a tuple (key, sub_dataframe)
        key, subdf = subdf
        # choose the quantile 0.5
        mean_val = subdf[subdf['quantile']==0.5]['value'].values[0]
        # add one column to the subdf
        # calculate the log score
        subdf['logsore']=subdf.apply(lambda x: cal_log(mean_val,x['value'],x['quantile'],x['cases']),axis=1)    
        # update the subdf to the preds_new
        new_preds = new_preds.append(subdf)
        new_preds.to_csv("log.csv")




    