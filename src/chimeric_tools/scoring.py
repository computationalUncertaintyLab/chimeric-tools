
import pandas as pd
import numpy as np
import scipy.stats as stat
# group the predictions by multiple columns
class Score:
    def __init__(self,pred,truth):
        self.preds = pred
        self.truth = truth
        self.preds.rename(columns={"forecast_date":"date"},inplace=True)
        self.preds_combined_with_truth = pd.merge(self.preds,self.truth)
    def log_score(self):
        def cal_log(mean,val,quantile,true):
            # print(mean,val,quantile,true)
            simg = (val- mean)/stat.norm.ppf(quantile)
            return np.log(stat.norm.pdf(true,mean,simg))
        self.preds_combined_with_truth["logscore"] = 0
        # convert the negative values to 0
        self.preds_combined_with_truth["cases"] = self.preds_combined_with_truth["cases"].apply(lambda x: 0 if x<0 else x)
        self.preds_combined_with_truth['value'] = self.preds_combined_with_truth['value'].apply(lambda x: 0 if x<0 else x)
        new_preds = pd.DataFrame()
        for subdf in self.preds_combined_with_truth.groupby(['sim', 'model','target_end_date','date']):
            # subdf is a tuple (key, sub_dataframe)
            key, subdf = subdf
            # choose the quantile 0.5
            mean_val = subdf[subdf['quantile']==0.5]['value'].values[0]
            # add one column to the subdf
            # calculate the log score
            subdf['logscore']=subdf.apply(lambda x: cal_log(mean_val,x['value'],x['quantile'],x['cases']),axis=1)    
            # update the subdf to the preds_new
            new_preds = new_preds.append(subdf)
        return new_preds
    def Weighted_Interval_Score():
        pass




    