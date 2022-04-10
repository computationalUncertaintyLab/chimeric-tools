import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# Written in 2022/3/16 By Wenxuan Ye

# formatIndividualPredictions
# input: 
#   1.the path of single csv file of individual predictions from metaculus
#   2.the path of the parameter file
# output: a pandas date frame that is long format
# info: Wenxuan, this is your code that transforms the csv file to long.
def formatIndividualPredictions(data_path, para_path):
    data = pd.read_csv(data_path,delimiter=";")
    para = pd.read_csv(para_path)
    # rename the columns name
    para["question_id"]=para["qid"]
    para=para.drop(["qid"],axis=1)
    data1 = data.merge(para, on="question_id")
    # extact the values where the user_id =112705
    usedata = data1
    # add a new column with the revision_number
    usedata["revision_number"]=0
    # modify the revision_id based on the question_id and the user_id
    test_userdata = usedata.groupby(["question_id","user_id"])
    # create a new empty dataframe
    newdata = pd.DataFrame()
    for i in test_userdata:
        i[1].loc[:,"revision_number"]=[j for j in range(1,len(i[1])+1)]
        newdata=newdata.append(i[1])
    cases=[]
    for i in range(1,100):
        cases.append(newdata["b"]*np.exp(newdata["exponent"]*i/100))
    # add 100 columns to the dataframe
    for i in range(0,101):
        newdata["Case_r"+str(i/100)]=0
        newdata["Case_r"+str(i/100)]=newdata["b"]*np.exp(newdata["exponent"]*i/100)
    outputdata=newdata
    outputdata["prediction_time"]=outputdata["time"]
    outputdata=outputdata.drop(["time"],axis=1)
    outputdata=outputdata.drop(["void"],axis=1)
    outputdata=outputdata.drop(["question_type"],axis=1)
    outputdata=outputdata.drop(["resolve_time"],axis=1)
    outputdata=outputdata.drop(["close_time"],axis=1)
    outputdata=outputdata.drop(["binary_prediction"],axis=1)
    outputdata=outputdata.drop(["exponent"],axis=1)
    outputdata=outputdata.drop(["b"],axis=1)
    outputdata=outputdata.drop(["resolution"],axis=1)
    # rename the columns
    # reorder the specified columns the keep the rest in the same order
    outputdata=outputdata[['question_id',
    'user_id',
    'revision_number',
    'prediction_time',
    'Case_r0.0',
    'Case_r0.01',
    'Case_r0.02',
    'Case_r0.03',
    'Case_r0.04',
    'Case_r0.05',
    'Case_r0.06',
    'Case_r0.07',
    'Case_r0.08',
    'Case_r0.09',
    'Case_r0.1',
    'Case_r0.11',
    'Case_r0.12',
    'Case_r0.13',
    'Case_r0.14',
    'Case_r0.15',
    'Case_r0.16',
    'Case_r0.17',
    'Case_r0.18',
    'Case_r0.19',
    'Case_r0.2',
    'Case_r0.21',
    'Case_r0.22',
    'Case_r0.23',
    'Case_r0.24',
    'Case_r0.25',
    'Case_r0.26',
    'Case_r0.27',
    'Case_r0.28',
    'Case_r0.29',
    'Case_r0.3',
    'Case_r0.31',
    'Case_r0.32',
    'Case_r0.33',
    'Case_r0.34',
    'Case_r0.35',
    'Case_r0.36',
    'Case_r0.37',
    'Case_r0.38',
    'Case_r0.39',
    'Case_r0.4',
    'Case_r0.41',
    'Case_r0.42',
    'Case_r0.43',
    'Case_r0.44',
    'Case_r0.45',
    'Case_r0.46',
    'Case_r0.47',
    'Case_r0.48',
    'Case_r0.49',
    'Case_r0.5',
    'Case_r0.51',
    'Case_r0.52',
    'Case_r0.53',
    'Case_r0.54',
    'Case_r0.55',
    'Case_r0.56',
    'Case_r0.57',
    'Case_r0.58',
    'Case_r0.59',
    'Case_r0.6',
    'Case_r0.61',
    'Case_r0.62',
    'Case_r0.63',
    'Case_r0.64',
    'Case_r0.65',
    'Case_r0.66',
    'Case_r0.67',
    'Case_r0.68',
    'Case_r0.69',
    'Case_r0.7',
    'Case_r0.71',
    'Case_r0.72',
    'Case_r0.73',
    'Case_r0.74',
    'Case_r0.75',
    'Case_r0.76',
    'Case_r0.77',
    'Case_r0.78',
    'Case_r0.79',
    'Case_r0.8',
    'Case_r0.81',
    'Case_r0.82',
    'Case_r0.83',
    'Case_r0.84',
    'Case_r0.85',
    'Case_r0.86',
    'Case_r0.87',
    'Case_r0.88',
    'Case_r0.89',
    'Case_r0.9',
    'Case_r0.91',
    'Case_r0.92',
    'Case_r0.93',
    'Case_r0.94',
    'Case_r0.95',
    'Case_r0.96',
    'Case_r0.97',
    'Case_r0.98',
    'Case_r0.99',
    'Case_r1.0',
    'PDF(r=0.00)',
    'PDF(r=0.01)',
    'PDF(r=0.02)',
    'PDF(r=0.03)',
    'PDF(r=0.04)',
    'PDF(r=0.05)',
    'PDF(r=0.06)',
    'PDF(r=0.07)',
    'PDF(r=0.08)',
    'PDF(r=0.09)',
    'PDF(r=0.10)',
    'PDF(r=0.11)',
    'PDF(r=0.12)',
    'PDF(r=0.13)',
    'PDF(r=0.14)',
    'PDF(r=0.15)',
    'PDF(r=0.16)',
    'PDF(r=0.17)',
    'PDF(r=0.18)',
    'PDF(r=0.19)',
    'PDF(r=0.20)',
    'PDF(r=0.21)',
    'PDF(r=0.22)',
    'PDF(r=0.23)',
    'PDF(r=0.24)',
    'PDF(r=0.25)',
    'PDF(r=0.26)',
    'PDF(r=0.27)',
    'PDF(r=0.28)',
    'PDF(r=0.29)',
    'PDF(r=0.30)',
    'PDF(r=0.31)',
    'PDF(r=0.32)',
    'PDF(r=0.33)',
    'PDF(r=0.34)',
    'PDF(r=0.35)',
    'PDF(r=0.36)',
    'PDF(r=0.37)',
    'PDF(r=0.38)',
    'PDF(r=0.39)',
    'PDF(r=0.40)',
    'PDF(r=0.41)',
    'PDF(r=0.42)',
    'PDF(r=0.43)',
    'PDF(r=0.44)',
    'PDF(r=0.45)',
    'PDF(r=0.46)',
    'PDF(r=0.47)',
    'PDF(r=0.48)',
    'PDF(r=0.49)',
    'PDF(r=0.50)',
    'PDF(r=0.51)',
    'PDF(r=0.52)',
    'PDF(r=0.53)',
    'PDF(r=0.54)',
    'PDF(r=0.55)',
    'PDF(r=0.56)',
    'PDF(r=0.57)',
    'PDF(r=0.58)',
    'PDF(r=0.59)',
    'PDF(r=0.60)',
    'PDF(r=0.61)',
    'PDF(r=0.62)',
    'PDF(r=0.63)',
    'PDF(r=0.64)',
    'PDF(r=0.65)',
    'PDF(r=0.66)',
    'PDF(r=0.67)',
    'PDF(r=0.68)',
    'PDF(r=0.69)',
    'PDF(r=0.70)',
    'PDF(r=0.71)',
    'PDF(r=0.72)',
    'PDF(r=0.73)',
    'PDF(r=0.74)',
    'PDF(r=0.75)',
    'PDF(r=0.76)',
    'PDF(r=0.77)',
    'PDF(r=0.78)',
    'PDF(r=0.79)',
    'PDF(r=0.80)',
    'PDF(r=0.81)',
    'PDF(r=0.82)',
    'PDF(r=0.83)',
    'PDF(r=0.84)',
    'PDF(r=0.85)',
    'PDF(r=0.86)',
    'PDF(r=0.87)',
    'PDF(r=0.88)',
    'PDF(r=0.89)',
    'PDF(r=0.90)',
    'PDF(r=0.91)',
    'PDF(r=0.92)',
    'PDF(r=0.93)',
    'PDF(r=0.94)',
    'PDF(r=0.95)',
    'PDF(r=0.96)',
    'PDF(r=0.97)',
    'PDF(r=0.98)',
    'PDF(r=0.99)',
    'PDF(r=1.00)',
    ]]
    new_outputdata=pd.DataFrame(columns=['question_id','user_id','revision_number','prediction_time','case','density'])
    for i in range(0,len(outputdata)):
        for j in range(0,101):
        # add a new row in new_outputdata
            new_outputdata.loc[len(new_outputdata)] = [outputdata.iloc[i,0],outputdata.iloc[i,1],outputdata.iloc[i,2],outputdata.iloc[i,3],outputdata.iloc[i,4+j],outputdata.iloc[i,105+j]]
    new_outputdata.to_csv("allIndividualPredictions.csv",index=False)



