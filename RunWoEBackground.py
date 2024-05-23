import numpy as np
import pandas as pd
from pandas import Series
from pandas import DataFrame
import pickle
from o3_Run_WoE_AnalysisAndTrainModel import MIV,test2015, fileName, inputData, chars, binners, charsAnalsysis, replaceWoe, runWoEAnalysis, replaceWoe, correlationsAnalysis, col_list, trainModel, modelDevelopment, scaleModel, loc
from matplotlib import pyplot

import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
import warnings
import statsmodels.api as sm
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.datasets import make_classification
from datetime import datetime

def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))
       

ninf = float('-inf')
pinf = float('+inf')  

def render_iv_table(data, col_width=3.0, row_height=0.625, font_size=14.5,
                     header_color='#33B3FF', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')
    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in mpl_table._cells.items():
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    return ax.get_figure(), ax

def self_bin_char(Y,X):
   good = Y.sum()
   bad = Y.count()-good
   d1=pd.DataFrame({'X':X,'Y':Y,chars[z]:X})
   d2=d1.groupby([chars[z]],dropna=False)
   d3=pd.DataFrame(d2['X'].min(),columns=['min'])
   
   d3['# Good']=d2['Y'].sum()
   d3['# Bad']=d2['Y'].count()-d3['# Good']
   d3['% Good'] = round(d3['# Good']/d3['# Good'].sum()*100,1)
   d3['% Bad'] = round(d3['# Bad']/d3['# Bad'].sum()*100,1)
   d3['# Total']=d2['Y'].count()
   d3['% Total'] = round(d3['# Total']/d3['# Total'].sum()*100,1)
   
   d3['Information Odds'] = round(d3['% Good'] / d3['% Bad'],2)
   d3['Bad Rate'] = round(d3['# Bad']/(d3['# Bad']+d3['# Good'])*100,2)
   
   d3['WoE'] = round(np.log(d3['% Good']/d3['% Bad']),2)
   iv = ((d3['% Good']-d3['% Bad'])*d3['WoE']/100)
   d4=d3.sort_index()
   d4 = d3.drop(columns=['min'], axis=1)
   woe = list(d3['WoE'].values)
   
   return d4,iv,woe
   
def self_bin(Y,X,cat):
    d1=pd.DataFrame({'X':X,'Y':Y,chars[z]:pd.cut(X,cat)})
    d2=d1.groupby([chars[z]],dropna=False)
   
    d3=pd.DataFrame(d2['X'].min(),columns=['min'])
     
    d3['# Good']=d2['Y'].sum()
    d3['# Bad']=d2['Y'].count()-d3['# Good']
    d3['% Good'] = round(d3['# Good']/d3['# Good'].sum()*100,1)
    d3['% Bad'] = round(d3['# Bad']/d3['# Bad'].sum()*100,1)
    d3['# Total']=d2['Y'].count()
    d3['% Total'] = round(d3['# Total']/d3['# Total'].sum()*100,1)

    d3['Information Odds'] = round(d3['% Good'] / d3['% Bad'],2)
    d3['Bad Rate'] = round(d3['# Bad']/(d3['# Bad']+d3['# Good'])*100,2)
   
    d3['WoE'] = round(np.log(d3['% Good']/d3['% Bad']),2)
    #print(d3)
    iv = ((d3['% Good']-d3['% Bad'])*d3['WoE']/100)
    d4=d3.sort_index()
    d4 = d3.drop(columns=['min'], axis=1)
    woe = list(d3['WoE'].values)
   
    return d4,iv,woe    

def replace_woe(series,cut,woe):
    list=[]
    i=0
    while i<len(series):
        valuek=series[i]
        j=len(cut)-2
        m=len(cut)-2
        while j>=0:
            if valuek>cut[j]:
                j=-1
            else:
                j -=1
                m -= 1
        list.append(woe[m])
        i += 1
    return list
   
   
def replace_woe_str(series,cut,woe):
    list=[]
    i=0
    while i<len(series):
        valuek=series[i]
        #print(valuek)
        index = cut.index(valuek)
        #print(index)
        list.append(woe[index])
        #print(list.append(woe[index]))
        i += 1
    return list

d = pd.DataFrame(index = [0],columns=['Characteristics','Information Value'])
d.to_csv(charsAnalsysis+'dd.csv')        

if runWoEAnalysis:
    print("#################################################################")
    print("#                                                               #")
    print("#                     WOE ANALYSIS                              #")
    print("#                                                               #")
    print("#################################################################")
   
    print("")
    print("Start WOE Analysis.....")
    print("")
   
   
    loc = inputData  
    d= pd.read_csv(loc+fileName)#, index_col=0
    d = d.loc[(d['Target'] < 2)]
   
    print(d.Target.value_counts(dropna=False))
   
   
    woeList = []
    cutList = []
    for z in range(len(chars)):
   
        #print(d[chars[z]])
        if(d[chars[z]].dtype !='object'):
           
           
            print("WOE analysis for this char: ",chars[z])
            loc = inputData
           
            df= pd.read_csv(loc+fileName)#, index_col=0
            #df = df.drop(columns=['Unnamed: 0'],axis=1)

            cutx3 = binners[z]
            #print(binners[z])
            dfx3, ivx3, woex3 = self_bin(df['Target'],df[chars[z]], cutx3)
            dfx3 = dfx3.reset_index()
       
               
           
           
            dfx3.to_csv(loc+'dfx3.csv')  
           
           
            a = pd.read_csv(charsAnalsysis+'dd.csv',index_col=0)
            dd = pd.DataFrame({'Characteristics':chars[z],'Information Value':"%.3f" %round(ivx3.sum(),3)}, index=[0])
            s = pd.concat([a, dd])
            s = s.dropna()
            s.to_csv(charsAnalsysis+'dd.csv')
           
           
            df = pd.read_csv(loc+'dfx3.csv')#, index_col=0
            df = df.drop(columns=['Unnamed: 0'],axis=1)
            df.loc['Total', '# Bad']= df['# Bad'].sum(axis=0)
            df.loc['Total', '# Good']= df['# Good'].sum(axis=0)
            df.loc['Total', '# Total']= df['# Total'].sum(axis=0)
            df.loc['Total', '% Total']= round(df['% Total'].sum(axis=0))
            df.loc['Total', '% Bad']= round(df['% Bad'].sum(axis=0))
            df.loc['Total', '% Good']= round(df['% Good'].sum(axis=0))
           
           
            df.iloc[len(df)-1, df.columns.get_loc(chars[z])] = 'Total'
            df.iloc[len(df)-1, df.columns.get_loc('Information Odds')] ="%.2f" % round(df['# Good'].sum() / (df['# Bad'].sum()),2)
            df.iloc[len(df)-1, df.columns.get_loc('Bad Rate')] = "%.2f" %round(df['# Bad'].sum()*100 / (df['# Good'].sum() + df['# Bad'].sum()),1)
            df.iloc[len(df)-1, df.columns.get_loc('WoE')] = ''
           
           
            df['# Bad'] = df['# Bad'].apply(lambda x : "{:,}".format(x))
            df['# Good'] = df['# Good'].apply(lambda x : "{:,}".format(x))
            df['# Total'] = df['# Total'].apply(lambda x : "{:,}".format(x))
            df.loc[''] = ''                                                                      
            df.loc['Inf. Value'] = 'Inf. Value:'                                                                      
           
            df.iloc[len(df)-1, df.columns.get_loc(chars[z])] = 'Inf. Value: '
            df.iloc[len(df)-1, df.columns.get_loc('# Bad')] = ''
            df.iloc[len(df)-1, df.columns.get_loc('# Good')] = "%.3f" %round(ivx3.sum(),3)
            df.iloc[len(df)-1, df.columns.get_loc('# Total')] = ''
            df.iloc[len(df)-1, df.columns.get_loc('% Total')] = ''
            df.iloc[len(df)-1, df.columns.get_loc('% Bad')] =''
            df.iloc[len(df)-1, df.columns.get_loc('% Good')] = ''
            df.iloc[len(df)-1, df.columns.get_loc('Information Odds')] = ''
            df.iloc[len(df)-1, df.columns.get_loc('Bad Rate')] = ''
            df.iloc[len(df)-1, df.columns.get_loc('WoE')] = ''
           
           
            def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=14.5,
                                 header_color='#33B3FF', row_colors=['#f1f1f2', 'w'], edge_color='w',
                                 bbox=[0, 0, 1, 1], header_columns=0,
                                 ax=None, **kwargs):
                if ax is None:
                    size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
                    fig, ax = plt.subplots(figsize=size)
                    ax.axis('off')
                mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)
                mpl_table.auto_set_font_size(False)
                mpl_table.set_fontsize(font_size)
           
                for k, cell in mpl_table._cells.items():
                    cell.set_edgecolor(edge_color)
                    if k[0] == 0 or k[1] < header_columns:
                        cell.set_text_props(weight='bold', color='w')
                        cell.set_facecolor(header_color)
                    else:
                        cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
                return ax.get_figure(), ax
           
            woeList.append(woex3)
            cutList.append(cutx3)
           
        else:
            loc = inputData
           
            print("WOE analysis for this char: ",chars[z])
            df= pd.read_csv(loc+fileName)
            #df = df.drop(columns=['Unnamed: 0'],axis=1)
            #df[chars[z]] = df[chars[z]].fillna('Missing')
            #binners[z]
            df[chars[z]+'_bin'] = (
                    df[chars[z]]
                    .apply(lambda x: [k for k in binners[z].keys() if x in binners[z][k]])
                    .str[0])
                   

            dfx3, ivx3, woex3 = self_bin_char(df['Target'],df[chars[z]+'_bin'])
            dfx3 = dfx3.reset_index()
            cutx3 = list((dfx3[chars[z]]))
               
 
           
            dfx3.to_csv(loc+'dfx3.csv')  
            #print(chars[z]+" - IV: ",round(ivx3.sum(),3))
           
            a = pd.read_csv(charsAnalsysis+'dd.csv',index_col=0)
            dd = pd.DataFrame({'Characteristics':chars[z],'Information Value':"%.3f" %round(ivx3.sum(),3)}, index=[0])
            s = pd.concat([a, dd])
            s = s.dropna()
            s.to_csv(charsAnalsysis+'dd.csv')
           
           
            df = pd.read_csv(loc+'dfx3.csv', index_col=0)
           
            df.loc['Total', '# Bad']= df['# Bad'].sum(axis=0)
            df.loc['Total', '# Good']= df['# Good'].sum(axis=0)
            df.loc['Total', '# Total']= df['# Total'].sum(axis=0)
            df.loc['Total', '% Total']= round(df['% Total'].sum(axis=0))
            df.loc['Total', '% Bad']= round(df['% Bad'].sum(axis=0))
            df.loc['Total', '% Good']= round(df['% Good'].sum(axis=0))
           
           
            df.iloc[len(df)-1, df.columns.get_loc(chars[z])] = 'Total'
            df.iloc[len(df)-1, df.columns.get_loc('Information Odds')] ="%.2f" % round(df['# Good'].sum() / (df['# Bad'].sum()),2)
            df.iloc[len(df)-1, df.columns.get_loc('Bad Rate')] = "%.2f" %round(df['# Bad'].sum()*100 / (df['# Good'].sum() + df['# Bad'].sum()),1)
            df.iloc[len(df)-1, df.columns.get_loc('WoE')] = ''
           
           
            df['# Bad'] = df['# Bad'].apply(lambda x : "{:,}".format(x))
            df['# Good'] = df['# Good'].apply(lambda x : "{:,}".format(x))
            df['# Total'] = df['# Total'].apply(lambda x : "{:,}".format(x))
            df.loc[''] = ''                                                                      
            df.loc['Information Value'] = 'Inf. Value:'                                                                      
           
            df.iloc[len(df)-1, df.columns.get_loc(chars[z])] = 'Information Value: '
            df.iloc[len(df)-1, df.columns.get_loc('# Bad')] = ''
            df.iloc[len(df)-1, df.columns.get_loc('# Good')] = "%.3f" %round(ivx3.sum(),3)
            df.iloc[len(df)-1, df.columns.get_loc('# Total')] = ''
            df.iloc[len(df)-1, df.columns.get_loc('% Total')] = ''
            df.iloc[len(df)-1, df.columns.get_loc('% Bad')] =''
            df.iloc[len(df)-1, df.columns.get_loc('% Good')] = ''
            df.iloc[len(df)-1, df.columns.get_loc('Information Odds')] = ''
            df.iloc[len(df)-1, df.columns.get_loc('Bad Rate')] = ''
            df.iloc[len(df)-1, df.columns.get_loc('WoE')] = ''
           
           
            def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=14.5,
                                 header_color='#33B3FF', row_colors=['#f1f1f2', 'w'], edge_color='w',
                                 bbox=[0, 0, 1, 1], header_columns=0,
                                 ax=None, **kwargs):
                if ax is None:
                    size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
                    fig, ax = plt.subplots(figsize=size)
                    ax.axis('off')
                mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)
                mpl_table.auto_set_font_size(False)
                mpl_table.set_fontsize(font_size)
           
                for k, cell in mpl_table._cells.items():
                    cell.set_edgecolor(edge_color)
                    if k[0] == 0 or k[1] < header_columns:
                        cell.set_text_props(weight='bold', color='w')
                        cell.set_facecolor(header_color)
                    else:
                        cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
                return ax.get_figure(), ax
           
   
            woeList.append(woex3)
            cutList.append(cutx3)
       

           
        fig,ax = render_mpl_table(df, header_columns=0, col_width=2)
        fig.savefig(charsAnalsysis+chars[z]+"_WOE.png")
        df.to_csv(charsAnalsysis+chars[z]+'_WOE.csv')      
    print("")
    print(" WOE Analysis Completed!")
    print("")

     
       
     
       
     
       
if replaceWoe:    
   
    print("#################################################################")
    print("#                                                               #")
    print("#              REPLACE WOE                                      #")
    print("#                                                               #")
    print("#################################################################")
   
    loc = inputData
    data = pd.read_csv(loc+fileName)

   
    print("")
    print("Start Replacing WOE in the input datset.....")
    print("")

   
    for z in range(len(chars)):
       
        print("Replace chars Value with WoE: ",chars[z])
        if(data[chars[z]].dtype !='object'):
            data[chars[z]] = Series(replace_woe(data[chars[z]], cutList[z], woeList[z]))
        else:
            data[chars[z]] = data[chars[z]].fillna('Missing')
            data[chars[z]+'_bin'] = (
                    data[chars[z]]
                    .apply(lambda x: [k for k in binners[z].keys() if x in binners[z][k]])
                    .str[0])
                       
            data[chars[z]] = Series(replace_woe_str(data[chars[z]+"_bin"], cutList[z], woeList[z]))
    print("Replace chars value with WoE: completed .....")
    data.to_csv(loc+'train_for_development_WOE.csv')

    d = pd.read_csv(charsAnalsysis+'dd.csv')
    d = d.sort_values(by=['Information Value'], ascending=False)
    d = d.drop(columns=['Unnamed: 0'],axis=1)
    fig,ax = render_iv_table(d, header_columns=0, col_width=7, fontsize = 15)
    fig.savefig(charsAnalsysis+"_FeaturesInformationValue.png")#,  
    print("")
    print("Replacing WOE compelted!")
    print("")
   
   
if correlationsAnalysis:
    print("#################################################################")
    print("#                                                               #")
    print("#              CORRELATION ANALYSIS                             #")
    print("#                                                               #")
    print("#################################################################")
   
    print("")
    print("Start correlation analysis..")
    data = pd.read_csv(loc+'train_for_development_WOE.csv',index_col=0)
    data = data[chars]
    corr=data.corr()
    corr.to_csv(charsAnalsysis+'correlationAnalysisReport.csv')
    xticks = [col_list]
    fig=plt.figure()
    fig.set_size_inches(30,30)
    ax1=fig.add_subplot(1,1,1)
    sns.heatmap(corr,vmin=-1, vmax=1 ,cmap='hsv', annot=True, square=True)
    #ax1.set_xticklabels(xticks,rotation= 0, fontsize = 22)
    #plt.xticks(ticks=X_Tick_List,labels=X_Tick_LabeL_List, rotation=25,fontsize=8)
   
    plt.show()    
    print("")
    print("Correlation analysis completed!")
    print("")
   
if trainModel:


    print("#################################################################")
    print("#                                                               #")
    print("#                       TRAIN MODEL                             #")
    print("#                                                               #")
    print("#################################################################")
   
    data = pd.read_csv(loc+'train_for_development_WOE.csv',index_col=0)
   
   
    data = data[['Target','grade','emp_length','dti','Orig_FicoScore','inq_last_6mths','acc_open_past_24mths','mort_acc','mths_since_recent_bc','num_rev_tl_bal_gt_0','percent_bc_gt_75']]
   
   
       
   
   
    Y=data['Target']
    X=data[col_list]
   
   
    X1=sm.add_constant(X)
       
    logit=sm.Logit(Y,X1)
    result=logit.fit()
    print(result.summary())
   
    plt.rcParams['font.sans-serif'] = ['FangSong']        # Specify default font
    plt.rcParams['axes.unicode_minus'] = False
    Y_test=data['Target']
    X_test=data[col_list]
   
    X2=sm.add_constant(X_test)
    resu=result.predict(X2)
    fpr,tpr,threshold=roc_curve(Y_test,resu)
    rocauc=auc(fpr,tpr)*100
    x = round((rocauc-50)*2,1)
   
   
   
    print("                 ")
    print("Train Gini:  ",x)
   
    print("                 ")
   
   
   
    fig = plt.figure()    
    fig.suptitle('Train Sample TOC')
    ax1 = fig.add_subplot(1,1,1)
    plt.plot(fpr,tpr,'b',label='ROC: %0.1f'% rocauc)
    plt.plot(fpr,tpr,'b',label='Gini: %0.1f'% x)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlabel('% Cumulative Good')
    plt.ylabel('% Cumulative Bad')
    plt.savefig(modelDevelopment+'_TrainSample_TOC.png')
    plt.show()
   
 
   
    from sklearn.linear_model import LogisticRegression
    from sklearn2pmml import sklearn2pmml
    from sklearn2pmml.pipeline import PMMLPipeline



   
    logreg = LogisticRegression()
    # set up a pmml pipeline to fit the model
    pipeline = PMMLPipeline([
        ("classifier", LogisticRegression())
    ])
    pipeline.fit(X2, Y)

    # save the pmml model to disk

    sklearn2pmml(pipeline, "LogisticRegression_model.pmml", with_repr = True)
# =============================================================================
#     import shap
#     import sklearn
#
#     model = sklearn.linear_model.LogisticRegression(penalty="l2", C=0.1)
#     model.fit(X2, Y_test)
# =============================================================================
# =============================================================================
#     LogisticRegression(C=0.1, class_weight=None, dual=False, fit_intercept=True,
#               intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
#               penalty='l1', random_state=None, solver='liblinear', tol=0.0001,
#               verbose=0, warm_start=False)
# =============================================================================

    explainer = shap.LinearExplainer(model, X2, feature_dependence="independent")
   
       
   
    #sklearn2pmml(pipeline, "MyArchivedLogRegModel.pmml")
   
    Pkl_Filename = modelDevelopment+"Trained_Model.pkl"  
   
    with open(Pkl_Filename, 'wb') as file:  
        pickle.dump(logit, file)
   
   
    data['Probability'] = resu
    def Prediction(row):
        if(row['Probability'] >= 0.50):
            return 1
        else:
            return 0
    data['Prediction'] = data.apply(Prediction, axis=1)
   
    print("Average probabiliyt of being Good: ",str(round(data['Probability'].mean()*100,1)))  
    data.to_csv(loc+'input.csv')        
    x = pd.crosstab(data['Probability'],data['Target'])        
    x.to_csv(loc+'scoreDistribution.csv')
    data = data['Probability']
    data = data.reset_index()
   
   
    d = pd.read_csv(loc+'train_for_development_WOE.csv', index_col=0)
    d = d.reset_index()
   
   
    df = d.merge(data, left_index=True, right_index=True)
   
   
    df = df.drop(columns=['index_x','index_y'], axis=1)
   
    df.to_csv(modelDevelopment+'MIV.csv')
   
   
    print("")
    print("Training Model: completed !")
    print("")

   
if MIV:
    print("#################################################################")
    print("#                                                               #")
    print("#                         MIV ANALYSIS                          #")
    print("#                                                               #")
    print("#################################################################")
   
    d = pd.DataFrame(index = [0],columns=['Characteristics','MIV'])
    d.to_csv(loc+'ee.csv')  
   
   
    d= pd.read_csv(modelDevelopment+'MIV.csv')#, index_col=0
    d = d.drop(columns=['Unnamed: 0'],axis=1)
   
   
   
    for z in range(len(chars)):
        if(chars[z] not in col_list):
            d= pd.read_csv(modelDevelopment+'MIV.csv')#, index_col=0
            d = d.drop(columns=['Unnamed: 0'],axis=1)
            if(d[chars[z]].dtype !='object'):    
                df= pd.read_csv(modelDevelopment+'MIV.csv')#, index_col=0
                df = df.drop(columns=['Unnamed: 0'],axis=1)
               
                df[chars[z]+'_bin'] = pd.cut(x=df[chars[z]], bins=binners[z])
                col = ['prob',chars[z]+'_bin']
                df = df[col]
                dfx3 = df.groupby(chars[z]+'_bin').mean()
                dfx3 = dfx3.reset_index()
                dfx3 = dfx3.drop(columns=[chars[z]+'_bin'],axis=1)
                d = pd.read_csv(charsAnalsysis+chars[z]+'.csv')
                d = d[:-3]
                data = d.merge(dfx3, left_index=True, right_index=True)
                data = data.drop(columns=['Unnamed: 0'],axis=1)    
                eg = []
               
                words = np.array(data['# Total'])
                #print(words)
                d = []
                for string in words:
                    #d.append(0)
                    newString = str(string).replace(',','')
                   
                    d.append(newString)
                   
                data['t'] =d
                #a = data['t'].astype(float)
                t = np.array(data['t'].astype(float))
                p = np.array(data['prob'])
                for n in range(len(t)):
                    eg.append(0)
                    eg[n] = round(float(t[n])*p[n],1)
                data['exp_Goods'] = eg
 
               
                eb = []
               
                p = np.array(data['prob'])
                for n in range(len(t)):
                    eb.append(0)
                    eb[n] = round(float(t[n])*(1-p[n]),1)
                data['exp_Bads'] = eb
               
                data = data.drop(columns=['t'],axis=1)
                data['exp_pct_Goods'] = round(data['exp_Goods']/data['exp_Goods'].sum()*100,1)
                data['exp_pct_Bads'] = round(data['exp_Bads']/data['exp_Bads'].sum()*100,1)
                data['exp_WoE'] = round(np.log(data['exp_pct_Goods']/data['exp_pct_Bads']),2)          
                data['DeltaScore'] = data['WoE'] - data['exp_WoE']

                data['MIV'] = data['DeltaScore']*data['% Good'] - data['DeltaScore']*data['% Bad']
                x = round(data['MIV'].sum(),2)
                data.to_csv(loc+chars[z]+'delta.csv')
               #print(chars[z]+': '+str(x))
                a = pd.read_csv(loc+'ee.csv',index_col=0)
                dd = pd.DataFrame({'Characteristics':chars[z],'MIV':"%.3f" %round(x,2)}, index=[0])
                s = pd.concat([a, dd])
                s = s.dropna()
                s.to_csv(loc+'ee.csv')            
               
               
               
            else:
                df= pd.read_csv(modelDevelopment+'MIV.csv')#, index_col=0
                df = df.drop(columns=['Unnamed: 0'],axis=1)
                df[chars[z]+'_bin'] = df[chars[z]]
                col = ['prob',chars[z]+'_bin']
                df = df[col]
                dfx3 = df.groupby(chars[z]+'_bin').mean()
                dfx3 = dfx3.reset_index()
                dfx3 = dfx3.drop(columns=[chars[z]+'_bin'],axis=1)
                d = pd.read_csv(charsAnalsysis+chars[z]+'.csv')
                d = d[:-3]
                data = d.merge(dfx3, left_index=True, right_index=True)
                data = data.drop(columns=['Unnamed: 0'],axis=1)    
                eg = []
               
                words = np.array(data['# Total'])
                d = []
                for string in words:
                    #d.append(0)
                    newString = str(string).replace(',','')
                    d.append(newString)
                   
                data['t'] =d
                #a = data['t'].astype(float)
                t = np.array(data['t'].astype(float))
                p = np.array(data['prob'])
                for n in range(len(t)):
                    eg.append(0)
                    eg[n] = round(float(t[n])*p[n],1)
                data['exp_Goods'] = eg

                eb = []
               
                p = np.array(data['prob'])
                for n in range(len(t)):
                    eb.append(0)
                    eb[n] = round(float(t[n])*(1-p[n]),1)
                data['exp_Bads'] = eb
               
                data = data.drop(columns=['t'],axis=1)
                data['exp_pct_Goods'] = round(data['exp_Goods']/data['exp_Goods'].sum()*100,1)
                data['exp_pct_Bads'] = round(data['exp_Bads']/data['exp_Bads'].sum()*100,1)
                data['exp_WoE'] = round(np.log(data['exp_pct_Goods']/data['exp_pct_Bads']),2)          
                data['DeltaScore'] = data['WoE'] - data['exp_WoE']

                data['MIV'] = data['DeltaScore']*data['% Good'] - data['DeltaScore']*data['% Bad']
                x = round(data['MIV'].sum(),2)
                data.to_csv(loc+chars[z]+'delta.csv')
                #print(chars[z]+': '+str(x))      
                a = pd.read_csv(loc+'ee.csv',index_col=0)
                dd = pd.DataFrame({'Characteristics':chars[z],'MIV':"%.3f" %round(x,2)}, index=[0])
                s = pd.concat([a, dd])
                s = s.dropna()
                s.to_csv(loc+'ee.csv')                  
    d = pd.read_csv(loc+'ee.csv')
    d = d.sort_values(by=['MIV'], ascending=False)
    d = d.drop(columns=['Unnamed: 0'],axis=1)
    fig,ax = render_iv_table(d, header_columns=0, col_width=7, fontsize = 15)
    fig.savefig("MIV_.png")#,

   
   
if scaleModel:
   
# =============================================================================
#         #################################################################################
#    
#         #        CHANGE THIS PART
#        
#         ################################################################################
#    
#    
#         print("#################################################################")
#         print("#                                                               #")
#         print("#                   SCALE THE MODEL                             #")
#         print("#                                                               #")
#         print("# Score: 660                                                    #")
#         print("# Odds: 15:1                                                    #")
#         print("# PDO: 40                                                       #")
#         print("#                                                               #")
#         print("#################################################################")
#    
#     # =============================================================================
#     #
# # =============================================================================
# # ========================================================================================
# #                            coef    std err          z      P>|z|      [0.025      0.975]
# # ----------------------------------------------------------------------------------------
# # const                    1.6079      0.004    387.797      0.000       1.600       1.616
# # grade                    0.9010      0.008    111.523      0.000       0.885       0.917
# # emp_length               0.9479      0.046     20.427      0.000       0.857       1.039
# # dti                      0.5118      0.019     26.629      0.000       0.474       0.549
# # Orig_FicoScore           0.0794      0.016      5.075      0.000       0.049       0.110
# # inq_last_6mths           0.2056      0.027      7.723      0.000       0.153       0.258
# # acc_open_past_24mths     0.4851      0.020     24.016      0.000       0.445       0.525
# # mort_acc                 0.6678      0.029     23.427      0.000       0.612       0.724
# # mths_since_recent_bc     0.1967      0.032      6.185      0.000       0.134       0.259
# # num_rev_tl_bal_gt_0      0.1295      0.030      4.274      0.000       0.070       0.189
# # percent_bc_gt_75         0.0459      0.026      1.766      0.077      -0.005       0.097
# # ========================================================================================
# # =============================================================================
#
#    
#    
#    
#         coe = [1.6079,0.9010,0.9479,0.5118,0.0794,0.2056,0.4851,0.6678,0.1967,0.1295,0.0459] # List of features coefficients
#    
#         import math
#        
#         score = 660
#         odds = 15
#         pdo = 40
#        
#         Factor = pdo/math.log(2)
#         Offset = score - (Factor*math.log(15))
#         print("")
#         print("Scaling Factor: ",str(Factor))
#         print("Scaling Offest: ",str(Offset))
#         print("")
#         data = pd.read_csv(loc+'development.csv')# input data without WOE values
#        # data = data.drop(columns=['Unnamed: 0'],axis=1)
#    
#         for z in range(len(chars)):
#            
#    
#             if(data[chars[z]].dtype !='object'):
#                 data[chars[z]] = Series(replace_woe(data[chars[z]], cutList[z], woeList[z]))
#             else:
#                 data[chars[z]+'_bin'] = (
#                         data[chars[z]]
#                         .apply(lambda x: [k for k in binners[z].keys() if x in binners[z][k]])
#                         .str[0])
#                 data[chars[z]] = Series(replace_woe_str(data[chars[z]+"_bin"], cutList[z], woeList[z]))
#    
#         data.to_csv(loc+'developmentDataset_with_WOE.csv')
#        
#         data['x1']  = round((data['grade']*coe[1]+(coe[0]/10))*Factor+(Offset/10),0)
#         data['x2']  = round((data['emp_length']*coe[2]+(coe[0]/10))*Factor+(Offset/10),0)
#         data['x3']  = round((data['dti']*coe[3]+(coe[0]/10))*Factor+(Offset/10),0)
#         data['x4']  = round((data['Orig_FicoScore']*coe[4]+(coe[0]/10))*Factor+(Offset/10),0)
#         data['x5']  = round((data['inq_last_6mths']*coe[5]+(coe[0]/10))*Factor+(Offset/10),0)
#         data['x6']  = round((data['acc_open_past_24mths']*coe[6]+(coe[0]/10))*Factor+(Offset/10),0)
#         data['x7']  = round((data['mort_acc']*coe[7]+(coe[0]/10))*Factor+(Offset/10),0)
#         data['x8']  = round((data['mths_since_recent_bc']*coe[8]+(coe[0]/10))*Factor+(Offset/10),0)
#         data['x9']  = round((data['num_rev_tl_bal_gt_0']*coe[9]+(coe[0]/10))*Factor+(Offset/10),0)
#         data['x10'] = round((data['percent_bc_gt_75']*coe[10]+(coe[0]/10))*Factor+(Offset/10),0)
#    
#         data['Score'] = data['x1']+    data['x2']+    data['x3']+    data['x4']+    data['x5']+    data['x6']+    data['x7']+data['x8']+data['x9']+data['x10']
#         #print(data)
#         data.to_csv(loc+'_development_scored_data.csv')
#         print("")
#         print("Average Score of Population: ", str(round(data['Score'].mean(),1)))
#         g = data.loc[data['Target'] == 1]
#         print("Average Score of Goods: ", str(round(g['Score'].mean(),1)))
#         g = data.loc[data['Target'] == 0]
#         print("Average Score of Bads: ", str(g['Score'].mean()))
#         g = data.loc[data['Target'] == -1]
#         print("Average Score of Indeterminates: ", str(round(g['Score'].mean(),1)))
#    
#         print("")
#         print("Minimum Score: ",str(data['Score'].min()))
#         print("Average Score: ",str(data['Score'].mean()))
#         print("Maximum Score: ",str(data['Score'].max()))
#         print("")
#    
#         data['ScoreBinned']=pd.cut(data['Score'], bins=[0,300,310,320,330,340,350,360,370,380,390,400,410,420,430,440,450,460,470,480,490,500,510,520,530,540,550,560,570,580,590,600,610,620,630,640,650,660,670,680,690,700,710,720,730,740,750,760,770,780,790,800,900])
#         x = pd.crosstab(data['ScoreBinned'],data['Target'])        
#        
#         x.to_csv(modelDevelopment+'Development_scaledScoreDistribution.csv')
#         print("The scaled score distribution has been saved here: ",modelDevelopment+'scaledScoreDistribution.csv')
#         print("")
#         print("Development Scaling completed!")
# =============================================================================
   
        print("#################################################################")
        print("#                                                               #")
        print("#                   EVALUATE 2015                               #")
        print("#                                                               #")
        print("#################################################################")  
   
   
   
   
   
        data = pd.read_csv(loc+'2015.csv')# input data without WOE values
        #data = data.drop(columns=['Unnamed: 0'],axis=1)
        data = data[['Target','grade','emp_length','dti','Orig_FicoScore','inq_last_6mths','acc_open_past_24mths','mort_acc','mths_since_recent_bc','num_rev_tl_bal_gt_0','percent_bc_gt_75']]
        for z in range(len(chars)):
   
            if(data[chars[z]].dtype !='object'):
                data[chars[z]] = Series(replace_woe(data[chars[z]], cutList[z], woeList[z]))
            else:
                data[chars[z]+'_bin'] = (
                        data[chars[z]]
                        .apply(lambda x: [k for k in binners[z].keys() if x in binners[z][k]])
                        .str[0])
                data[chars[z]] = Series(replace_woe_str(data[chars[z]+"_bin"], cutList[z], woeList[z]))
   
        Y_test=data['Target']
        X_test=data[col_list]
       
       
        #model = pickle.load(open(modelDevelopment+"Trained_Model.pkl", 'rb'))
       
       
        X2=sm.add_constant(X_test)
        resu=result.predict(X2)
        fpr,tpr,threshold=roc_curve(Y_test,resu)
        rocauc=auc(fpr,tpr)*100
        x = round((rocauc-50)*2,1)
       
       
       
        print("                 ")
        print("Generalisation Gini:  ",x)
       
        print("                 ")
           
        data['Probability'] = resu
   
        def Prediction(row):
            if(row['Probability'] > 0.50):
                return 1
            else:
                return 0
       
        data['Prediction'] = data.apply(Prediction, axis=1)
       
        #data.to_csv(loc+'va')
# =============================================================================
#         shap_values = explainer.shap_values(X2)
#         shap.summary_plot(shap_values, X2)
#
# =============================================================================
# =============================================================================
#         print("Import pmml")
#         from pypmml import Model
#
#         model = Model.fromFile("LogisticRegression_model.pmml")
#        
#         resu=model.predict(X2)
#         fpr,tpr,threshold=roc_curve(Y_test,resu)
#         rocauc=auc(fpr,tpr)*100
#         x = round((rocauc-50)*2,1)
#        
#        
#        
#         print("                 ")
#         print("Generalisation Gini pmml:  ",x)
#        
#         print("                 ")
# =============================================================================

if False:        
       
        from sklearn.metrics import f1_score
       
        actual = np.array(data['Target'])
        pred = np.array(data['Prediction'])
        print("F1 Score: ",str(f1_score(actual, pred)))

        from sklearn.metrics import confusion_matrix

        print(confusion_matrix(actual, pred))


        import shap
        # load the model from disk    
        #model = pickle.load(open(filename, 'rb'))    
   
        from pypmml import Model

        model = Model.fromFile("LogisticRegression_model.pmml")
       



if False:
        # Variable Importance - Global Interpretability
        shap_values = shap.Explainer(model, masker=shap.maskers.Impute(data=X2),
                           feature_names=X2.columns, algorithm="linear")
       
        #shap_values = shap.LinearExplainer(result).shap_values(X)
        shap.summary_plot(shap_values, X, plot_type="bar")
   
   
        # positive and negative relationships of the predictors with the target variable
        shap.summary_plot(shap_values, X)  


       
        print("Average probabiliyt of being Good: ",str(round(data['Probability'].mean()*100,1)))  
        data.to_csv(loc+'generalisatioon.csv')      
   
   
if False:    
        data.to_csv(loc+'developmentDataset_with_WOE.csv')
       
        data['x1']  = round((data['grade']*coe[1]+(coe[0]/10))*Factor+(Offset/10),0)
        data['x2']  = round((data['emp_length']*coe[2]+(coe[0]/10))*Factor+(Offset/10),0)
        data['x3']  = round((data['dti']*coe[3]+(coe[0]/10))*Factor+(Offset/10),0)
        data['x4']  = round((data['Orig_FicoScore']*coe[4]+(coe[0]/10))*Factor+(Offset/10),0)
        data['x5']  = round((data['inq_last_6mths']*coe[5]+(coe[0]/10))*Factor+(Offset/10),0)
        data['x6']  = round((data['acc_open_past_24mths']*coe[6]+(coe[0]/10))*Factor+(Offset/10),0)
        data['x7']  = round((data['mort_acc']*coe[7]+(coe[0]/10))*Factor+(Offset/10),0)
        data['x8']  = round((data['mths_since_recent_bc']*coe[8]+(coe[0]/10))*Factor+(Offset/10),0)
        data['x9']  = round((data['num_rev_tl_bal_gt_0']*coe[9]+(coe[0]/10))*Factor+(Offset/10),0)
        data['x10'] = round((data['percent_bc_gt_75']*coe[10]+(coe[0]/10))*Factor+(Offset/10),0)
   
        data['Score'] = data['x1']+    data['x2']+    data['x3']+    data['x4']+    data['x5']+    data['x6']+    data['x7']+data['x8']+data['x9']+data['x10']
        #print(data)
        data.to_csv(loc+'_development_scored_data.csv')
        print("")
        print("Average Score of Population: ", str(round(data['Score'].mean(),1)))
        g = data.loc[data['Target'] == 1]
        print("Average Score of Goods: ", str(round(g['Score'].mean(),1)))
        g = data.loc[data['Target'] == 0]
        print("Average Score of Bads: ", str(g['Score'].mean()))
        g = data.loc[data['Target'] == -1]
        print("Average Score of Indeterminates: ", str(round(g['Score'].mean(),1)))
   
        print("")
        print("Minimum Score: ",str(data['Score'].min()))
        print("Average Score: ",str(data['Score'].mean()))
        print("Maximum Score: ",str(data['Score'].max()))
        print("")
   
        data['ScoreBinned']=pd.cut(data['Score'], bins=[0,300,310,320,330,340,350,360,370,380,390,400,410,420,430,440,450,460,470,480,490,500,510,520,530,540,550,560,570,580,590,600,610,620,630,640,650,660,670,680,690,700,710,720,730,740,750,760,770,780,790,800,900])
        x = pd.crosstab(data['ScoreBinned'],data['Target'])        
       
        x.to_csv(modelDevelopment+'2015_scaledScoreDistribution.csv')
        print("The scaled score distribution has been saved here: ",modelDevelopment+'scaledScoreDistribution.csv')
        print("")
        print("Development Scaling completed!")    

if False:    
        print("#################################################################")
        print("#                                                               #")
        print("#                   EVALUATE 2016                               #")
        print("#                                                               #")
        print("#################################################################")  
   
   
   
   
   
        data = pd.read_csv(loc+'2016.csv')# input data without WOE values
       # data = data.drop(columns=['Unnamed: 0'],axis=1)
   
        for z in range(len(chars)):
            if(data[chars[z]].dtype !='object'):
                data[chars[z]] = Series(replace_woe(data[chars[z]], cutList[z], woeList[z]))
            else:
                data[chars[z]+'_bin'] = (
                        data[chars[z]]
                        .apply(lambda x: [k for k in binners[z].keys() if x in binners[z][k]])
                        .str[0])
                data[chars[z]] = Series(replace_woe_str(data[chars[z]+"_bin"], cutList[z], woeList[z]))
   
        data.to_csv(loc+'developmentDataset_with_WOE.csv')
       
        data['x1']  = round((data['grade']*coe[1]+(coe[0]/10))*Factor+(Offset/10),0)
        data['x2']  = round((data['emp_length']*coe[2]+(coe[0]/10))*Factor+(Offset/10),0)
        data['x3']  = round((data['dti']*coe[3]+(coe[0]/10))*Factor+(Offset/10),0)
        data['x4']  = round((data['Orig_FicoScore']*coe[4]+(coe[0]/10))*Factor+(Offset/10),0)
        data['x5']  = round((data['inq_last_6mths']*coe[5]+(coe[0]/10))*Factor+(Offset/10),0)
        data['x6']  = round((data['acc_open_past_24mths']*coe[6]+(coe[0]/10))*Factor+(Offset/10),0)
        data['x7']  = round((data['mort_acc']*coe[7]+(coe[0]/10))*Factor+(Offset/10),0)
        data['x8']  = round((data['mths_since_recent_bc']*coe[8]+(coe[0]/10))*Factor+(Offset/10),0)
        data['x9']  = round((data['num_rev_tl_bal_gt_0']*coe[9]+(coe[0]/10))*Factor+(Offset/10),0)
        data['x10'] = round((data['percent_bc_gt_75']*coe[10]+(coe[0]/10))*Factor+(Offset/10),0)
   
        data['Score'] = data['x1']+    data['x2']+    data['x3']+    data['x4']+    data['x5']+    data['x6']+    data['x7']+data['x8']+data['x9']+data['x10']
        #print(data)
        data.to_csv(loc+'_development_scored_data.csv')
        print("")
        print("Average Score of Population: ", str(round(data['Score'].mean(),1)))
        g = data.loc[data['Target'] == 1]
        print("Average Score of Goods: ", str(round(g['Score'].mean(),1)))
        g = data.loc[data['Target'] == 0]
        print("Average Score of Bads: ", str(g['Score'].mean()))
        g = data.loc[data['Target'] == -1]
        print("Average Score of Indeterminates: ", str(round(g['Score'].mean(),1)))
   
        print("")
        print("Minimum Score: ",str(data['Score'].min()))
        print("Average Score: ",str(data['Score'].mean()))
        print("Maximum Score: ",str(data['Score'].max()))
        print("")
   
        data['ScoreBinned']=pd.cut(data['Score'], bins=[0,300,310,320,330,340,350,360,370,380,390,400,410,420,430,440,450,460,470,480,490,500,510,520,530,540,550,560,570,580,590,600,610,620,630,640,650,660,670,680,690,700,710,720,730,740,750,760,770,780,790,800,900])
        x = pd.crosstab(data['ScoreBinned'],data['Target'])        
       
        x.to_csv(modelDevelopment+'2016_scaledScoreDistribution.csv')
        print("The scaled score distribution has been saved here: ",modelDevelopment+'scaledScoreDistribution.csv')
        print("")
        print("Development Scaling completed!")            
       
        print("#################################################################")
        print("#                                                               #")
        print("#                   EVALUATE 2017                               #")
        print("#                                                               #")
        print("#################################################################")  
   
   
   
   
   
        data = pd.read_csv(loc+'2017.csv')# input data without WOE values
       # data = data.drop(columns=['Unnamed: 0'],axis=1)
   
        for z in range(len(chars)):
           
   
            if(data[chars[z]].dtype !='object'):
                data[chars[z]] = Series(replace_woe(data[chars[z]], cutList[z], woeList[z]))
            else:
                data[chars[z]+'_bin'] = (
                        data[chars[z]]
                        .apply(lambda x: [k for k in binners[z].keys() if x in binners[z][k]])
                        .str[0])
                data[chars[z]] = Series(replace_woe_str(data[chars[z]+"_bin"], cutList[z], woeList[z]))
   
        data.to_csv(loc+'developmentDataset_with_WOE.csv')
       
        data['x1']  = round((data['grade']*coe[1]+(coe[0]/10))*Factor+(Offset/10),0)
        data['x2']  = round((data['emp_length']*coe[2]+(coe[0]/10))*Factor+(Offset/10),0)
        data['x3']  = round((data['dti']*coe[3]+(coe[0]/10))*Factor+(Offset/10),0)
        data['x4']  = round((data['Orig_FicoScore']*coe[4]+(coe[0]/10))*Factor+(Offset/10),0)
        data['x5']  = round((data['inq_last_6mths']*coe[5]+(coe[0]/10))*Factor+(Offset/10),0)
        data['x6']  = round((data['acc_open_past_24mths']*coe[6]+(coe[0]/10))*Factor+(Offset/10),0)
        data['x7']  = round((data['mort_acc']*coe[7]+(coe[0]/10))*Factor+(Offset/10),0)
        data['x8']  = round((data['mths_since_recent_bc']*coe[8]+(coe[0]/10))*Factor+(Offset/10),0)
        data['x9']  = round((data['num_rev_tl_bal_gt_0']*coe[9]+(coe[0]/10))*Factor+(Offset/10),0)
        data['x10'] = round((data['percent_bc_gt_75']*coe[10]+(coe[0]/10))*Factor+(Offset/10),0)
   
        data['Score'] = data['x1']+    data['x2']+    data['x3']+    data['x4']+    data['x5']+    data['x6']+    data['x7']+data['x8']+data['x9']+data['x10']
        #print(data)
        data.to_csv(loc+'_development_scored_data.csv')
        print("")
        print("Average Score of Population: ", str(round(data['Score'].mean(),1)))
        g = data.loc[data['Target'] == 1]
        print("Average Score of Goods: ", str(round(g['Score'].mean(),1)))
        g = data.loc[data['Target'] == 0]
        print("Average Score of Bads: ", str(g['Score'].mean()))
        g = data.loc[data['Target'] == -1]
        print("Average Score of Indeterminates: ", str(round(g['Score'].mean(),1)))
   
        print("")
        print("Minimum Score: ",str(data['Score'].min()))
        print("Average Score: ",str(data['Score'].mean()))
        print("Maximum Score: ",str(data['Score'].max()))
        print("")
   
        data['ScoreBinned']=pd.cut(data['Score'], bins=[0,300,310,320,330,340,350,360,370,380,390,400,410,420,430,440,450,460,470,480,490,500,510,520,530,540,550,560,570,580,590,600,610,620,630,640,650,660,670,680,690,700,710,720,730,740,750,760,770,780,790,800,900])
        x = pd.crosstab(data['ScoreBinned'],data['Target'])        
       
        x.to_csv(modelDevelopment+'2017_scaledScoreDistribution.csv')
        print("The scaled score distribution has been saved here: ",modelDevelopment+'scaledScoreDistribution.csv')
        print("")
        print("Development Scaling completed!")            
       
       
        print("#################################################################")
        print("#                                                               #")
        print("#                   EVALUATE 2018                               #")
        print("#                                                               #")
        print("#################################################################")  
   
   
   
   
   
        data = pd.read_csv(loc+'2018.csv')# input data without WOE values
       # data = data.drop(columns=['Unnamed: 0'],axis=1)
   
        for z in range(len(chars)):
           
   
            if(data[chars[z]].dtype !='object'):
                data[chars[z]] = Series(replace_woe(data[chars[z]], cutList[z], woeList[z]))
            else:
                data[chars[z]+'_bin'] = (
                        data[chars[z]]
                        .apply(lambda x: [k for k in binners[z].keys() if x in binners[z][k]])
                        .str[0])
                data[chars[z]] = Series(replace_woe_str(data[chars[z]+"_bin"], cutList[z], woeList[z]))
   
        data.to_csv(loc+'developmentDataset_with_WOE.csv')
       
        data['x1']  = round((data['grade']*coe[1]+(coe[0]/10))*Factor+(Offset/10),0)
        data['x2']  = round((data['emp_length']*coe[2]+(coe[0]/10))*Factor+(Offset/10),0)
        data['x3']  = round((data['dti']*coe[3]+(coe[0]/10))*Factor+(Offset/10),0)
        data['x4']  = round((data['Orig_FicoScore']*coe[4]+(coe[0]/10))*Factor+(Offset/10),0)
        data['x5']  = round((data['inq_last_6mths']*coe[5]+(coe[0]/10))*Factor+(Offset/10),0)
        data['x6']  = round((data['acc_open_past_24mths']*coe[6]+(coe[0]/10))*Factor+(Offset/10),0)
        data['x7']  = round((data['mort_acc']*coe[7]+(coe[0]/10))*Factor+(Offset/10),0)
        data['x8']  = round((data['mths_since_recent_bc']*coe[8]+(coe[0]/10))*Factor+(Offset/10),0)
        data['x9']  = round((data['num_rev_tl_bal_gt_0']*coe[9]+(coe[0]/10))*Factor+(Offset/10),0)
        data['x10'] = round((data['percent_bc_gt_75']*coe[10]+(coe[0]/10))*Factor+(Offset/10),0)
   
        data['Score'] = data['x1']+    data['x2']+    data['x3']+    data['x4']+    data['x5']+    data['x6']+    data['x7']+data['x8']+data['x9']+data['x10']
        #print(data)
        data.to_csv(loc+'_development_scored_data.csv')
        print("")
        print("Average Score of Population: ", str(round(data['Score'].mean(),1)))
        g = data.loc[data['Target'] == 1]
        print("Average Score of Goods: ", str(round(g['Score'].mean(),1)))
        g = data.loc[data['Target'] == 0]
        print("Average Score of Bads: ", str(g['Score'].mean()))
        g = data.loc[data['Target'] == -1]
        print("Average Score of Indeterminates: ", str(round(g['Score'].mean(),1)))
   
        print("")
        print("Minimum Score: ",str(data['Score'].min()))
        print("Average Score: ",str(data['Score'].mean()))
        print("Maximum Score: ",str(data['Score'].max()))
        print("")
   
        data['ScoreBinned']=pd.cut(data['Score'], bins=[0,300,310,320,330,340,350,360,370,380,390,400,410,420,430,440,450,460,470,480,490,500,510,520,530,540,550,560,570,580,590,600,610,620,630,640,650,660,670,680,690,700,710,720,730,740,750,760,770,780,790,800,900])
        x = pd.crosstab(data['ScoreBinned'],data['Target'])        
       
        x.to_csv(modelDevelopment+'2018_scaledScoreDistribution.csv')
        print("The scaled score distribution has been saved here: ",modelDevelopment+'scaledScoreDistribution.csv')
        print("")
        print("Development Scaling completed!")            
