# from kerykeion import Report, AstrologicalSubject, KerykeionChartSVG
import airportsdata
from geopy.geocoders import Nominatim
import ephem
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt 
             
import seaborn as sns

import scipy 
from scipy import stats 
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, LinearRegression
from sklearn.model_selection import train_test_split, StratifiedKFold,GridSearchCV, cross_validate, GroupKFold, RandomizedSearchCV, validation_curve, learning_curve
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, accuracy_score, auc
import statsmodels.formula.api as smf
import math
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import shap 
from shap import TreeExplainer, Explanation
from shap.plots import waterfall
import requests
from   datetime import datetime, timedelta,time
import matplotlib.dates as mdates
import statsmodels.api as sm
import pytz
from xgboost import XGBClassifier

from sklearn import preprocessing
def optimal_number_of_clusters(data):
        # FUNCTIONS
        cost = []
        for n in range(1, 20):
            kmeans = KMeans(n_clusters=n)
            kmeans.fit(X=data)
            cost.append(kmeans.inertia_)
        x1, y1 = 2, cost[0]
        x2, y2 = 20, cost[len(cost)-1]
        distances = []
        for i in range(len(cost)):
            x0 = i+2
            y0 = cost[i]
            numerator = np.abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
            denominator = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
            distances.append(numerator/denominator)
        n_clusters = distances.index(max(distances)) + 2
        plt.plot(range(1, 20), cost, "--o")
        plt.plot(n_clusters, cost(n_clusters-1), "o", color="red")
        plt.xlabel("#Number of Clusters")
        plt.ylabel("#Squared Error")
        plt.show()
        return n_clusters
import pickle
from sklearn.cluster import KMeans, DBSCAN
from mpl_toolkits import mplot3d
from matplotlib import cm
import tqdm as tqdm
import tensorflow as tf 

import scipy 
from keras.wrappers.scikit_learn import KerasClassifier 
from sklearn.preprocessing import LabelEncoder, StandardScaler

import mplfinance as mpf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from pandas_ta.momentum import rsi


def build_clf(unit): 
  # creating the layers of the NN 
  ann = tf.keras.models.Sequential() 
  ann.add(tf.keras.layers.Dense(units=unit, activation='relu')) 
  ann.add(tf.keras.layers.Dense(units=unit, activation='relu')) 
  ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid')) 
  ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) 
  return ann




def calculate_thresholds(n_records, n_bads, n_goods):
    # Total threshold
    if n_records <= 3000: tot_thresh = 0.1
    elif n_records > 3000 and n_records <= 5000: tot_thresh = 0.05
    elif n_records > 5000 and n_records <= 20000: tot_thresh = 0.03
    elif n_records > 20000 and n_records <= 50000: tot_thresh = 0.01
    else: tot_thresh = 0.005
    # Bad threshold
    if n_bads <= 3000: bad_thresh = 0.10
    elif n_bads > 3000 and n_bads <= 5000: bad_thresh = 0.05
    elif n_bads > 5000 and n_bads <= 20000: bad_thresh = 0.03
    elif n_bads > 20000 and n_bads <= 50000: bad_thresh = 0.01
    else:bad_thresh = 0.005
    # Good threshold
    if n_goods <= 3000: good_thresh = 0.10
    elif n_goods > 3000 and n_goods <= 5000: good_thresh = 0.05
    elif n_goods > 5000 and n_goods <= 20000: good_thresh = 0.03
    elif n_goods > 20000 and n_goods <= 50000: good_thresh = 0.01
    else: good_thresh = 0.005
    return tot_thresh, bad_thresh, good_thresh

def calculate_scorecard(d1, char):
    d2 = d1.groupby([char], dropna=False)
    d3 = pd.DataFrame(d2["X"].min(), columns=["min"])
    d3["# Good"] = d2["Y"].sum()
    d3["# Bad"] = d2["Y"].count() - d3["# Good"]
    d3["% Good"] = round(d3["# Good"] / d3["# Good"].sum() * 100, 1)
    d3["% Bad"] = round(d3["# Bad"] / d3["# Bad"].sum() * 100, 1)
    d3["# Total"] = d2["Y"].count()
    d3["% Total"] = round(d3["# Total"] / d3["# Total"].sum() * 100, 1)
    d3["Information Odds"] = round(d3["% Good"] / d3["% Bad"], 2)
    d3["Bad Rate"] = round(d3["# Bad"] / (d3["# Bad"] + d3["# Good"]) * 100, 2)
    d3["WoE"] = round(np.log(d3["% Good"] / d3["% Bad"]), 2)
    iv = (d3["% Good"] - d3["% Bad"]) * d3["WoE"] / 100
    d4 = d3.sort_index().drop(columns=["min"], axis=1)
    return d4, iv

def featureMonotonicBinning(Y, X, char):
    r = 0
    bad_flag = 0
    n = 20
    while np.abs(r) != 1 and bad_flag == 0:
        d1 = pd.DataFrame({"X": X, "Y": Y})
        d1["Value"], bins = pd.qcut(d1["X"], n, duplicates="drop", retbins=True, precision=3)
        if len(bins) == 2:
            bins = bins.tolist()
            bins.insert(0, float("-inf"))
            bins.append(float("+inf"))
            d1["Value"] = pd.cut(d1["X"], bins=bins, precision=3, include_lowest=True)
        d2 = d1.groupby("Value", as_index=True)
        r,p = stats.spearmanr(d2.mean().X, d2.mean().Y)
        d3, iv = calculate_scorecard(d1, "Value")
        d3.dropna(inplace=True)
        
        if len(d3) < 3:
            bad_flag = 1
        n = n-1
       
    pctThresh, badThresh, goodThresh = calculate_thresholds(d3["# Total"].sum(),d3["# Bad"].sum(),d3["# Good"].sum())
    condition = [(d3["% Total"] < pctThresh*100) | (d3["% Bad"] < badThresh*100) | (d3["% Good"] < goodThresh*100)]
    d3["Not Robust"] = np.select(condition, [1], 0)
    criteria = d3["Not Robust"].sum()
    d3 = d3.reset_index()
    while criteria > 0:
        i = d3[d3["Not Robust"] == 1].index[0]
        #if first row -> merge two first categories
        if i == 0:
            bins = np.delete(bins, 1)
        # if last row -> merge two last categories
        elif i == (len(d3) - 1):
            bins = np.delete(bins, len(d3)-1)
        else:
            # if number of samples greater in former -> merge with latter
            if (d3.at[i-1 , "# Total"] > d3.at[i+1 , "# Total"]):
                bins = np.delete(bins, i+1)
            # if number of samples greater in latter -> merge with former
            else:
                bins = np.delete(bins, i)        
        d1 = pd.DataFrame({"X": X, "Y": Y, "Value": pd.cut(X, bins, precision=3, include_lowest=True)})
        d3, iv = calculate_scorecard(d1, "Value")
        condition = [
            (d3["% Total"] < pctThresh*100) | 
            (d3["% Bad"] < badThresh*100) | 
            (d3["% Good"] < goodThresh*100) ]
        d3["Not Robust"] = np.select(condition, [1], 0)
        criteria = d3["Not Robust"].sum()
        d3 = d3.reset_index()
    infValue = round(iv.sum(),3)
    d3 = d3.drop(columns=["Not Robust"])
    return d3, iv, infValue
  
def get_RSI(df, column, time_window):
    """Return the RSI indicator for the specified time window."""
    diff = df[column].diff(1)

    # This preservers dimensions off diff values.
    up_chg = 0 * diff
    down_chg = 0 * diff

    # Up change is equal to the positive difference, otherwise equal to zero.
    up_chg[diff > 0] = diff[diff > 0]

    # Down change is equal to negative deifference, otherwise equal to zero.
    down_chg[diff < 0] = diff[diff < 0]

    # We set com = time_window-1 so we get decay alpha=1/time_window.
    up_chg_avg = up_chg.ewm(com=time_window - 1,
                            min_periods=time_window).mean()
    down_chg_avg = down_chg.ewm(com=time_window - 1,
                                min_periods=time_window).mean()

    RS = abs(up_chg_avg / down_chg_avg)
    df['RSI'] = 100 - 100 / (1 + RS)
    # df = df[['RSI']]
    return df


from pandas_ta.momentum import rsi
from pandas_ta.momentum import willr

from termcolor import colored as cl


plt.style.use('ggplot')
plt.rcParams["figure.figsize"] = (12,6)

import re
def distance(x,y):
    if(x > y):
        z = 360 - (abs(x - y)) 
    else:
        z = abs(x - y)
    return z

def deg_to_rad(dr):
    return (dr*math.pi)/360

def get_price(ticker, start_date, end_date):
    """Return a DataFrame with price information (open, high, low, close, adjusted close, and volume) for the ticker between the specified dates."""
    df = yf.download(ticker, start_date, end_date, progress=False)
    df.reset_index(inplace=True)

    return df# 

def get_closed_dates(df):
    """Return a list containing all dates on which the stock market was closed."""
    # Create a dataframe that contains all dates from the start until today.
    timeline = pd.date_range(start=df['Date'].iloc[0], end=df['Date'].iloc[-1])

    # Create a list of the dates existing in the dataframe.
    df_dates = [day.strftime('%Y-%m-%d') for day in pd.to_datetime(df['Date'])]

    # Finally, determine which dates from the 'timeline' do not exist in our dataframe.
    closed_dates = [
        day for day in timeline.strftime('%Y-%m-%d').tolist()
        if not day in df_dates
    ]

    return closed_dates


from numpy import unique
import investpy
import os

from pandas_ta.momentum import rsi
from pandas_ta.volatility import atr


from stock_indicators import indicators
import pandas as pd
import numpy as np
import ta

import os
import yfinance as yf

def higherRSI(nq, rsiLevel):

    xx = (100/(100-rsiLevel)-1)
    df = nq[['descences']]
    df.loc[len(df)] = 0

    df['avg_descences'] = df['descences'].ewm(com=time_window - 1,
                            min_periods=time_window).mean()
    avg_asc = (-1*df['avg_descences'].iloc[-1])*xx
    avg_desc = (df['avg_descences'].iloc[-1])
    RSTest = abs(avg_asc / avg_desc)
    RSITest = (100 - (100/(1+RSTest)))
    prevEMA = nq['avg_advances'].iloc[-1]

    com = time_window - 1
    a = 1/(1+com) 
    latestValue = (avg_asc - ((1-a)*prevEMA)) / a

    targetClose = nq['Close'].iloc[-1]+latestValue

    d = nq[['Close']]
    d.loc[len(d)] = targetClose
    get_RSI(d,'Close',time_window)
    checkCorrect = round(d['RSI'].iloc[-1],1)
    print(targetClose)
    return targetClose

    #if(checkCorrect == rsiLevel):
    #    print("RSI("+str(time_window)+"): "+str(rsiLevel)+" - Target Price: ",str(targetClose)+" -> OK")
    # else:
    #     print("")

def lowerRSI(nq, rsiLevel):

    xx =(100/(100-rsiLevel)-1)
    df = nq[['advances']]
    df.loc[len(df)] = 0
    df['avg_advances'] = df['advances'].ewm(com=time_window - 1,
                            min_periods=time_window).mean()
    avg_asc = df['avg_advances'].iloc[-1]
    avg_desc = -1*(avg_asc/xx)
    prevEMA = nq['avg_descences'].iloc[-1]
    com = time_window - 1
    a = 1/(1+com) 
    latestValue = (avg_desc - (1-a)*prevEMA) / a
    targetClose = nq['Close'].iloc[-1]+latestValue

    d = nq[['Close']]
    d.loc[len(d)] = targetClose
    get_RSI(d,'Close',time_window)
    checkCorrect = round(d['RSI'].iloc[-1],1)
    print(targetClose)
    return targetClose

    # if(checkCorrect == rsiLevel):
        # print("RSI("+str(time_window)+"): "+str(rsiLevel)+" - Target Price: ",str(targetClose)+" -> OK")
    # else:
    #     print("")

import time

from trading_ig import IGService, IGStreamService
from trading_ig.config import config
from trading_ig.lightstreamer import Subscription
from trading_ig.rest import IGService, ApiExceededException
from tenacity import Retrying, wait_exponential, retry_if_exception_type
from datetime import date

retryer = Retrying(wait=wait_exponential(),
                   retry=retry_if_exception_type(ApiExceededException))

from datetime import datetime
import MetaTrader5 as mt5
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint

from datetime import datetime
import pytz
from IPython.display import  clear_output

from scipy import stats, signal
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from sklearn import mixture as mix