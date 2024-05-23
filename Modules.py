import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # Visualization
import seaborn as sns #Visualization
plt.rcParams['figure.figsize'] = [8,5]
plt.rcParams['font.size'] =14
plt.rcParams['font.weight']= 'bold'
import scipy 
from scipy import stats 
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, LinearRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, accuracy_score, auc
import statsmodels.formula.api as smf
import math
from sklearn.model_selection import GridSearchCV, cross_validate, GroupKFold
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = [8,5]
plt.rcParams['font.size'] =14
plt.rcParams['font.weight']= 'bold'

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, accuracy_score, auc
import statsmodels.formula.api as smf
import math
import warnings
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')

from sklearn.model_selection import RandomizedSearchCV, validation_curve
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import shap 
from shap import TreeExplainer, Explanation
from shap.plots import waterfall
from sklearn.model_selection import learning_curve


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
        plt.plot(n_clusters, cost[n_clusters-1], "o", color="red")
        plt.xlabel("Number of Clusters")
        plt.ylabel("Squared Error")
        plt.show()
        return n_clusters
import pickle
from sklearn.cluster import KMeans, DBSCAN
from mpl_toolkits import mplot3d
from matplotlib import cm
# Progress Bar
import tqdm as tqdm



import tensorflow as tf 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # Visualization
import seaborn as sns #Visualization
plt.rcParams['figure.figsize'] = [8,5]
plt.rcParams['font.size'] =14
plt.rcParams['font.weight']= 'bold'
import scipy 
from scipy import stats 
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from keras.wrappers.scikit_learn import KerasClassifier 
from sklearn.model_selection import GridSearchCV 
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import StandardScaler 

import mplfinance as mpf
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from pandas_ta.momentum import rsi
from sklearn.linear_model import LinearRegression

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
    d3 = d3.drop(columns=["Not Robust"])
    infValue = round(iv.sum(),3)

    ax1 = d3.plot.bar(x='Value', y='WoE', rot=0)
    ax1.legend("")
    ax1.tick_params(axis='both', which='major', labelsize=9)
    ax1.set_title("Train Sample - "+char+" WoE - Inf. Value: " + str(infValue), fontsize=10)
    ax1.set_xlabel(char, fontsize = 10)
    ax1.set_ylabel("Weight of Evidence", fontsize = 10)
    display(d3)
    return d3, iv, infValue

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
    d3 = d3.drop(columns=["Not Robust"])
    infValue = round(iv.sum(),3)
    display(d3)
    ax1 = d3.plot.bar(x='Value', y='WoE', rot=0)
    ax1.legend("")
    ax1.tick_params(axis='both', which='major', labelsize=9)
    ax1.set_title("Train Sample - "+char+" WoE - Inf. Value: " + str(infValue), fontsize=10)
    ax1.set_xlabel(char, fontsize = 10)
    ax1.set_ylabel("Weight of Evidence", fontsize = 10)
    return d3, iv, infValue



def fMb(Y, X, char):
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
    d3 = d3.drop(columns=["Not Robust"])
    infValue = round(iv.sum(),3)
    return d3, iv, infValue
