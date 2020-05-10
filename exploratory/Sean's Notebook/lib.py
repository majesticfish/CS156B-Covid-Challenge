import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import git
import math
import os
import json
from sklearn import neighbors
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import pacf
from pandas.plotting import autocorrelation_plot
from datetime import datetime
from urllib.request import urlopen
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import lightgbm as lgb
import statsmodels.tsa.stattools as ts

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

repo = git.Repo("./", search_parent_directories=True)
homedir = repo.working_dir

# gets list of all fips numbers
def get_fips():
    Y = pd.read_csv(f"{homedir}/data/us/covid/deaths.csv")
    fips_list = Y.countyFIPS.values
    fips_list = fips_list[fips_list != 1] # shitty fucking
    fips_list = fips_list[fips_list != 0] # data
    return set(fips_list)
def get_date(datestr, formatstr='%Y-%m-%d'):
    return datetime.strptime(datestr, formatstr)

class CumDeathCounter():
    def __init__(self):
        self.cum_deaths = pd.read_csv(f"{homedir}/data/us/covid/deaths.csv")
        self.cum_deaths = self.cum_deaths.iloc[1:]
        fips_list = self.cum_deaths.countyFIPS.values
        fips_list = fips_list[fips_list != 1] # shitty fucking
        fips_list = fips_list[fips_list != 0] # data

        self.cache = {}
        for fips in fips_list:
            self.cache[fips] = self.get_cum_deaths(fips)
    def get_cum_deaths(self, fips, clip_zeros=False):
        idx = self.cum_deaths.index[self.cum_deaths['countyFIPS'] == fips].values[0]
        county_deaths = self.cum_deaths.loc[self.cum_deaths['countyFIPS'] == fips]
        dates = pd.to_datetime(county_deaths.columns[4:].values).map(lambda dt : str(dt))
        X = np.array([(get_date(d[:10]) - get_date('2020-01-01')).days for d in dates])
        y = []
        for i in range(4, len(county_deaths.columns)):
            y.append(county_deaths.loc[idx,county_deaths.columns[i]])
        if not clip_zeros:
            return X, y
        for i in range(len(y)):
            if y[i] != 0:
                return X[i:], y[i:]
    def getY(self, fips):
        return self.cache[fips]
class CumCaseCounter():
    def __init__(self):
        self.cum_cases = pd.read_csv(f"{homedir}/data/us/covid/confirmed_cases.csv")
        self.cum_cases = self.cum_cases.iloc[1:]
        self.cum_cases = self.cum_cases.iloc[:, :-1]

        fips_list = self.cum_cases.countyFIPS.values
        fips_list = fips_list[fips_list != 1] # shitty fucking
        fips_list = fips_list[fips_list != 0] # data

        self.cache = {}
        for fips in fips_list:
            self.cache[fips] = self.get_cum_cases(fips)

    def get_cum_cases(self, fips,clip_zeros=False):
        idx = self.cum_cases.index[self.cum_cases['countyFIPS'] == fips].values[0]
        county_cases = self.cum_cases.loc[self.cum_cases['countyFIPS'] == fips]
        dates = pd.to_datetime(county_cases.columns[4:].values).map(lambda dt : str(dt))
        X = np.array([(get_date(d[:10]) - get_date('2020-01-01')).days for d in dates])
        y = []
        for i in range(4, len(county_cases.columns)):
            y.append(county_cases.loc[idx,county_cases.columns[i]])
        if not clip_zeros:
            return X, y
        for i in range(len(y)):
            if y[i] != 0:
                return X[i:], y[i:]
    def getY(self, fips):
        return self.cache[fips]
class DeltaDeathCounter():
    def __init__(self):
        self.df = pd.read_csv(f"{homedir}/data/us/covid/nyt_us_counties_daily.csv")
        fips_list = self.df.fips.unique()

        self.cache = {}
        for fips in tqdm(fips_list):
            county = self.df.loc[self.df['fips'] == fips]
            X = np.array([(get_date(d) - get_date('2020-01-01')).days for d in county.date])
            y = county.deaths
            self.cache[fips] = X,y
    def getY(self, fips):
        return self.cache[fips]
class DeltaCaseCounter():
    def __init__(self):
        self.df = pd.read_csv(f"{homedir}/data/us/covid/nyt_us_counties_daily.csv")
        fips_list = self.df.fips.unique()

        self.cache = {}
        for fips in tqdm(fips_list):
            county = self.df.loc[self.df['fips'] == fips]
            X = np.array([(get_date(d) - get_date('2020-01-01')).days for d in county.date])
            y = county.cases
            self.cache[fips] = X,y
    def getY(self, fips):
        return self.cache[fips]

class DeltaCounter:
    def __init__(self, counter):
        self.counter = counter
    def getY(self, fips):
        X, y = self.counter.getY(fips)
        y_true = [y[0]]
        for i in range(1, len(y)):
            y_true.append(y[i] - y[i-1])
        return X, y_true
class RatioCounter:
    def __init__(self, counter):
        self.counter = counter
    def getY(self, state, avg_period=5):
        X_raw, y_raw = self.counter.getY(state)
        y = []
        running_sum = 0.0
        running_time = 0
        for i in range(len(X_raw)):
            if y_raw[i] == 0:
                y.append(0)
            elif running_sum == 0:
                y.append(1) # if this is the first case we define the signal as 1
            else:
                avg = running_sum/running_time
                y.append(y_raw[i]/avg)
            if running_time == avg_period:
                running_sum = running_sum + y_raw[i] - y_raw[i - avg_period]
            else:
                running_sum = running_sum + y_raw[i]
                running_time = running_time + 1
            if running_sum == 0:
                running_time = 1
        return (X_raw, np.array(y))
class MobilityCounter:
    def __init__(self, states=False):
        data_dir = f'{homedir}/data/us/mobility'
        df = pd.read_csv(os.path.join(data_dir, 'DL-us-mobility-daterow.csv'))
        counties = df[df['admin_level'] == 2]
        counties['fips'] = counties['fips'].map(lambda f: int(f))
        self.fips_list = np.unique(counties.fips.values)
        self.data = {}

        for fips in tqdm(self.fips_list):
            county_data = counties[counties['fips'] == fips]
            county_data['date'] = county_data['date'].map(lambda l: (get_date(l) - get_date('2020-01-01')).days)
            self.data[fips] = county_data['date'], county_data['m50']
    def getY(self, fips):
        if fips not in self.fips_list:
            raise ValueError('No Mobility data for this fips')
        return self.data[fips]
    def getYByDay(self, fips, day):
        try:
            X,y = self.getY(fips)
            return y.values[np.where(X == day)[0][0]]
        except:
            raise ValueError("FIPS not found or day out of range")

def add_neighbors(df, neighbor_cols = ['pclon10', 'pclat10'], k = 5, feature_cols = ['deaths', 'cases']):

    for feature in feature_cols:
        for i in range(k):
            df['neighbor'+str(i+1)+'_'+feature] = np.nan

    if(len(df) > k):
        # N+1 neighbors because the closest neighbor is itself
        model = neighbors.NearestNeighbors(k+1, n_jobs = 4).fit(df[neighbor_cols])
        distances, indices = model.kneighbors(df[neighbor_cols])
        indices = np.transpose(indices[:, 1:]) # Remove itself from the neighbors
        distances = np.transpose(distances[:, 1:])



        for i in range(k):
            df['dist_'+str(i)] = distances[i]
            data = df.iloc[indices[i]]
            for feature in feature_cols:
                df['neighbor'+str(i+1)+'_'+feature] = data[feature].values
    return df
def add_radius_neighbors(df, neighbor_cols = ['pclon10', 'pclat10'], k = 30, radius = 0.5):
    if(len(df) > k):
        # N+1 neighbors because the closest neighbor is itself
        model = neighbors.NearestNeighbors(k+1, n_jobs = 4).fit(df[neighbor_cols])
        distances, indices = model.kneighbors(df[neighbor_cols])
        indices = np.transpose(indices[:, 1:]) # Remove itself from the neighbors
        distances = np.transpose(distances[:, 1:])
        deaths = df['deaths'].values[indices]
        cases = df['cases'].values[indices]
        truncated_distances = distances.copy()
        truncated_distances[truncated_distances > radius] = np.nan
        df['num_in_radius'] = np.sum(distances <= radius , axis = 0)
        df['deaths_in_radius'] = np.sum(deaths* (distances <= radius) , axis = 0)
        df['cases_in_radius'] = np.sum(cases* (distances <= radius) , axis = 0)
    return df

