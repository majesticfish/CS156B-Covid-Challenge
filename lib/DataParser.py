import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import git
import math
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import pacf
from pandas.plotting import autocorrelation_plot
from datetime import datetime
repo = git.Repo("./", search_parent_directories=True)
homedir = repo.working_dir

df_jhu = pd.read_csv(f"{homedir}/data/us/aggregate_jhu.csv")

# Get rid of the aggregate country data
df_jhu = df_jhu.drop([0])

# convert data into a string
df_jhu['FIPS'] = df_jhu['FIPS'].map(lambda f : str(f))

# make sure FIPS codes are all length 5 strings
def alter(fips):
    if len(fips) == 4:
        return '0' + fips
    return fips
df_jhu['FIPS'] = df_jhu['FIPS'].map(alter)
df_jhu = df_jhu.set_index('FIPS')
df_jhu['fips'] = df_jhu.index.map(lambda s : int(s))

# gets list of all fips numbers
def get_fips():
    Y = pd.read_csv(f"{homedir}/data/us/covid/nyt_us_counties_daily.csv")
    return set(Y.fips.values)

# helper date function
def get_date(datestr, formatstr='%Y-%m-%d'):
    return datetime.strptime(datestr, formatstr)

cum_deaths = pd.read_csv(f"{homedir}/data/us/covid/deaths.csv")
cum_deaths = cum_deaths.iloc[1:]

def get_cum_deaths(fips, clip_zeros=True):
    """ function that returns cumulative death data for a county

    Parameters
    -----------
    fips: int
        FIPS code of county in question
    clip_zeros: bool
        When this is set to true, the function will only start reporting
        when deaths start occuring

    Returns
    ----------
    (X, y) : (ndarray, ndarry)
        X: array of number of days since Jan 1st
        y: number of cumulative deaths
    """
    idx = cum_deaths.index[cum_deaths['countyFIPS'] == fips].values[0]
    county_deaths = cum_deaths.loc[cum_deaths['countyFIPS'] == fips]
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

cum_cases = pd.read_csv(f"{homedir}/data/us/covid/confirmed_cases.csv")
cum_cases = cum_cases.iloc[1:]
cum_cases = cum_cases.iloc[:, :-1]
def get_cum_cases(fips,clip_zeros=False):
    """ function that returns cumulative cases data for a county
    Parameters
    -----------
    fips: int
        FIPS code of county in question
    clip_zeros: bool
        When this is set to true, the function will only start reporting
        when deaths start occuring. WARNING: setting this to be true
        could case the return value to be none.

    Returns
    ----------
    (X, y) : (ndarray, ndarry)
        X: array of number of days since Jan 1st
        y: number of cumulative cases
    """
    idx = cum_cases.index[cum_cases['countyFIPS'] == fips].values[0]
    county_cases = cum_cases.loc[cum_cases['countyFIPS'] == fips]
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

NYT_counties_daily = pd.read_csv(f"{homedir}/data/us/covid/nyt_us_counties_daily.csv")
Y_county = NYT_counties_daily.loc[NYT_counties_daily['fips'] == 1005, :]
def get_delta_deaths(fips, clip_zeros=False):
    """Returns the number of new deaths per day of a given county

    Parameters
    ----------
    fips: int
        FIPS code of county in question
    clip_zeros: bool
        If set to true, it will only report data after the 1st death occurs

    Returns
    ----------
    (X, y): ndarray, ndarray
        X: number of days since Jan 1st
        y: number of deaths per day
    """
    Y_county = NYT_counties_daily.loc[NYT_counties_daily['fips'] == fips, :]
    Y_county.head()
    start_date = '2020-01-01'
    Y_county['time'] =  Y_county['date'].map(lambda d : (get_date(d) - get_date('2020-02-01')).days)
    X, y = (Y_county.time.values, Y_county.deaths.values)
    if not clip_zeros:
        return X, y
    for i in range(len(y)):
        if y[i] != 0:
            break
    return X[i:], y[i:]

def get_delta_cases(fips, clip_zeros=False):
    """Returns the number of new cases per day of a given county

    Parameters
    ----------
    fips: int
        FIPS code of county in question
    clip_zeros: bool
        If set to true, it will only report data after the 1st death occurs

    Returns
    ----------
    (X, y): ndarray, ndarray
        X: number of days since Jan 1st
        y: number of new cases per day
    """

    Y_county = NYT_counties_daily.loc[NYT_counties_daily['fips'] == fips, :]
    Y_county.head()
    start_date = '2020-01-01'
    Y_county['time'] =  Y_county['date'].map(lambda d : (get_date(d) - get_date('2020-02-01')).days)
    X, y = (Y_county.time.values, Y_county.cases.values)
    if not clip_zeros:
        return X, y
    for i in range(len(y)):
        if y[i] != 0:
            break
    return X[i:], y[i:]

def get_delta_deaths_ratio(fips, clip_zeros=False, avg_period=5):
    """Returns the number of new deaths per day as a ratio over the running
    average number of new deaths. When ratio is undefined, we set to 1

    Parameters
    ----------
    fips: int
        FIPS code of county in question
    clip_zeros: bool
        If set to true, it will only report data after the 1st death occurs
    avg_period: int
        Length of running average to keep track of

    Returns
    ----------
    (X, y): ndarray, ndarray
        X: number of days since Jan 1st
        y: ratio number of deaths per day to the running average
    """

    X_raw, y_raw = get_delta_deaths(fips, clip_zeros)
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
def get_delta_cases_ratio(fips, clip_zeros=False, avg_period=5):
    """Returns the number of new cases per day as a ratio over the running
    average number of new deaths

    Parameters
    ----------
    fips: int
        FIPS code of county in question
    clip_zeros: bool
        If set to true, it will only report data after the 1st death occurs
    avg_period: int
        Length of running average to keep track of

    Returns
    ----------
    (X, y): ndarray, ndarray
        X: number of days since Jan 1st
        y: ratio number of cases per day to the running average
    """
    X_raw, y_raw = get_delta_cases(fips, clip_zeros)
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

def get_XY(features, delta_y, look_back_y, get_y):
    """
    This is kinda jank maybe don't use it.
    """
    df = df_jhu[features]
    df = df[df.fips % 1000 != 0]
    df = df[df.State != 'PR']   # peurto rico has some weird data...
    df = df[df.POP_ESTIMATE_2018 > 1000] # restrict to large counties since getting lots of data is difficult

    # fill out missing data
    df.at['02158', 'Area in square miles - Land area'] = 19673
    df.at['02158', 'Density per square mile of land area - Population'] = 0.44
    df.at['46102', 'Area in square miles - Land area'] = 2097
    df.at['46102', 'Density per square mile of land area - Population'] = 6.5

    n, d = df.shape
    col_names = []
    for i in range(look_back_y):
        col_name = "y at t = -%d" %i
        col_names.append(col_name)
        df[col_name] = np.zeros(n)
    Y = []
    for fips in df.index:
        X, ys = get_y(int(fips))
        if len(ys) == 0:
            Y.append(0)
            continue
        Y.append(ys[-1])
        for i in range(look_back_y):
            if i + delta_y < len(ys):
                df.at[fips, col_names[i]] = ys[-1 - i - delta_y]
    df['target'] = Y
    return df
