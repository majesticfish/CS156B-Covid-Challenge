import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import git
import math
import os
import random
import csv
import json
from sklearn import neighbors
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.feature_selection import RFE
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import pacf
from pandas.plotting import autocorrelation_plot
import datetime
from urllib.request import urlopen
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import lightgbm as lgb
import statsmodels.tsa.stattools as ts
from lib import *
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

repo = git.Repo("./", search_parent_directories=True)
homedir = repo.working_dir

df_jhu = pd.read_csv(f"{homedir}/data/us/aggregate_jhu.csv")
county_centers = pd.read_csv(f'{homedir}/data/us/geolocation/county_centers.csv')
df_jhu_merged = 0
df_jhu_merged_counties = 0
bookings = pd.read_csv(f"{homedir}/data/us/state_of_industry_data.csv")
bookings_data = {}

def process_jhu():
    global df_jhu
    # Get rid of the aggregate country data
    df_jhu = df_jhu.drop([0])
    df_jhu['FIPS'] = df_jhu['FIPS'].map(lambda f : str(f))
    def alter(fips):
        if len(fips) == 4:
            return '0' + fips
        return fips
    df_jhu['FIPS'] = df_jhu['FIPS'].map(alter)
    df_jhu = df_jhu.set_index('FIPS')
    df_jhu['fips'] = df_jhu.index.map(lambda s : int(s))
    # fill out missing data
    df_jhu.at['02158', 'Area in square miles - Land area'] = 19673
    df_jhu.at['02158', 'Density per square mile of land area - Population'] = 0.44
    df_jhu.at['46102', 'Area in square miles - Land area'] = 2097
    df_jhu.at['46102', 'Density per square mile of land area - Population'] = 6.5

def init_county_centers():
    global county_centers
    county_centers = county_centers[['pclon10', 'pclat10', 'fips']]

def merge_centers_with_jhu():
    global df_jhu
    global df_jhu_merged
    global df_jhu_merged_counties
    df_jhu_merged = df_jhu.merge(county_centers, how = 'left', on='fips')
    df_jhu_merged['FIPS'] = df_jhu_merged['fips']
    df_jhu_merged = df_jhu_merged.set_index('FIPS')
    df_jhu_merged_counties = df_jhu_merged[df_jhu_merged.fips % 1000 != 0]

us_state_abbrev = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'American Samoa': 'AS',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'District of Columbia': 'DC',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Guam': 'GU',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Northern Mariana Islands':'MP',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Puerto Rico': 'PR',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virgin Islands': 'VI',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY'
}

def init_bookings_data():
    global bookings
    global bookings_data
    bookings = bookings[bookings.Type == 'state']
    def name(name):
        if name in us_state_abbrev:
            return us_state_abbrev[name]
        return "Not US"
    bookings.Name = bookings.Name.map(name)
    bookings = bookings[bookings.Name != 'Not US']
    bookings.set_index('Name')
    for idx, row in bookings.iterrows():
        bookings_data[row['Name']] = {}
        for col in bookings.columns:
            try:
                t = (get_date('2020/' + col, formatstr='%Y/%m/%d') - get_date('2020-01-01')).days
                bookings_data[row['Name']][t] = row[col]
            except:
                pass

# gets list of all fips numbers
def get_fips():
    Y = pd.read_csv(f"{homedir}/data/us/covid/deaths.csv")
    fips_list = Y.countyFIPS.values
    fips_list = fips_list[fips_list != 1] # shitty fucking
    fips_list = fips_list[fips_list != 0] # data
    return set(fips_list)
def get_date(datestr, formatstr='%Y-%m-%d'):
    return datetime.datetime.strptime(datestr, formatstr)

def split_fips(counter, min_thresh=10):
    fips_list = get_fips()
    large_fips = []
    small_fips = []
    for fips in fips_list:
        X,y = counter.getY(fips)
        if y[-1] > min_thresh:
            large_fips.append(fips)
        else:
            small_fips.append(fips)
    return large_fips, small_fips

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
            self.data[fips] = county_data['date'], county_data['m50_index']
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
def add_radius_neighbors(df, neighbor_cols = ['pclon10', 'pclat10'], k = 30, radius = 0.65):
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

def get_XY(df, delta_y, look_back_y, y_generator, fips_list=get_fips(),
           moving_window=True, start_date=110, offset=0, features=None,
           add_cases=False, delta_case_counter=None, mobility_counter=None):
    if features != None:
        df = df[features]
    df = df[df.fips % 1000 != 0] # remove aggregate states
    df = df[df.State != 'PR']   # peurto rico has some weird data...

    df['state resturant data'] = df['fips'].map(lambda fip: 0)
    df['mobility'] = df['fips'].map(lambda fip: 0)
    df['cases'] = df['fips'].map(lambda fip: 0)

    col_names = []
    for i in range(look_back_y):
        if i == 0:
            col_name = "deaths"
        else:
            col_name = "deaths %d days ago" %i
        col_names.append(col_name)
        df[col_name] = df['fips'].map(lambda fip: 0)
    X = []
    Y = []
    Y_add = []
    for fips in df.index:
        if int(fips) not in fips_list:
            continue
        base = df.loc[fips].values
        _, cum_cases = delta_case_counter.getY(int(fips))
        try:
            t, ys = y_generator.getY(int(fips))
            ys = ys.values
        except KeyError:
            continue
        if len(ys) == 0:
            X.append(base)
            Y.append(0)
            continue
        for j in range(-1-offset, -len(ys), -1):
            base = df.loc[fips].values
            time = t[j-delta_y]
            try:
                base[-look_back_y - 3] = bookings_data[df.at[fips, 'State']][time]
            except:
                base[-look_back_y - 3] = -100
            if j - delta_y >= -len(cum_cases):
                base[-look_back_y - 1] = cum_cases[j - delta_y]
            try:
                base[-look_back_y - 2] = mobility_counter.getYByDay(int(fips), time)
                if time < start_date:
                    break
            except:
                try:
                    tmp1,tmp2 = mobility_counter.getY(int(fips))
                    base[-look_back_y-2] = tmp2.values[-1]
                except:
                    pass
            for i in range(look_back_y):
                if j - delta_y - i >= -len(ys):
                    base[-look_back_y + i] = ys[j - delta_y - i]
            X.append(base)
            Y.append(ys[j])
            if add_cases:
                Y_add.append(cum_cases[j])
            if not moving_window:
                break
    df_new = pd.DataFrame(X, columns = df.columns)
    df_new['target'] = Y
    if add_cases:
        df_new['target_cases'] = Y_add
    return df_new

def get_XY_with_neighs(df, delta_y, look_back_y, y_generator, fips_list=get_fips(),
                       moving_window=True,offset=0, features=None, add_cases=False,
                       delta_case_counter=None, mobility_counter=None):
    if features != None:
        df = df[features]
    df_new = get_XY(df, delta_y, look_back_y, y_generator, fips_list=fips_list,
                    moving_window=moving_window,offset=offset, add_cases=add_cases,
                    delta_case_counter=delta_case_counter, mobility_counter=mobility_counter)
    df_new = df_new.loc[~np.isnan(df_new.pclon10)]
    if add_cases:
        X = df_new.iloc[:,2:-2]
        y_cases = df_new['target_cases']
    else:
        X = df_new.iloc[:,2:-1]
    y = df_new['target'].values
    X = add_radius_neighbors(add_neighbors(X))
    if add_cases:
        return X,y,y_cases
    return X, y

def Rsquare(pred, actual):
    return np.corrcoef(pred, actual)[0,1]**2

col_names = []
for j in range(10):
    if j > 0:
        copy_col = 'deaths %d days ago' % (j)
    else:
        copy_col = 'deaths'
    col_names.append(copy_col)
def get_percentile_obj(model, X, good_cols, min_look_ahead=1, max_look_ahead=14, cases_predictor=None):
    model_neigh = neighbors.NearestNeighbors(6, n_jobs = 4).fit(X[['pclon10', 'pclat10']])
    distances, indices = model_neigh.kneighbors(X[['pclon10', 'pclat10']])
    indices = np.transpose(indices[:, 1:]) # Remove itself from the neighbors
    distances = np.transpose(distances[:, 1:])
    output = {}
    df = X.copy()
    for i in range(min_look_ahead, max_look_ahead + 1):
        pred = model.predict(df[good_cols].values)  # Unbiased prediction
        std_prev = np.std(df[col_names].values,axis=1) # standard deviation of prediction
        pred = np.random.normal(pred,std_prev) # draw from normal dist
        pred = pred.clip(min=0) # No negative deaths!

        if cases_predictor is not None:
            pred_cases = cases_predictor.predict(df[good_cols].values) # Num case predictions
            pred_cases = pred_cases.clip(min=0) #no negative cases
            df['cases'] = pred_cases
        fips_to_pred = {}
        for j in range(len(df.index)):
            fips = df.index[j]
            if pred[j] < 0:
                fips_to_pred[fips] = 0
            else:
                fips_to_pred[fips] = pred[j]
        for j in range(9,0,-1):
            df[col_names[j]] = df[col_names[j-1]] # Shift the columns to simulate the progress of a day
        df['deaths'] = pred

        # Updating neighbor case/death counts
        for j in range(5):
            data = df.iloc[indices[j]]
            df['neighbor'+str(j+1)+'_'+'deaths'] = data['deaths'].values
            df['neighbor'+str(j+1)+'_'+'cases'] = data['cases'].values
        deaths = df['deaths'].values[indices]
        cases = df['cases'].values[indices]
        truncated_distances = distances.copy()
        truncated_distances[truncated_distances > 0.65] = np.nan
        df['deaths_in_radius'] = np.sum(deaths* (distances <= 0.65) , axis = 0)
        df['cases_in_radius'] = np.sum(cases* (distances <= 0.65) , axis = 0)

        output[i] = fips_to_pred
    return output
def get_percentile_obj_rand(model, X, good_cols, min_look_ahead=1, max_look_ahead=14,cases_predictor=None):
    num_samples = 200
    dummy = get_percentile_obj(model, X, good_cols, min_look_ahead=min_look_ahead, max_look_ahead=max_look_ahead,
                              cases_predictor=cases_predictor)
    freq = {}
    output = {}
    for key in dummy.keys():
        freq[key] = {}
        output[key] = {}
        for fip in dummy[key].keys():
            freq[key][fip] = [dummy[key][fip]]
            output[key][fip] = {}
    for i in tqdm(range(num_samples)):
        dummy = get_percentile_obj(model, X, good_cols, min_look_ahead=min_look_ahead, max_look_ahead=max_look_ahead)
        for key in dummy.keys():
            for fip in dummy[key].keys():
                freq[key][fip].append(dummy[key][fip])
    for key in dummy.keys():
        for fip in dummy[key].keys():
            for percentile in range(10, 100, 10):
                output[key][fip][percentile] = np.percentile(freq[key][fip], percentile, axis=0)
    return output

df_sample_sub = pd.read_csv(f"{homedir}/sample_submission.csv")
df_sample_sub['fips'] = df_sample_sub['id'].map(lambda i : int(i[11:]))
output_fips = df_sample_sub.fips.unique()
def truncate(dec):
    return int(100*dec)/100
import datetime
def write_percentiles(percentile_obj, file_name, prediction_limit, offset=0, delta_death_counter=None,
                      start_date='2020-04-01', end_date='2020-06-30'):
    with open(file_name, mode='w') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['id', '10', '20', '30', '40', '50', '60', '70', '80', '90'])

        today = datetime.datetime.today() + datetime.timedelta(days=-offset);
        today = datetime.datetime.combine(today, datetime.datetime.min.time())
        st = get_date(start_date)
        ed = get_date(end_date)
        for i in range((st - today).days, 0):
            target_day = today + datetime.timedelta(days=i)
            string_pre = target_day.strftime('%Y-%m-%d-')
            for fips in output_fips:
                print_lst = [string_pre + str(fips)]
                try:
                    X, y = delta_death_counter.getY(fips)
                    y = y.values
                    for j in range(9):
                        print_lst.append(y[i-offset])
                except:
                    for j in range(9):
                        print_lst.append(0)
                writer.writerow(print_lst)
        for delta_y in range(1, (ed - today).days + 2):
            target_day = today + datetime.timedelta(days=delta_y-1)
            string_pre = target_day.strftime('%Y-%m-%d-')
            if delta_y > prediction_limit:
                for fips in output_fips:
                    l = [string_pre + str(fips)]
                    writer.writerow(l + [0] * 9)
                continue
            for fips in output_fips:
                print_lst = [string_pre + str(fips)]
                if fips in percentile_obj[delta_y].keys():
                    for percentiles in range(10,100,10):
                        datum = percentile_obj[delta_y][fips][percentiles]
                        if datum < 0:
                            datum = 0
                        print_lst.append(truncate(datum))
                else:
                    for i in range(9):
                        print_lst.append(0)
                writer.writerow(print_lst)
