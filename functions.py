'''Some helper functions for data ETL including:
    - Load features from dataframe
    - Normalization and denormalize
    - Load dataset, pytorch dataset
    - Load adjacent matrix, load graph network
    - Preprocess dataset
'''
import numpy as np
import pandas as pd
import torch
from datetime import datetime
import dgl

###
# Function: Load features from given dataframe
def load_features(feat_path, dtype=np.float32):
    feat_df = pd.read_csv(feat_path)
    feat = np.array(feat_df, dtype=dtype)
    return feat

###
# Function: normalize data using min max approach
def min_max_normalization(array):
    return 2*(array-np.min(array))/(np.max(array)-np.min(array))-1

###
# Function: denormalize the array given min max
def denormalize(array, min, max):
    return (array+1) * (max - min)/2 + min

###
# Function: get csv file and return pandas dataframe
# Input: path to csv file
# Output: pandas dataframe
def load_dataset(feat_path, dtype=np.float32):
    feat_df = pd.read_csv(feat_path)
    feat_df['date'] = feat_df[['year', 'month', 'day']].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    feat_df['date'] = pd.to_datetime(feat_df['date'], format='%Y %m %d', errors='coerce')
    return feat_df

###
# Function: return hourly summary of crowd information
# Input: pandas dataframe
# Output: hourly 3D numpy array crowd information:
#       [time][location][outflow,inflow]
def load_hourly_features(feat_df, dtype=np.float32):
    hourly_df = feat_df.copy()
    hourly_df.drop('date', axis=1, inplace=True)
    #feat = np.array(hourly_df, dtype=dtype)
    # add leading zero on month, day, hour (up to 2)
    if 'year' in hourly_df.columns and 'month' in hourly_df.columns and 'day' in hourly_df.columns and 'hour' in hourly_df.columns:
        hourly_df['month'] = hourly_df['month'].apply(lambda x: '{0:0>2}'.format(x))
        hourly_df['day'] = hourly_df['day'].apply(lambda x: '{0:0>2}'.format(x))
        hourly_df['hour'] = hourly_df['hour'].apply(lambda x: '{0:0>2}'.format(x))
        hourly_df['year_month_day_hour'] = hourly_df['year'].astype(str) + '_' + hourly_df['month'].astype(str) + '_' + hourly_df['day'].astype(str) + '_' + hourly_df['hour'].astype(str)
        hourly_df.drop(['year','month', 'day', 'hour'], axis=1, inplace=True)
    hourly_df.sort_values(["year_month_day_hour", "locationid"], ascending = (True, True), inplace=True)
    ymdh_unique = hourly_df['year_month_day_hour'].sort_values().unique()
    loc_unique = hourly_df['locationid'].sort_values().unique()
    n_max = len(loc_unique)

    # Assign MultiIndex
    index = pd.MultiIndex.from_arrays([hourly_df['year_month_day_hour'].tolist(), hourly_df['locationid'].tolist()], names=('year_month_day_hour', 'locationid'))
    hourly_df = pd.DataFrame({'outflow': hourly_df['outflow'].tolist(), 'inflow' : hourly_df['inflow'].tolist()}, index=index)

    # Fill in zero values for non existing inflow or outflow
    new_index = pd.MultiIndex.from_product([ymdh_unique,loc_unique], names=('year_month_day_hour', 'locationid'))
    hourly_df = hourly_df.reindex(new_index, fill_value=0)
    feat = np.array(list(hourly_df.groupby('year_month_day_hour').apply(pd.DataFrame.to_numpy).apply(lambda x: np.pad(x, ((0, n_max-len(x)), (0, 0)), 'constant'))))
    return feat


###
# Function: Load adjacent matrix from given path
def load_adjacency_matrix(adj_path, dtype=np.float32):
    adj_df = pd.read_csv(adj_path, header=None)
    adj = np.array(adj_df, dtype=dtype)
    return adj

###
# Function: Load adjacent matrix in graph form from given path
def load_graph(adj_path):
    ## Creat list of Edge <u,v>
    adj = load_adjacency_matrix(adj_path)
    g = dgl.DGLGraph()
    g.add_nodes(adj.shape[0])
    for i in range(adj.shape[0]//2+1):
        for j in range(adj.shape[1]):
            if adj[i][j] > 0:
                g.add_edges(i,j)
    g = dgl.add_self_loop(g)
    return g

### test load_graph
#graph = load_graph('./data/nyc_adj.csv')
#print(graph.adj())
#print(graph.number_of_nodes())

###
# Function: Multi view generation
# input:
#    data: 3D array [time frame, location_id, features]
#    t : training/testing data for time t
#    interval: length of time frame used for prediction
# output:
#    trainX: input of model, previous data used for prediction, a 4d array [location_id,features,time_frame,4 different time_interval]
#    trainY: output of model, a 2d array [location_id,features]

def multi_view_generate(data,t,interval):
    recent = data[t-interval:t].transpose((1,2,0))
    daily = data[t-interval*24:t:24].transpose((1,2,0))
    weekly = data[t-interval*24*7:t:24*7].transpose((1,2,0))
    monthly = data[t-interval*24*30:t:24*30].transpose((1,2,0))
    train_X = np.stack((recent,daily,weekly,monthly), axis=3)
    train_Y = data[t]

    return train_X,train_Y


###
# Function: Generate dataset
# input:
#    :param data: feature matrix
#    :param ln: Last n items of the given time
#    :param interval: temporal increase
# output:
#    :train set (X, Y) and test set (X, Y)
def generate_dataset(file,interval):
    split_ratio = 0.7

    try:
        data = np.load('./data/crowd_flow.npy')
        #print(np.min(data),np.max(data))
    except:
        feat_df = load_dataset(file)
        data = load_hourly_features(feat_df)
        np.save('./data/crowd_flow.npy',data)

    data = min_max_normalization(data)
    
    seq_len = int(data.shape[0])
    begin = int(interval*30*24)
    train_size = int((seq_len-begin)*split_ratio)
    train_X, train_Y, test_X, test_Y = list(), list(), list(), list()

    for t in range(begin,begin+train_size):
        X,Y = multi_view_generate(data,t,interval)
        train_X.append(X)
        train_Y.append(Y)
    for t in range(begin+train_size,seq_len):
        X,Y = multi_view_generate(data,t,interval)
        test_X.append(X)
        test_Y.append(Y)
    return np.array(train_X), np.array(train_Y), np.array(test_X), np.array(test_Y)

    #numpy.random.shuffle(x)

###
# Function: Generate dataset understandable by pytorch 
def generate_torch_datasets(file, interval, normalize=True):
    train_X, train_Y, test_X, test_Y = generate_dataset(file,interval)
    print('loading data, max and min is printed:',np.min(train_X),np.max(train_Y))
    train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(train_X), torch.FloatTensor(train_Y))
    test_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(test_X), torch.FloatTensor(test_Y))
    return train_dataset, test_dataset
    
