#!/usr/bin/env python
# coding: utf-8


import os
import pandas as pd
import numpy as np
from datetime import datetime



#####
# Helper Functions
#####

###
# Objective: Get all the csv files including within the subfolder
# input: root folder location
# output: list of csv files within given folder
def getCSVfile(foldername):
    csvFile = []
    for i, fname in enumerate(sorted(os.listdir(foldername))):
        if fname.endswith(".csv"):
            csvFile.append(foldername+"/"+fname)
        elif (os.path.isdir(foldername+"/"+fname)):
            csvFile += getCSVfile(foldername+"/"+fname)
    return csvFile


###
# Objective: Read .csv file and standardize the column name (to avoid error in different column name)
def readCSVfile(filename):
    df = pd.read_csv(filename)
    df.columns = map(str.lower, df.columns)
    
    return df



# Global variable
folder = 'E:/Henry/nyctaxi/'
folder = 'E:/Henry/nyctaxi/'
zone_fname = 'taxi_zone_lookup.csv'

# list of data files
year_arr = [2017,2018,2019]
taxi_arr = ['fhvhv','fhv','green','yellow']
hour = 24
public_holiday_arr = ['02-01-2017', '16-01-2017', '13-02-2017', '20-02-2017', '29-05-2017',]

# list of array of interested columns name
column_arr = ['orig_STR', 'dest_STR', 'pick_datetime_STR', 'drop_datetime_STR', 'passenger_count_STR']

# Get zone information
zone_df = pd.read_csv(folder+zone_fname, index_col=0)
# Remove unknown zone
zone_df = zone_df[zone_df['Borough'] != 'Unknown']

filter_zone_index = len(zone_df.index)

# Get nyctaxi information
files = getCSVfile(folder+'data')



# Variables to keep track removed samples
samples_removed_dict = {}

# Read taxi data CSV files
for i,f in enumerate(files):
    # Default variable 
    # column name for origin pickup location
    orig_STR = 'pulocationid'
    # column name for destination dropoff location
    dest_STR = 'dolocationid'
    # column name for pickup date time 
    pick_datetime_STR = 'pickup_datetime'
    # column name for dropoff date time 
    drop_datetime_STR = 'dropoff_datetime'
    # column name for passenger count
    passenger_count_STR = 'passenger_count'

    try:
        print('Working on file: ', f)
        fname = f.split('_')[-1].split('-')
        _year = int(fname[0])
        _month = fname[1].replace('.csv','')
        _taxi = ''
        for t in taxi_arr:
            if t in f:
                _taxi = t
                break
        #print(_year,_month,_taxi)
        
                
        df = readCSVfile(f)
        total_samples = len(df.index)
        print('Total sample size: ', total_samples)
        ###
        # Get the right column naming for pickup and dropoff datetime
        for c in df.columns:
            if pick_datetime_STR in c:
                pick_datetime_STR = c
            if drop_datetime_STR in c:
                drop_datetime_STR = c
        
        # check if passenger_count columns exist:
        if passenger_count_STR not in df.columns:
            df[passenger_count_STR] = [1 for idx in df.index]

        
        ###
        # Change the date-time columns from object to datetime
        df[pick_datetime_STR] = pd.to_datetime(df[pick_datetime_STR], format='%Y%m%d %H:%M:%S', errors='coerce')
        df[drop_datetime_STR] = pd.to_datetime(df[drop_datetime_STR], format='%Y%m%d %H:%M:%S', errors='coerce')

        
        
        ###
        # Remove samples with nan values - for column of interest
        total_nan = 0
        for c in column_arr:
            col = globals()[c]
            if col in df.columns:
                total_nan += len(df[~df[col].notna()].index)
                df = df[df[col].notna()]
        print('Sample(s) removed due to NAN: ', total_nan)

        # Remove samples with unknown location (id > 263) - for column PULocationID and DOLocationID
        clean_indexes = df[(df[orig_STR]<=filter_zone_index) & (df[dest_STR]<=filter_zone_index)].index
        total_unknown = len(df.index)-len(clean_indexes)
        print('Sample(s) removed due to unknown zone: ', total_unknown)
        df = df.loc[clean_indexes]
        
        # Remove samples due to different year
        clean_indexes = df.loc[~(df[pick_datetime_STR].dt.year > _year+1) | (df[pick_datetime_STR].dt.year < _year-1)].index
        total_wrong_year = len(df.index)-len(clean_indexes)
        print('Sample(s) removed due to different / wrong year: ', total_wrong_year)
        df = df.loc[clean_indexes]
        
        # Save removed information to dictionary
        samples_removed_dict[f] = {}
        samples_removed_dict[f]['nan'] = total_nan
        samples_removed_dict[f]['unknown'] = total_unknown
        samples_removed_dict[f]['wrongyear'] = total_wrong_year
        samples_removed_dict[f]['total_samples'] = total_samples
        
        
        
        ###
        # Group samples based on pickup date/time and location 
        gtime = pd.to_datetime(df[pick_datetime_STR])
        orig_data_df = df.groupby(by=[gtime.dt.year, gtime.dt.month, gtime.dt.day, gtime.dt.hour, orig_STR])[passenger_count_STR].sum()
        orig_data_df.index.rename(["year", "month", "day", "hour", "locationID"], inplace=True)

        # Group samples based on dropoff date/time and location
        gtime = pd.to_datetime(df[drop_datetime_STR])
        dest_data_df = df.groupby(by=[gtime.dt.year, gtime.dt.month, gtime.dt.day, gtime.dt.hour, dest_STR])[passenger_count_STR].sum()
        dest_data_df.index.rename(["year", "month", "day", "hour", "locationID"], inplace=True)


        # AVOID DUPLICATE Counts - Ignore this if forecasting taxi demand
        # 1. For samples origin and destination ID are same
        duplicate_df = df.loc[(df[orig_STR] == df[dest_STR])]
        #### Based on origin-destination within hour
        duplicate_df = duplicate_df.loc[(duplicate_df[pick_datetime_STR].dt.hour == duplicate_df[drop_datetime_STR].dt.hour)]
        gtime = pd.to_datetime(duplicate_df[pick_datetime_STR])
        dupl_data_df = duplicate_df.groupby(by=[gtime.dt.year, gtime.dt.month, gtime.dt.day, gtime.dt.hour, orig_STR])[passenger_count_STR].sum()
        dupl_data_df.index.rename(["year", "month", "day", "hour", "locationID"], inplace=True)

        # Merged all the data together
        combined_data_df = pd.concat([orig_data_df,dest_data_df,dupl_data_df], axis=1).fillna(0)
        combined_data_df.columns = ['origin_passenger_count', 'destination_passenger_count', 'duplicate_passenger_count']
        combined_data_df['total_crowd'] = combined_data_df['origin_passenger_count'] + combined_data_df['destination_passenger_count'] - combined_data_df['duplicate_passenger_count']


        ###
        # Save to file temporarily. Get Year and Month (based on first index)
        print('Saving segmented result to file')
        combined_data_df.to_csv(new_folder+'segmented_output_'+str(_year)+'_'+str(_month)+'_'+_taxi+'.csv', header=True, index=True)

        print('Done.')
        print('---------------------------------------------------------------------------------------------------')
    except Exception as e:
        print('Error: ', e)
        print('---------------------------------------------------------------------------------------------------')
        continue
        



# Save samples_removed_dictionary into file
removed_dict_df = pd.DataFrame(samples_removed_dict).T
removed_dict_df.to_csv(new_folder+'_removed_samples__.csv')



###
# Combine all the segmented files based on the same index

df = None
# Get all the segmented files
files = getCSVfile(new_folder)

for f in files:
    if df is None:
        df = readCSVfile(f)
    else:
        tmp_df = readCSVfile(f)
        # Merge the two files together based on the 5 ids
        df = df.set_index(['year','month','day','hour','locationid']).add(tmp_df.set_index(['year','month','day','hour','locationid']), fill_value=0).reset_index()



# Read file (optional)
# df = readCSVfile(new_folder+'summary_segmented_output.csv)

###
# Final Cleaning of the merged data
# 1. Exclude year that is not in year_arr
df = df[df['year'].isin(year_arr)]
# 2. Only include columns that we want
df.drop(['origin_passenger_count', 'destination_passenger_count', 'duplicate_passenger_count'], axis=1, inplace=True)



###
# Add New Temporal Features
df['datetime'] = df.apply(lambda x:datetime.strptime("{0} {1} {2}".format(int(x['year']),int(x['month']), int(x['day'])), "%Y %m %d"),axis=1)
# 1. Day of Week (1,2,3,4,5,6,7)
df['dayofweek'] = df['datetime'].dt.dayofweek
# 2. Quarter of year (1,2,3,4)
df['quarter'] = df['datetime'].dt.quarter
# 3. Week of year (1,to 52)
df['weekofyear'] = df['datetime'].dt.week



# Save to File
if 'datetime' in df.columns:
    df.drop(['datetime'], axis=1, inplace=True)
df = df.astype(int)
df.to_csv(new_folder+'crowd_flow.csv', header=True)



# Save the files into different segment
for y in year_arr:
    df[df['year']==y].to_csv(new_folder+'crowd_flow_'+str(y)+'.csv', header=True)


