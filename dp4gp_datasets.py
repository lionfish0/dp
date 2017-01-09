# Methods for loading dataset for demonstrating the DP4GP tools

import pandas as pd
from datetime import datetime
import numpy as np
import os

def add_citibike_extra_columns(df):
    """
    Add columns to citibike dataframe:
       - with seconds, minutes and hours since start of week
       - day of week
       - tripduration in mins
       - tripduration in hours
     Alters the dataframe inplace
    """
    seconds = np.zeros(df.shape[0])
    dow = np.zeros(df.shape[0])
    for i,p in enumerate(df.iterrows()):
        hiredatetime = datetime.strptime(p[1]['starttime'], '%m/%d/%Y %H:%M:%S')
        midnight = hiredatetime.replace(hour=0, minute=0, second=0, microsecond=0)
        dow[i] = hiredatetime.weekday()    
        seconds[i] = (hiredatetime - midnight).seconds + dow[i]*(3600*24.0)
    df['seconds'] = seconds #total number of seconds
    df['hours'] = seconds/3600.0 #total number of hours
    df['mins'] = seconds/60.0 #total number of hours

    df['dow'] = dow
    df['tripduration_mins'] = df['tripduration']/60.0
    df['tripduration_hours'] = df['tripduration']/3600.0
    
    
def load_citibike():
    """
    Download and load station 300 data for June 2016.
    Returns a pandas dataframe
    """
    if not os.path.isfile('201606-citibike-tripdata.csv'):
        os.system('wget https://s3.amazonaws.com/tripdata/201606-citibike-tripdata.zip')
        os.system('unzip 201606-citibike-tripdata.zip')
    full_df = pd.read_csv('201606-citibike-tripdata.csv')

    df = full_df[full_df['start station id']==300].copy() #we'll just use one station (number 300)
    return df

def load_pricepaid():
    """
    Download and load UK housing price data from the Land Registry, 2016
    Returns panda dataframe with just the price and postcode
    """
    #for the year's data, use: http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/pp-2016.txt

    filename = "pp-monthly-update-new-version.csv"
    if not os.path.isfile(filename):
        os.system('wget http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/'+filename)
    pp = pd.read_csv(filename,header=None,usecols=[1,3],names=["price", "postcode"])
    return pp
    
def load_postcode():
    """
    Download and load UK postcode locations, from 'freepostcodes.org.uk', where
    they've converted the large ordnance survey datafiles in to smaller csv files.
    
    This takes a little while to run
    
    Returns a panda dataframe with the postcode, easting and northing
    """
    
    unzip_data_path = "Code-Point Open/Data"
    if not os.path.isfile(unzip_data_path+"/ab.csv"):
        os.system("wget http://www.freepostcodes.org.uk/static/code-point-open/codepo_gb.zip")
        os.system("unzip codepo_gb.zip")

    df = pd.DataFrame()
    for filename in os.listdir(unzip_data_path): 
        df = df.append( pd.read_csv(unzip_data_path+"/"+filename,header=None,usecols=[0,10,11],names=["postcode","easting","northing"]) )
    return df

def load_prices_and_postcode():
    """
    Download and load both the prices and postcodes, and inner join the two tables
    
    Returns a dataframe with the postcode, price, easting and northing
    """
    pp = load_pricepaid()
    pc = load_postcode()
    complete = pd.merge(pc,pp,on="postcode",how="inner")
    return complete
    
def load_fishlength():
    """
    Returns a matrix of:
    fish index, age (days), temp (C), length (inches)
    #data from Freund, R. J., & Minton, P. D. (1979). Regression methods: a tool for data analysis (new york). Dekker, (p. 111).
    #note that they don't provide what unit the length is in! But likely to be in thou (thousandths of an inch)
    """
    data = np.array([[1.0,14,25,620],
    [2,28,25,1315],
    [3,41,25,2120],
    [4,55,25,2600],
    [5,69,25,3110],
    [6,83,25,3535],
    [7,97,25,3935],
    [8,111,25,4465],
    [9,125,25,4530],
    [10,139,25,4570],
    [11,153,25,4600],
    [12,14,27,625],
    [13,28,27,1215],
    [14,41,27,2110],
    [15,55,27,2805],
    [16,69,27,3255],
    [17,83,27,4015],
    [18,97,27,4315],
    [19,111,27,4495],
    [20,125,27,4535],
    [21,139,27,4600],
    [22,153,27,4600],
    [23,14,29,590],
    [24,28,29,1305],
    [25,41,29,2140],
    [26,55,29,2890],
    [27,69,29,3920],
    [28,83,29,3920],
    [29,97,29,4515],
    [30,111,29,4520],
    [31,125,29,4525],
    [32,139,29,4565],
    [33,153,29,4566]])

    #conversion from thou to inches
    data[:,3] = data[:,3] / 1000.0
    return data

