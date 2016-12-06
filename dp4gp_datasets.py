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
