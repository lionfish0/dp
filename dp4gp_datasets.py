import pandas as pd
from datetime import datetime
import numpy as np
import os

import pymc as pm
import numpy as np
import pandas as pd
import re
import xml.etree.ElementTree as ET
import urllib2
import sqlite3 as lite
import urllib
import zipfile,os.path,os
import sqlalchemy as sa
import pandas as pd
import csv
from StringIO import StringIO
from zipfile import ZipFile
import shutil
import time
from datetime import datetime
import random

def adjustpostcode(postcode):
    """Formats postcode into 7 character format, so "a1 2cd" becomes "A1  2CD" or "Gl54 1AB" becomes "GL541AB"."""
    postcode = postcode.upper()
    res = re.search('([A-Z]{1,2}[0-9]{1,2}) *([0-9][A-Z]{2})',postcode);
    if (res==None):
        return postcode #TODO can't understand it, just send it back, need to do something better, throw an exception?
    groups = res.groups()
    if len(groups)==2:
        first = groups[0]
        last = groups[1]
        middle = " "*(7-(len(first)+len(last)))
        return first+middle+last
    return postcode 

def go_get_data(postcodes,dataset,pathToData=''):
    """
    Returns a list of dictionaries, one for each postcode, providing the latitude, longitude, output area and an array of datafor each
    """
    results = []
    geoAreas = []
    for postcode in postcodes:
        pc = adjustpostcode(postcode)
        pathToData = ''
        conn = lite.connect(pathToData+'geo.db')
        geodb = conn.cursor()        
        c_oa = geodb.execute("SELECT oa11, lat, long FROM geo WHERE pcd=?;",(pc,));
        oa = None;
        for r in c_oa:
            results.append({'oa':str(r[0]),'lat':r[1],'lon':r[2],'postcode':postcode})
            geoAreas.append(str(r[0]))

    geoAreaslist = ','.join(geoAreas)    
    #QS414EW
    #url = "http://web.ons.gov.uk/ons/api/data/dataset/QS102EW.xml?context=Census&apikey=cHkIiioOQX&geog=2011STATH&diff=&totals=false&dm/2011STATH=%s" % geoAreaslist
    url = "http://web.ons.gov.uk/ons/api/data/dataset/%s.xml?context=Census&apikey=cHkIiioOQX&geog=2011STATH&diff=&totals=false&dm/2011STATH=%s" % (dataset,geoAreaslist)
    response = urllib2.urlopen(url)
    xmlstring = response.read();
    xmlstring = re.sub('(xmlns:[^=]*)="[^"]*"', '\\1="_"', xmlstring)
    root = ET.fromstring(xmlstring);
    
    data_results = {}
    for a in root.findall("{_}genericData/{_}DataSet/{_}Group/{_}Series"):
        loc = a.find("{_}SeriesKey/{_}Value[@concept='Location']")
        if loc is None:            
            continue
        location_string = loc.attrib['value']
        if location_string not in data_results:
            data_results[location_string] = []
        for dp in a.findall("{_}Obs/{_}ObsValue"):
            data_string = dp.attrib['value']
            data_results[location_string].append( float(data_string) )
    
    for res in results:
        for i,d in enumerate(data_results[res['oa']]):
            res[dataset+"_%d" % i] = d
       #res[dataset] = data_results[res['oa']]
    return results

def get_data(postcodes,dataset,pathToData=''):
    chunksize = 200
    results = []
    while len(postcodes)>0:
        num = min(chunksize,len(postcodes))
        results.extend(go_get_data(postcodes[0:num],dataset,pathToData))
        del postcodes[0:num]
        time.sleep(2) #just give their server some time
        print("%d remaining" % len(postcodes))
    return results

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
    
    
def load_citibike(station=300,year=2016,month=6):
    """
    Download and load station 300 data for June 2016 (default).
    Returns a pandas dataframe
    
    parameters:
        station = id of station, set to None to use all stations
        year = 2016 by default
        month = 6 by default
    """
    yearmonthstring = "%04d%02d" % (year, month)
    if not os.path.isfile('%s-citibike-tripdata.csv' % yearmonthstring):
        os.system('wget https://s3.amazonaws.com/tripdata/%s-citibike-tripdata.zip' % yearmonthstring)
        os.system('unzip %s-citibike-tripdata.zip' % yearmonthstring)
    full_df = pd.read_csv('%s-citibike-tripdata.csv' % yearmonthstring)

    if station is not None:
        df = full_df[full_df['start station id']==station].copy() #we'll just use one station (number 300)
    else:
        df = full_df
        
    return df


def load_pricepaid():
    """
    Download and load UK housing price data from the Land Registry, 2016
    Returns panda dataframe with just the price and postcode
    """
    #for the year's data, use: http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/pp-2016.txt
    # for the whole history of sales use: http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/pp-complete.csv
    
    #Property Type 	D = Detached, S = Semi-Detached, T = Terraced, F = Flats/Maisonettes, O = Other

    #reading the whole landregistry for 24 million records from 1995-2016 takes a lot of time, so we
    #produce a subsampled set of just 300,000 purchases, which are returned instead
    
    filename = "pp-complete.csv"
    if not os.path.isfile('sampled_pp.csv'):        
        if not os.path.isfile(filename):        
            os.system('wget http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/'+filename)
        pp = pd.read_csv(filename,header=None,usecols=[1,2,3,4],names=["price", "date", "postcode", "type"])
        pp = pp.ix[random.sample(pp.index, 300000)]
        pp.to_csv('sampled_pp.csv')
    else:
        print("Using presampled dataset.")
    pp = pd.read_csv('sampled_pp.csv')

    #add seconds since epoch and year.
    seconds = np.zeros(len(pp))
    years = seconds.copy()
    for i,date in enumerate(pp['date']):
        seconds[i] = int(datetime.strptime(date, '%Y-%m-%d %H:%M').strftime("%s"))
        years[i] = int(datetime.strptime(date, '%Y-%m-%d %H:%M').strftime("%Y"))
    pp['seconds'] = seconds
    pp['years'] = years
    print("Loaded property prices.")
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

def add_ons_column(df,dataset):
    """
    This adds a column from the ONS dataset. It makes API queries to the census API, and so it is recommended 
    that it only be used on a reduced dataset.
    """
    x = df['postcode'].values.tolist()
    ons_results = get_data(x,dataset)    
    ons_df = pd.DataFrame(ons_results).drop_duplicates()
    return pd.merge(df,ons_df,on="postcode",how="inner")
    
def setup_postcodes(pathToData):
    """Creates databases and files, downloads data, and populates the datafiles"""       
    url = "https://ago-item-storage.s3-external-1.amazonaws.com/a26683d2393743f4b87c89141cd1b2e8/NSPL_FEB_2017_UK.zip?X-Amz-Security-Token=FQoDYXdzEOz%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaDNnej9L6SZy5Qb3j8iKcA8DP4euIlUueTPtPplc0%2Ft2xEqK558PzosoBZG03VDr5kDJSTHYfvxXUTsaQM3KHYrAJjd7QMzPuzPRV6Vin%2FP6W5ZMa%2FKFmOQ7i33WJF4i9l17HSrq4PzMmfAENbBXVyBvBVSIgSdbZ61RLsunOz1Z%2Fz1%2FLtVFikM20J1ZUsyOeNCuDsgJMqH3KmIiwnfqSJdb%2FqyE2w3%2FBDlw8%2Fn1tGmP01bzL%2BPRk%2BXrNVbCi1Qzv%2F8QqJTjTrLGn3qWNXg48lt86RObkOtpfr9JY26D%2FpvrFZS6%2FAKKryFBBTvKcprjnE9EOpGbS8ouwaOdWg03sK0yoR%2Ffkns%2BoaEdgAmTnvtxGUfg7oxDu%2BczwP7s1ddvyTwUSdKsllN38Rpv%2Bhyb5i35iKdWHqM2pFiBGzIj29%2BCHTs%2BkDXAepj3a194nwxSceMlJUgsIhE3NtSkKkIyFPYR0FMzKapOf3zNXrv9jgS6YfKoVaigMWfFLLQM8RqyRkguT93Zoiz%2BPuJa3GC7f5JRf4EEvICNDPgmgbZY47Vj7AHRECO6S3F7G%2FEAo84r1xQU%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20170306T120558Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAI32ZWKV2CB37RBWQ%2F20170306%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=a038f0108ed73b2e42c7fa994065edb4e5cb5b4e91a2391adfd701aa349d497e"
    print "Creating postcode database in %s" % (pathToData+'geo.db')
    if os.path.isfile(pathToData+'geo.db'):
        print "geo.db exists, skipping"
        return
    print "Downloading "+url
    if os.path.exists('/tmp/psych_postcodes'):
        shutil.rmtree('/tmp/psych_postcodes')
    os.makedirs('/tmp/psych_postcodes')
    urllib.urlretrieve(url, "/tmp/psych_postcodes/postcodes.zip")
    postcode_zipfile = "/tmp/psych_postcodes/postcodes.zip"

    print "Opening postcodes.zip"
    zf = zipfile.ZipFile(postcode_zipfile)
    for f in zf.infolist():       
        zf.extract(f.filename,"/tmp/psych_postcodes")

    print "Importing CSV file to sqlite"  #note:Switched from using pandas as it ran out of memory.


    csvfile = '/tmp/psych_postcodes/Data/NSPL_FEB_2017_UK.csv'
    csvReader = csv.reader(open(csvfile), delimiter=',', quotechar='"')
    conn = lite.connect(pathToData+'geo.db')

    conn.execute('CREATE TABLE IF NOT EXISTS geo (pcd TEXT, oa11 TEXT, lsoa11 TEXT, lat REAL, long REAL)')
    firstRow = True
    n = 0
    for row in csvReader:
        n+=1
        if (n%500000==0):
            print "     %d rows imported" % n                
        if firstRow:
            firstRow = False
            continue
        conn.execute('INSERT INTO geo (pcd, oa11, lsoa11, lat, long) values (?, ?, ?, ?, ?)', (row[0],row[9],row[24],row[32], row[33]))

    print "     Creating indices"        
    conn.execute('CREATE INDEX pcds ON geo(pcd)')
    conn.execute('CREATE INDEX oa11s ON geo(oa11)')
    print "Complete"
    conn.close()
   
def prepare_preloaded_prices(filename, boundingbox=[-np.Inf,-np.Inf,np.Inf,np.Inf], N=10000, col_list=['QS501EW']):
    """
    Create a csv file for a specified region bounded by the boundingbox, of N points
    
    adds columns specified by col_list (defaults to qualifications ('QS501EW'))
    
    boundingbox = [minEast,minNorth,maxEast,maxNorth]
    London: [480e3, 130e3, 580e3, 230e3]
    """

    setup_postcodes('')
    dataset = load_prices_and_postcode()

    samp = (dataset['easting']>boundingbox[0]) & (dataset['easting']<boundingbox[2]) & (dataset['northing']>boundingbox[1]) & (dataset['northing']<boundingbox[3])
    dataset = dataset[samp]
    dataset = dataset.ix[random.sample(dataset.index, N)]

    #adds column of highest qualifications
    for col in col_list:
        dataset = add_ons_column(dataset,col)
    dataset.to_csv(filename)


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

