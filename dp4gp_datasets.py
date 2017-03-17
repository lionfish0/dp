import pandas as pd
from datetime import datetime
import numpy as np
import os

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

def load_malawichildren():
    """
    Returns a matrix of:
    Age,Weight,Height,MUAC
    #age in years
    #weight in kg
    #height in cm
    #MUAC in cm
    """
    data = np.array([[3.59,15.2,94,15.6],
    [5.00,16.7,106,15.6],
    [4.92,16.7,97,17.6],
    [6.06,16.9,108,15.1],
    [2.52,13.6,86,15.1],
    [4.69,18.1,110,15.5],
    [5.27,15.9,102,14.9],
    [6.20,14.3,103,14.3],
    [5.42,18.8,105,15.9],
    [3.36,10.5,84,13.4],
    [6.35,18.6,114,16.2],
    [4.38,15.4,100,15.8],
    [6.16,15.9,108,15.4],
    [5.67,16.6,106,15.6],
    [4.39,16.4,105,14.6],
    [2.88,12.9,91,15.1],
    [6.08,17.2,110,15],
    [5.33,13.7,99,16.5],
    [5.87,16.5,105,14.8],
    [5.13,17.2,106,15.4],
    [4.24,15.2,99,16.2],
    [4.98,15.5,95,15.4],
    [3.93,14.8,97,15.4],
    [3.28,15.2,98,16.7],
    [1.98,9.8,78,13.1],
    [7.48,21.5,115,17.9],
    [2.75,13.2,97,14.2],
    [5.15,20.2,111,16.4],
    [4.45,14.9,95,16.3],
    [5.23,15.9,101,15.4],
    [1.87,10.6,80,14.1],
    [5.49,15.3,105,14.2],
    [6.18,19.2,108,15.6],
    [4.69,15.5,96,15.1],
    [5.71,15.6,103,14.4],
    [4.85,14.9,99,14.9],
    [5.13,14.3,99,14.2],
    [5.25,18.6,111,16.4],
    [5.99,18.1,110,14.4],
    [6.24,14.5,102,14.2],
    [4.16,14.1,100,13],
    [6.18,16.1,105,13.2],
    [5.35,17.2,110,15.6],
    [4.69,16.9,106,14.8],
    [5.08,16.5,102,16],
    [6.14,15.4,104,15],
    [5.20,16.8,99,17.7],
    [4.95,18,100,16.5],
    [5.76,17,107,15.3],
    [5.35,14.6,98,15.1],
    [4.54,17.1,95,18.2],
    [2.44,12.4,89,14],
    [3.29,11.8,85,14.8],
    [3.78,12,88,14.2],
    [5.32,14.8,94,16.3],
    [3.23,14.1,91,16.2],
    [2.20,15.4,87,17.2],
    [5.03,17.3,107,15.8],
    [4.50,16,97,16.5],
    [4.87,16.4,103,15],
    [3.49,12.9,88,15.4],
    [3.64,14.5,92,15.2],
    [3.42,14.2,90,16.5],
    [5.58,17.1,104,16],
    [4.01,14.7,96,16.2],
    [5.16,17.9,110,15.8],
    [5.11,17.7,106,14.6],
    [5.33,15,103,14.6],
    [5.25,17.6,112,14.1],
    [4.30,16.3,102,14.8],
    [5.13,15.8,101,15.7],
    [6.18,15.8,104,15],
    [5.32,15,107,13.1],
    [3.29,16.2,94,17.5],
    [6.12,16.6,111,14.5],
    [4.82,19.5,105,17.6],
    [5.44,17.3,105,15.8],
    [4.85,15.3,96,16.1],
    [3.20,12.5,86,14.6],
    [6.22,17.1,105,15.1],
    [6.53,18.9,110,15.8],
    [6.50,18.8,109,15.4],
    [3.08,11.6,89,13.5],
    [5.44,17.9,108,16.8],
    [3.34,13.5,96,14.2],
    [5.90,16.7,106,15.3],
    [5.01,15.6,98,15.5],
    [6.12,17,103,15.9],
    [6.07,17.2,104,15.4],
    [3.88,15.2,99,15.1],
    [5.01,17.9,102,17.2],
    [3.21,13.5,89,15.2],
    [4.94,16.2,100,16.2]])
    return data

def load_kung():
    """
    Returns a matrix of:
    Height, weight, age and sex
    #height in cm
    #weight in kg
    #age in years
    #gender (1=male)
    
    The height, weight, age, and sex from a partial census of the Dobe area 
    !Kung San, is shown here, compiled from interviews conducted by Nancy Howell
    from 1967 to 1969. The !Kung are a San people living in the Kalahari Desert 
    in Namibia, Botswana and in Angola. They speak the !Kung language, noted for
    its extensive use of click consonants.
    
    #https://public.tableau.com/profile/john.marriott#!/vizhome/kung-san/Attributes
    #https://github.com/rmcelreath/rethinking/blob/master/data/Howell2.csv
    """

    data = np.array([[151.765,47.8256065,63,1],
    [139.7,36.4858065,63,0],
    [136.525,31.864838,65,0],
    [156.845,53.0419145,41,1],
    [145.415,41.276872,51,0],
    [163.83,62.992589,35,1],
    [149.225,38.2434755,32,0],
    [168.91,55.4799715,27,1],
    [147.955,34.869885,19,0],
    [165.1,54.487739,54,1],
    [154.305,49.89512,47,0],
    [151.13,41.220173,66,1],
    [144.78,36.0322145,73,0],
    [149.9,47.7,20,0],
    [150.495,33.849303,65.3,0],
    [163.195,48.5626935,36,1],
    [157.48,42.3258035,44,1],
    [143.9418,38.3568735,31,0],
    [121.92,19.617854,12,1],
    [105.41,13.947954,8,0],
    [86.36,10.489315,6.5,0],
    [161.29,48.987936,39,1],
    [156.21,42.7226965,29,0],
    [129.54,23.586784,13,1],
    [109.22,15.989118,7,0],
    [146.4,35.493574,56,1],
    [148.59,37.9032815,45,0],
    [147.32,35.4652245,19,0],
    [137.16,27.328918,17,1],
    [125.73,22.6796,16,0],
    [114.3,17.860185,11,1],
    [147.955,40.312989,29,1],
    [161.925,55.111428,30,1],
    [146.05,37.5063885,24,0],
    [146.05,38.498621,35,0],
    [152.7048,46.606578,33,0],
    [142.875,38.838815,27,0],
    [142.875,35.5786225,32,0],
    [147.955,47.400364,36,0],
    [160.655,47.8823055,24,1],
    [151.765,49.4131785,30,1],
    [162.8648,49.384829,24,1],
    [171.45,56.5572525,52,1],
    [147.32,39.12231,42,0],
    [147.955,49.89512,19,0],
    [144.78,28.803092,17,0],
    [121.92,20.41164,8,1],
    [128.905,23.359988,12,0],
    [97.79,13.267566,5,0],
    [154.305,41.2485225,55,1],
    [143.51,38.55532,43,0],
    [146.7,42.4,20,1],
    [157.48,44.6504625,18,1],
    [127,22.0105518,13,1],
    [110.49,15.422128,9,0],
    [97.79,12.757275,5,0],
    [165.735,58.5984165,42,1],
    [152.4,46.719976,44,0],
    [141.605,44.22522,60,0],
    [158.8,50.9,20,0],
    [155.575,54.317642,37,0],
    [164.465,45.8978405,50,1],
    [151.765,48.024053,50,0],
    [161.29,52.219779,31,1],
    [154.305,47.62716,25,0],
    [145.415,45.642695,23,0],
    [145.415,42.410852,52,0],
    [152.4,36.4858065,79.3,1],
    [163.83,55.9335635,35,1],
    [144.145,37.194544,27,0],
    [129.54,24.550667,13,1],
    [129.54,25.627948,14,0],
    [153.67,48.307548,38,1],
    [142.875,37.3362915,39,0],
    [146.05,29.596878,12,0],
    [167.005,47.173568,30,1],
    [158.4198,47.286966,24,0],
    [91.44,12.927372,0.59909,1],
    [165.735,57.549485,51,1],
    [149.86,37.931631,46,0],
    [147.955,41.900561,17,0],
    [137.795,27.5840635,12,0],
    [154.94,47.2019175,22,0],
    [160.9598,43.204638,29,1],
    [161.925,50.2636635,38,1],
    [147.955,39.3774555,30,0],
    [113.665,17.463292,6,1],
    [159.385,50.689,45,1],
    [148.59,39.4341545,47,0],
    [136.525,36.28736,79,0],
    [158.115,46.266384,45,1],
    [144.78,42.2691045,54,0],
    [156.845,47.62716,31,1],
    [179.07,55.7067675,23,1],
    [118.745,18.824068,9,0],
    [170.18,48.5626935,41,1],
    [146.05,42.807745,23,0],
    [147.32,35.0683315,36,0],
    [113.03,17.8885345,5,1],
    [162.56,56.755699,30,0],
    [133.985,27.442316,12,1],
    [152.4,51.255896,34,0],
    [160.02,47.230267,44,1],
    [149.86,40.936678,43,0],
    [142.875,32.715323,73.3,0],
    [167.005,57.0675435,38,1],
    [159.385,42.977842,43,1],
    [154.94,39.9444455,33,0],
    [148.59,32.4601775,16,0],
    [111.125,17.123098,11,1],
    [111.76,16.499409,6,1],
    [162.56,45.9545395,35,1],
    [152.4,41.106775,29,0],
    [124.46,18.257078,12,0],
    [111.76,15.081934,9,1],
    [86.36,11.4815475,7.5991,1],
    [170.18,47.5988105,58,1],
    [146.05,37.5063885,53,0],
    [159.385,45.019006,51,1],
    [151.13,42.2691045,48,0],
    [160.655,54.8562825,29,1],
    [169.545,53.523856,41,1],
    [158.75,52.1914295,81.75,1],
    [74.295,9.752228,1,1],
    [149.86,42.410852,35,0],
    [153.035,49.5832755,46,0],
    [96.52,13.097469,5,1],
    [161.925,41.730464,29,1],
    [162.56,56.018612,42,1],
    [149.225,42.1557065,27,0],
    [116.84,19.391058,8,0],
    [100.076,15.081934,6,1],
    [163.195,53.0986135,22,1],
    [161.925,50.235314,43,1],
    [145.415,42.52425,53,0],
    [163.195,49.101334,43,1],
    [151.13,38.498621,41,0],
    [150.495,49.8100715,50,0],
    [141.605,29.313383,15,1],
    [170.815,59.760746,33,1],
    [91.44,11.7083435,3,0],
    [157.48,47.9390045,62,1],
    [152.4,39.292407,49,0],
    [149.225,38.1300775,17,1],
    [129.54,21.999212,12,0],
    [147.32,36.8826995,22,0],
    [145.415,42.127357,29,0],
    [121.92,19.787951,8,0],
    [113.665,16.782904,5,1],
    [157.48,44.565414,33,1],
    [154.305,47.853956,34,0],
    [120.65,21.1770765,12,0],
    [115.6,18.9,7,1],
    [167.005,55.1964765,42,1],
    [142.875,32.998818,40,0],
    [152.4,40.879979,27,0],
    [96.52,13.267566,3,0],
    [160,51.2,25,1],
    [159.385,49.044635,29,1],
    [149.86,53.4388075,45,0],
    [160.655,54.090846,26,1],
    [160.655,55.3665735,45,1],
    [149.225,42.240755,45,0],
    [125.095,22.3677555,11,0],
    [140.97,40.936678,85.599,0],
    [154.94,49.6966735,26,1],
    [141.605,44.338618,24,0],
    [160.02,45.9545395,57,1],
    [150.1648,41.95726,22,0],
    [155.575,51.482692,24,0],
    [103.505,12.757275,6,0],
    [94.615,13.0124205,4,0],
    [156.21,44.111822,21,0],
    [153.035,32.205032,79,0],
    [167.005,56.755699,50,1],
    [149.86,52.673371,40,0],
    [147.955,36.4858065,64,0],
    [159.385,48.8461885,32,1],
    [161.925,56.9541455,38.7,1],
    [155.575,42.0990075,26,0],
    [159.385,50.178615,63,1],
    [146.685,46.549879,62,0],
    [172.72,61.80191,22,1],
    [166.37,48.987936,41,1],
    [141.605,31.524644,19,1],
    [142.875,32.205032,17,0],
    [133.35,23.756881,14,0],
    [127.635,24.4089195,9,1],
    [119.38,21.5172705,7,1],
    [151.765,35.2951275,74,0],
    [156.845,45.642695,41,1],
    [148.59,43.885026,33,0],
    [157.48,45.5576465,53,0],
    [149.86,39.008912,18,0],
    [147.955,41.163474,37,0],
    [102.235,13.1258185,6,0],
    [153.035,45.245802,61,0],
    [160.655,53.637254,44,1],
    [149.225,52.3048275,35,0],
    [114.3,18.3421265,7,1],
    [100.965,13.7495075,4,1],
    [138.43,39.0939605,23,0],
    [91.44,12.530479,4,1],
    [162.56,45.699394,55,1],
    [149.225,40.3980375,53,0],
    [158.75,51.482692,59,1],
    [149.86,38.668718,57,0],
    [158.115,39.235708,35,1],
    [156.21,44.338618,29,0],
    [148.59,39.519203,62,1],
    [143.51,31.071052,18,0],
    [154.305,46.776675,51,0],
    [131.445,22.509503,14,0],
    [157.48,40.6248335,19,1],
    [157.48,50.178615,42,1],
    [154.305,41.276872,25,0],
    [107.95,17.57669,6,1],
    [168.275,54.6,41,1],
    [145.415,44.9906565,37,0],
    [147.955,44.735511,16,0],
    [100.965,14.401546,5,1],
    [113.03,19.050864,9,1],
    [149.225,35.8054185,82,1],
    [154.94,45.2174525,28,1],
    [162.56,48.1091015,50,1],
    [156.845,45.6710445,43,0],
    [123.19,20.808533,8,1],
    [161.0106,48.420946,31,1],
    [144.78,41.1918235,67,0],
    [143.51,38.4135725,39,0],
    [149.225,42.127357,18,0],
    [110.49,17.6617385,11,0],
    [149.86,38.2434755,48,0],
    [165.735,48.3358975,30,1],
    [144.145,38.9238635,64,0],
    [157.48,40.029494,72,1],
    [154.305,50.2069645,68,0],
    [163.83,54.2892925,44,1],
    [156.21,45.6,43,0],
    [153.67,40.766581,16,0],
    [134.62,27.1304715,13,0],
    [144.145,39.4341545,34,0],
    [114.3,20.4966885,10,0],
    [162.56,43.204638,62,1],
    [146.05,31.864838,44,0],
    [120.65,20.8935815,11,1],
    [154.94,45.4442485,31,1],
    [144.78,38.045029,29,0],
    [106.68,15.989118,8,0],
    [146.685,36.0889135,62,0],
    [152.4,40.879979,67,0],
    [163.83,47.910655,57,1],
    [165.735,47.7122085,32,1],
    [156.21,46.379782,24,0],
    [152.4,41.163474,77,1],
    [140.335,36.5992045,62,0],
    [158.115,43.09124,17,1],
    [163.195,48.137451,67,1],
    [151.13,36.7126025,70,0],
    [171.1198,56.5572525,37,1],
    [149.86,38.6970675,58,0],
    [163.83,47.4854125,35,1],
    [141.605,36.2023115,30,0],
    [93.98,14.288148,5,0],
    [149.225,41.276872,26,0],
    [105.41,15.2236815,5,0],
    [146.05,44.7638605,21,0],
    [161.29,50.4337605,41,1],
    [162.56,55.281525,46,1],
    [145.415,37.931631,49,0],
    [145.415,35.493574,15,1],
    [170.815,58.456669,28,1],
    [127,21.488921,12,0],
    [159.385,44.4236665,83,0],
    [159.4,44.4,54,1],
    [153.67,44.565414,54,0],
    [160.02,44.622113,68,1],
    [150.495,40.483086,68,0],
    [149.225,44.0834725,56,0],
    [127,24.4089195,15,0],
    [142.875,34.416293,57,0],
    [142.113,32.772022,22,0],
    [147.32,35.947166,40,0],
    [162.56,49.5549,19,1],
    [164.465,53.183662,41,1],
    [160.02,37.081146,75.901,1],
    [153.67,40.5114355,73.901,0],
    [167.005,50.6038575,49,1],
    [151.13,43.9700745,26,1],
    [147.955,33.792604,17,0],
    [125.3998,21.375523,13,0],
    [111.125,16.669506,8,0],
    [153.035,49.89,88,1],
    [139.065,33.5941575,68,0],
    [152.4,43.8566765,33,1],
    [154.94,48.137451,26,0],
    [147.955,42.751046,56,0],
    [143.51,34.8415355,16,1],
    [117.983,24.097075,13,0],
    [144.145,33.906002,34,0],
    [92.71,12.076887,5,0],
    [147.955,41.276872,17,0],
    [155.575,39.7176495,74,1],
    [150.495,35.947166,69,0],
    [155.575,50.915702,50,1],
    [154.305,45.756093,44,0],
    [130.6068,25.2594045,15,0],
    [101.6,15.3370795,5,0],
    [157.48,49.214732,18,0],
    [168.91,58.8252125,41,1],
    [150.495,43.4597835,27,0],
    [111.76,17.8318355,8.9009,1],
    [160.02,51.9646335,38,1],
    [167.64,50.688906,57,1],
    [144.145,34.246196,64.5,0],
    [145.415,39.3774555,42,0],
    [160.02,59.5622995,24,1],
    [147.32,40.312989,16,1],
    [164.465,52.16308,71,1],
    [153.035,39.972795,49.5,0],
    [149.225,43.941725,33,1],
    [160.02,54.601137,28,0],
    [149.225,45.075705,47,0],
    [85.09,11.453198,3,1],
    [84.455,11.7650425,1,1],
    [59.6138,5.896696,1,0],
    [92.71,12.1052365,3,1],
    [111.125,18.313777,6,0],
    [90.805,11.3681495,5,0],
    [153.67,41.333571,27,0],
    [99.695,16.2442635,5,0],
    [62.484,6.80388,1,0],
    [81.915,11.8784405,2,1],
    [96.52,14.968536,2,0],
    [80.01,9.865626,1,1],
    [150.495,41.900561,55,0],
    [151.765,42.524,83.401,1],
    [140.6398,28.859791,12,1],
    [88.265,12.7856245,2,0],
    [158.115,43.147939,63,1],
    [149.225,40.82328,52,0],
    [151.765,42.864444,49,1],
    [154.94,46.209685,31,0],
    [123.825,20.581737,9,0],
    [104.14,15.87572,6,0],
    [161.29,47.853956,35,1],
    [148.59,42.52425,35,0],
    [97.155,17.066399,7,0],
    [93.345,13.1825175,5,1],
    [160.655,48.5059945,24,1],
    [157.48,45.869491,41,1],
    [167.005,52.900167,32,1],
    [157.48,47.570461,43,1],
    [91.44,12.927372,6,0],
    [60.452,5.6699,1,1],
    [137.16,28.91649,15,1],
    [152.4,43.544832,63,0],
    [152.4,43.431434,21,0],
    [81.28,11.509897,1,1],
    [109.22,11.7083435,2,0],
    [71.12,7.540967,1,1],
    [89.2048,12.700576,3,0],
    [67.31,7.200773,1,0],
    [85.09,12.360382,1,1],
    [69.85,7.7961125,1,0],
    [161.925,53.2120115,55,0],
    [152.4,44.678812,38,0],
    [88.9,12.5588285,3,1],
    [90.17,12.700576,3,1],
    [71.755,7.37087,1,0],
    [83.82,9.2135875,1,0],
    [159.385,47.2019175,28,1],
    [142.24,28.632995,16,0],
    [142.24,31.6663915,36,0],
    [168.91,56.4438545,38,1],
    [123.19,20.014747,12,1],
    [74.93,8.50485,1,1],
    [74.295,8.3064035,1,0],
    [90.805,11.623295,3,0],
    [160.02,55.791816,48,1],
    [67.945,7.9662095,1,0],
    [135.89,27.21552,15,0],
    [158.115,47.4854125,45,1],
    [85.09,10.8011595,3,1],
    [93.345,14.004653,3,0],
    [152.4,45.1607535,38,0],
    [155.575,45.529297,21,0],
    [154.305,48.874538,50,0],
    [156.845,46.5782285,41,1],
    [120.015,20.128145,13,0],
    [114.3,18.14368,8,1],
    [83.82,10.9145575,3,1],
    [156.21,43.885026,30,0],
    [137.16,27.158821,12,1],
    [114.3,19.050864,7,1],
    [93.98,13.834556,4,0],
    [168.275,56.0469615,21,1],
    [147.955,40.086193,38,0],
    [139.7,26.5634815,15,1],
    [157.48,50.802304,19,0],
    [76.2,9.2135875,1,1],
    [66.04,7.5693165,1,1],
    [160.7,46.3,31,1],
    [114.3,19.4194075,8,0],
    [146.05,37.9032815,16,1],
    [161.29,49.3564795,21,1],
    [69.85,7.314171,0,0],
    [133.985,28.1510535,13,1],
    [67.945,7.824462,0,1],
    [150.495,44.111822,50,0],
    [163.195,51.0291,39,1],
    [148.59,40.766581,44,1],
    [148.59,37.5630875,36,0],
    [161.925,51.59609,36,1],
    [153.67,44.8205595,18,0],
    [68.58,8.0229085,0,0],
    [151.13,43.4030845,58,0],
    [163.83,46.719976,58,1],
    [153.035,39.5475525,33,0],
    [151.765,34.7848365,21.5,0],
    [132.08,22.792998,11,1],
    [156.21,39.292407,26,1],
    [140.335,37.4496895,22,0],
    [158.75,48.6760915,28,1],
    [142.875,35.606972,42,0],
    [84.455,9.3836845,2,1],
    [151.9428,43.714929,21,1],
    [161.29,48.19415,19,1],
    [127.9906,29.8520235,13,1],
    [160.9852,50.972401,48,1],
    [144.78,43.998424,46,0],
    [132.08,28.292801,11,1],
    [117.983,20.354941,8,1],
    [160.02,48.19415,25,1],
    [154.94,39.179009,16,1],
    [160.9852,46.6916265,51,1],
    [165.989,56.415505,25,1],
    [157.988,48.591043,28,1],
    [154.94,48.2224995,26,0],
    [97.9932,13.2959155,5,1],
    [64.135,6.6621325,1,0],
    [160.655,47.4854125,54,1],
    [147.32,35.550273,66,0],
    [146.7,36.6,20,0],
    [147.32,48.9595865,25,0],
    [172.9994,51.255896,38,1],
    [158.115,46.5215295,51,1],
    [147.32,36.967748,48,0],
    [124.9934,25.117657,13,1],
    [106.045,16.272613,6,1],
    [165.989,48.647742,27,1],
    [149.86,38.045029,22,0],
    [76.2,8.50485,1,0],
    [161.925,47.286966,60,1],
    [140.0048,28.3495,15,0],
    [66.675,8.1363065,0,0],
    [62.865,7.200773,0,1],
    [163.83,55.394923,43,1],
    [147.955,32.488527,12,1],
    [160.02,54.204244,27,1],
    [154.94,48.477645,30,1],
    [152.4,43.0628905,29,0],
    [62.23,7.257472,0,0],
    [146.05,34.189497,23,0],
    [151.9936,49.951819,30,0],
    [157.48,41.3052215,17,1],
    [55.88,4.8477645,0,0],
    [60.96,6.23689,0,1],
    [151.765,44.338618,41,0],
    [144.78,33.45241,42,0],
    [118.11,16.896302,7,0],
    [78.105,8.221355,3,0],
    [160.655,47.286966,43,1],
    [151.13,46.1246365,35,0],
    [121.92,20.184844,10,0],
    [92.71,12.757275,3,1],
    [153.67,47.400364,75.5,1],
    [147.32,40.8516295,64,0],
    [139.7,50.348712,38,1],
    [157.48,45.132404,24.2,0],
    [91.44,11.623295,4,0],
    [154.94,42.240755,26,1],
    [143.51,41.6454155,19,0],
    [83.185,9.1568885,2,1],
    [158.115,45.2174525,43,1],
    [147.32,51.255896,38,0],
    [123.825,21.205426,10,1],
    [88.9,11.5949455,3,1],
    [160.02,49.271431,23,1],
    [137.16,27.952607,16,0],
    [165.1,51.199197,49,1],
    [154.94,43.8566765,41,0],
    [111.125,17.690088,6,1],
    [153.67,35.5219235,23,0],
    [145.415,34.246196,14,0],
    [141.605,42.88542,43,0],
    [144.78,32.545226,15,0],
    [163.83,46.776675,21,1],
    [161.29,41.8722115,24,1],
    [154.9,38.2,20,1],
    [161.3,43.3,20,1],
    [170.18,53.637254,34,1],
    [149.86,42.977842,29,0],
    [123.825,21.54562,11,1],
    [85.09,11.4248485,3,0],
    [160.655,39.7743485,65,1],
    [154.94,43.3463855,46,0],
    [106.045,15.478827,8,0],
    [126.365,21.9141635,15,1],
    [166.37,52.673371,43,1],
    [148.2852,38.441922,39,0],
    [124.46,19.27766,12,0],
    [89.535,11.113004,3,1],
    [101.6,13.494362,4,0],
    [151.765,42.807745,43,0],
    [148.59,35.890467,70,0],
    [153.67,44.22522,26,0],
    [53.975,4.252425,0,0],
    [146.685,38.0733785,48,0],
    [56.515,5.159609,0,0],
    [100.965,14.3164975,5,1],
    [121.92,23.2182405,8,1],
    [81.5848,10.659412,3,0],
    [154.94,44.111822,44,1],
    [156.21,44.0267735,33,0],
    [132.715,24.9759095,15,1],
    [125.095,22.5945515,12,0],
    [101.6,14.344847,5,0],
    [160.655,47.8823055,41,1],
    [146.05,39.405805,37.4,0],
    [132.715,24.777463,13,0],
    [87.63,10.659412,6,0],
    [156.21,41.050076,53,1],
    [152.4,40.82328,49,0],
    [162.56,47.0318205,27,0],
    [114.935,17.519991,7,1],
    [67.945,7.2291225,1,0],
    [142.875,34.246196,31,0],
    [76.835,8.0229085,1,1],
    [145.415,31.127751,17,1],
    [162.56,52.16308,31,1],
    [156.21,54.0624965,21,0],
    [71.12,8.051258,0,1],
    [158.75,52.5316235,68,1]])
    return data

