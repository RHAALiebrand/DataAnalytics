import pandas as pd
import numpy as np
import datetime

import pandas_datareader.data as pd_data
import pandas_datareader.wb as pd_wb
import time

## FINANCIAL CRISIS
date_start_FC=datetime.datetime(2008,1,1)
date_end_FC=datetime.datetime(2013,1,1)


# EURO Banks
ING = pd_data.DataReader('ING', 'av-daily', date_start_FC, date_end_FC,api_key='I6LLFEJHN7JJRVKI') # ABN listed at AEX
DDB = pd_data.DataReader('DB', 'av-daily', date_start_FC, date_end_FC,api_key='I6LLFEJHN7JJRVKI') # DB listed at ETR
HSB = pd_data.DataReader('HSBC', 'av-daily', date_start_FC, date_end_FC,api_key='I6LLFEJHN7JJRVKI') # HSBC listed at LON

time.sleep(60)

# US Banks
CIT = pd_data.DataReader('C', 'av-daily', date_start_FC, date_end_FC,api_key='I6LLFEJHN7JJRVKI') # C listed at NYSE
GMS = pd_data.DataReader('GS', 'av-daily', date_start_FC, date_end_FC,api_key='I6LLFEJHN7JJRVKI') # GS listed at NYSE
JPM = pd_data.DataReader('JPM', 'av-daily', date_start_FC, date_end_FC,api_key='I6LLFEJHN7JJRVKI') # JPM listed at NYSE

EU = pd.concat([ING, DDB, HSB],axis=1,keys='ING DDB HSB'.split())
US = pd.concat([CIT, GMS, JPM],axis=1,keys='CIT GMS JPM'.split())
stocks_FC = pd.concat([EU, US],axis=1,keys='EU US'.split())
stocks_FC.columns.names = ['Region','Bank','Stock']
#stocks_FC.to_pickle('stocks_FC')
print(stocks_FC.head())

## COVID Crisis
date_start_COVID=datetime.datetime(2020,2,1)
date_end_COVID=datetime.datetime(2020,7,1) # I update this everytime I check this notebook

# EURO Banks
ING = pd_data.DataReader('ING', 'av-daily', date_start_COVID, date_end_COVID,api_key='I6LLFEJHN7JJRVKI') # ABN listed at AEX
DDB = pd_data.DataReader('DB', 'av-daily', date_start_COVID, date_end_COVID,api_key='I6LLFEJHN7JJRVKI') # DB listed at ETR
HSB = pd_data.DataReader('HSBC', 'av-daily', date_start_COVID, date_end_COVID,api_key='I6LLFEJHN7JJRVKI') # HSBC listed at LON

time.sleep(60)

# US Banks
CIT = pd_data.DataReader('C', 'av-daily', date_start_COVID, date_end_COVID,api_key='I6LLFEJHN7JJRVKI') # C listed at NYSE
GMS = pd_data.DataReader('GS', 'av-daily', date_start_COVID, date_end_COVID,api_key='I6LLFEJHN7JJRVKI') # GS listed at NYSE
JPM = pd_data.DataReader('JPM', 'av-daily', date_start_COVID, date_end_COVID,api_key='I6LLFEJHN7JJRVKI') # JPM listed at NYSE

EU = pd.concat([ING, DDB, HSB],axis=1,keys='ING DDB HSB'.split())
US = pd.concat([CIT, GMS, JPM],axis=1,keys='CIT GMS JPM'.split())
stocks_COVID = pd.concat([EU, US],axis=1,keys='EU US'.split())
stocks_COVID.columns.names = ['Region','Bank','Stock']
#stocks_COVID.to_pickle('stocks_COVID')
print(stocks_COVID.head())

