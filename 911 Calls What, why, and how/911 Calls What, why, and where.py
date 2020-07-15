## Importing the required libraries
# Analysing libs
import numpy as np
import pandas as pd

# Plotting libs
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('whitegrid')
sns.set_context('notebook')
sns.set(font='times new roman',font_scale=1,palette='Greens')

# Make sure I can print more lines in Jupiter....
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "last_expr"

calls = pd.read_csv('911.csv')
calls.head()

print(calls.info())
print(calls.describe())
print(calls.columns)

## LETS DROP THE DUMMY
calls.drop('e',axis=1,inplace=True)
print(calls.columns) #Verification

from mpl_toolkits.basemap import Basemap

## GEO PLOTS AND DEPENDENCEfrom mpl_toolkits.basemap import Basemap
lat = calls['lat'].values
lon = calls['lng'].values

# Draw the map background
fig = plt.figure(figsize=(8, 8))
m = Basemap(projection='lcc', resolution='l', 
             lat_0=calls['lat'].values.mean(), lon_0=calls['lng'].values.mean(),
             width=1E6, height=1.2E6)
m.shadedrelief()

# Scatter lat and long values
m.scatter(lon, lat, latlon=True,
           cmap='Reds', alpha=1.0,s=0.3)
plt.show()

## Now a zoomed plot
# Draw the map background
fig = plt.figure(figsize=(8, 8))
m = Basemap(projection='lcc', resolution='l', 
             lat_0=calls['lat'].values.mean(), lon_0=calls['lng'].values.mean(),
             width=0.5E5, height=0.5E5)
m.shadedrelief()
m.drawcoastlines(color='gray')
m.drawcountries(color='gray')
m.drawstates(color='gray')

 # Scatter lat and long values
m.scatter(lon, lat, latlon=True,
           cmap='Reds', alpha=1.0,s=0.3)
plt.show()


##ZIPCODES
tot_zip=len(calls['zip'].unique()) # Total amount of zip codes
print(tot_zip)

tot_call=calls['zip'].value_counts().sum() # Total amount of calls
print(tot_call)
avg_calls_zip=tot_call/tot_zip
print(avg_calls_zip)

print('The dashed line indicated the total number of calls devided by the number of included zips')
plt.figure()
sns.countplot(x='zip',data=calls,order = calls['zip'].value_counts().index,color='green',alpha=0.5)
plt.plot([0, tot_zip],[avg_calls_zip, avg_calls_zip],color='black',linestyle='--')

plt.figure()
sns.countplot(x='zip',data=calls,order = calls['zip'].value_counts().index[:5],color='green',alpha=0.5)
plt.plot([0, 4],[avg_calls_zip, avg_calls_zip],color='black',linestyle='--')


## TOWNSHIP INFORMATION
tot_twp=len(calls['twp'].unique()) # Total amount of zip codes
avg_calls_twp=tot_call/tot_zip


plt.figure()
sns.countplot(x='twp',data=calls,order = calls['twp'].value_counts().index,color='green',alpha=0.5)
plt.plot([0, tot_twp],[avg_calls_twp, avg_calls_twp],color='black',linestyle='--')

plt.figure()
sns.countplot(x='twp',data=calls,order = calls['twp'].value_counts().index[0:5],color='green',alpha=0.5)
plt.plot([0, 4],[avg_calls_twp, avg_calls_twp],color='black',linestyle='--')


## CALL REASON
def give_reason(title):
    return title.split(':')[0]

calls['reason']=calls['title'].apply(lambda x:give_reason(x))
print(calls.head()) # Verification

print(calls['reason'].unique())

# PLOT RESULTS
plt.figure()
ax=sns.countplot(x='reason',data=calls)
total = float(len(calls))
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,height + 3,'{:1.0f}'.format(int(height/total*100))+'%',ha="center")

## WITH CLASSIFICATION
plt.figure()
sns.countplot(x='zip',data=calls,order = calls['zip'].value_counts().index[0:5],color='green',alpha=0.5,hue='reason')
plt.plot([0, 4],[avg_calls_zip, avg_calls_zip],color='black',linestyle='--')

## TIME DEPENDENCE
type(calls['timeStamp'][0])
calls['timeStamp_conv']=pd.to_datetime(calls['timeStamp']) # Add column to stay with original dataset
type(calls['timeStamp_conv'][0]) #verification

# Convert some data
calls['Year'] = calls['timeStamp_conv'].apply(lambda time: time.year)
calls['Month'] = calls['timeStamp_conv'].apply(lambda time: time.month)
calls['Hour'] = calls['timeStamp_conv'].apply(lambda time: time.hour)
calls['Day'] = calls['timeStamp_conv'].apply(lambda time: time.day)
calls['Day of Week'] = calls['timeStamp_conv'].apply(lambda time: time.dayofweek)

print(calls['Day of Week'].unique())

dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
calls['Day of Week'] = calls['Day of Week'].map(dmap)
print(calls['Day of Week'].unique())

print(calls['Year'].value_counts()/calls['Year'].value_counts().sum())


calls_2016 = calls[calls['Year']>2015]

fig,axes=plt.subplots(nrows=2,ncols=2,figsize=(15,10))

sns.countplot(x='Month',data=calls_2016,ax=axes[0][0],color='green',alpha=0.5)
sns.countplot(x='Month',data=calls_2016,ax=axes[0][1],hue='reason',hue_order='EMS Traffic Fire'.split())
sns.countplot(x='Day of Week',data=calls_2016,ax=axes[1][0],order='Mon Tue Wed Thu Fri Sat Sun'.split(),color='green',alpha=0.5)
sns.countplot(x='Day of Week',data=calls_2016,ax=axes[1][1],hue='reason',hue_order='EMS Traffic Fire'.split())
plt.tight_layout()

## NYE??
calls['Date']=calls['timeStamp_conv'].apply(lambda x: x.date())
print(calls['Date'].head())


fig,axes=plt.subplots(nrows=2,ncols=1,figsize=(15,10))
calls_jan=calls[calls['Month']==1]
#calls_jan.groupby('Date').count()['twp'].plot.bar(figsize=(15,3),label='total',color='green',hue='reason')

sns.countplot(x='Day',data=calls_jan,ax=axes[0],color='green',alpha=0.5)
sns.countplot(x='Day',data=calls_jan,ax=axes[1],hue='reason',hue_order='EMS Traffic Fire'.split())

## GLOBAL VIEW
# Day - hour
calls_overweek = calls.groupby(['Day of Week', 'Hour']).count()['twp'].unstack(level=1)
print(calls_overweek)
calls_overweek=calls_overweek.reindex('Mon Tue Wed Thu Fri Sat Sun'.split())

print(calls_overweek)

plt.figure(figsize=(20,6))
axes=sns.heatmap(calls_overweek,cmap='Greens',linewidth=3)
calls_traffic= calls[calls['reason']=='Traffic']
calls_traffic_overweek = calls_traffic.groupby(['Day of Week', 'Hour']).count()['twp'].unstack(level=1)
calls_traffic_overweek=calls_traffic_overweek.reindex('Mon Tue Wed Thu Fri Sat Sun'.split())


plt.figure(figsize=(20,6))
axes=sns.heatmap(calls_traffic_overweek,cmap='Greens',linewidth=3)
# Month - day
calls_traffic_weekmonth = calls_traffic.groupby(['Day of Week', 'Month']).count()['twp'].unstack(level=1)
calls_traffic_weekmonth=calls_traffic_weekmonth.reindex('Mon Tue Wed Thu Fri Sat Sun'.split())

plt.figure(figsize=(20,6))
axes=sns.heatmap(calls_traffic_weekmonth,cmap='Greens',linewidth=3)


