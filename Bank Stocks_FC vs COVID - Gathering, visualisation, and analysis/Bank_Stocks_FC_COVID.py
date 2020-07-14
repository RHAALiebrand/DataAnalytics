## IMPORTS
# Analysis
import pandas as pd
import numpy as np
import datetime

# Visualisation
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('whitegrid')
sns.set_context('notebook')
sns.set(font='times new roman',font_scale=1,palette='Greens')
plt.close('all')

# Cut warnings for now
import warnings
warnings.filterwarnings('ignore')

## READ AND PLOT DOW JONES
DowJ = pd.read_csv('DOW_JONES.csv',index_col=0)
DowJ['Price_num']=DowJ['Price'].apply(lambda x: float(x.split(',')[0]+x.split(',')[1]))
DowJ.index=pd.to_datetime(DowJ.index)
plt.figure()
DowJ['Price_num'].plot.line(figsize=(12,4),c='green',style="-")
plt.title('Dow Jones total value')

fig,axes=plt.subplots(nrows=1,ncols=2)
dates = (DowJ.index > datetime.datetime(2007,6,1)) & (DowJ.index <= datetime.datetime(2013,1,1))
DowJ.loc[dates].plot.line(figsize=(12,4),c='green',style="-",ax=axes[0])
axes[0].set_title('FC')
dates = (DowJ.index > datetime.datetime(2020,2,1)) & (DowJ.index <= datetime.datetime(2020,7,1))
DowJ.loc[dates].plot.line(figsize=(12,4),c='green',style="-",ax=axes[1])
axes[1].set_title('COVID')

## SET DATES BASED ON FIGURES
date_start_FC=datetime.datetime(2008,1,1)
date_end_FC=datetime.datetime(2013,1,1)

date_start_COVID=datetime.datetime(2020,2,1)
date_end_COVID=datetime.datetime(2020,7,1) # I update this everytime I check this notebook

## LET'S DISCOVER THE DATA
stocks_FC=pd.read_pickle('stocks_FC.csv')

plt.figure()
ax=sns.heatmap(stocks_FC.isna(),cmap='Greens')
ax.set_title('I am happy - no empties')

print(stocks_FC.xs('close', level='Stock', axis=1).describe())

stocks_FC_close = stocks_FC.xs('close', level='Stock', axis=1)

plt.figure(figsize=(12,5))
ax1=sns.boxplot(data=stocks_FC_close[['EU','US']],width=0.3)
ax1.set_title('Boxplot - Absolute stock price for the six banks during the FC')

## NOW CHANGES
stocks_FC_close_change = stocks_FC_close.pct_change()*100

## I have observed a very high value for Citigroup (Order 800-900%) why??
stocks_FC_close_change.index=pd.to_datetime(stocks_FC_close_change.index)
dates = (stocks_FC_close_change.index > datetime.datetime(2011,5,2)) & (stocks_FC_close_change.index <= datetime.datetime(2011,5,13))
stocks_FC_close_change.loc[dates]

print(stocks_FC_close.loc[dates])

stocks_FC_close_change[stocks_FC_close_change>800]=0

plt.figure(figsize=(12,5))
ax1=sns.boxplot(data=stocks_FC_close_change.droplevel('Region', axis=1),width=0.3)
ax1.set_ylim([-60,60])
ax1.set_title('Boxplot - Percental stock change for the six banks during the FC')


## CREATE PAIRPLOT
pair_plot=sns.pairplot(stocks_FC_close_change.droplevel('Region', axis=1), plot_kws=dict(s=30, edgecolor="g", linewidth=2),diag_kws=dict(edgecolor='g',linewidth=2))

for i in range(0,5):
    for j in range(0,5):
        pair_plot.axes[i,j].set_ylim(-60,60)
        pair_plot.axes[i,j].set_xlim(-60,60)

## PRINT SOME INTERESTING POINTS
print(stocks_FC_close_change.min())

print(stocks_FC_close_change.idxmax())

print(stocks_FC_close_change.std())

dates_FCrec = (stocks_FC_close_change.index > datetime.datetime(2012,1,1)) & (stocks_FC_close_change.index <= datetime.datetime(2013,1,1))
stocks_FC_close_change.loc[dates_FCrec]
print(stocks_FC_close_change.loc[dates_FCrec].std())

## LOOP OVER THE BANKS AND PLOT
banks='ING DDB HSB CIT GMS JPM'.split()
fig,axes=plt.subplots(nrows=2,ncols=3,figsize=(15,10))
row=0
col=0
for bank in banks:
    sns.distplot(stocks_FC_close_change.loc[dates_FCrec].xs(bank,level='Bank',axis=1),color='green',bins=100,ax=axes[row][col])
    axes[row][col].set_title(bank)
    axes[row][col].set_xlim([-10, 10])
    axes[row][col].set_ylim([0, 0.4])
    

    if col==2:
        col-=3
        row+=1
    else:
        col+=1
        
 ## PLOT CORRELATIONS
fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(20,7))
sns.heatmap(stocks_FC_close_change.loc[dates_FCrec].corr(),annot=True,ax=axes[1],linewidths=2,cmap='Greens')
axes[1].set_title('Correlation of stock prices in 2012')
sns.heatmap(stocks_FC_close_change.corr(),annot=True,ax=axes[0],linewidths=2,cmap='Greens')
axes[0].set_title('Correlation of stock prices between 2006 and 2013',fontsize=15)

## TIME SERIES
# Optional Plotly Method Imports
import plotly
import cufflinks as cf
cf.go_offline()

stocks_FC_close.xs(bank,level='Bank',axis=1) # We want to plot this

ax=plt.figure()
ax=stocks_FC_close.droplevel('Region', axis=1).iplot()

dates_FCrec_2 = (stocks_FC_close_change.index > datetime.datetime(2009,3,13)) & (stocks_FC_close_change.index <= datetime.datetime(2009,10,13))

ax=plt.figure()
ax=stocks_FC_close[dates_FCrec_2].droplevel('Region', axis=1).iplot()

stocks_FC_close = stocks_FC_close.droplevel(level='Region',axis=1)
print(stocks_FC_close)


## LINEAR FITS
colors=['orange','blue','green','purple','red','cyan']
plt.figure(figsize=(20,10))
lin_coef_FC=[]  # This is for later reference

for i in range(0,len(banks)):
    price = (stocks_FC_close[dates_FCrec_2][banks[i]].values/stocks_FC_close[dates_FCrec_2][banks[i]].values[0]-1)*100
    days_from_start = np.arange(0,len(price))
    linFit = np.polyfit(days_from_start,price,1) # Some good old numpy fitting
    lin_coef_FC.append(linFit[0])
    
    plt.scatter(days_from_start,price,c=colors[i],s=1)
    plt.plot([days_from_start[0],days_from_start[-1]],[linFit[0]*days_from_start[0]+linFit[1],linFit[0]*days_from_start[-1]+linFit[1]],c=colors[i],label=banks[i])
    plt.legend(fontsize=20)
    plt.ylabel('Stock price increase since recovery started [%]', fontsize=20)
    plt.xlabel('Days of recovery', fontsize=20)
    
    
## FORECASTING
forecast_days = 2

stocks_FC_close_pred=stocks_FC_close[dates_FCrec_2].shift(-forecast_days)
print(stocks_FC_close_pred)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

## INITIALISE LISTS OF R2 OF THE DIFFERENT MODELS
linReg_R2=[]
SVR_R2=[]
SVR_GS_R2=[]


forecast_days = np.array([3, 10, 20])  # Amount of forecast days
R2_values=np.zeros((len(banks)+1,3*len(forecast_days))) # The matrix to be filled


for j in range(0,len(forecast_days)):
    stocks_FC_close_pred=stocks_FC_close[dates_FCrec_2].shift(-forecast_days[j])

    for i in range(0,len(banks)):
        ## SPLIT THE DATA
        X=np.array(stocks_FC_close[dates_FCrec_2][banks[i]])
        X=X[:-forecast_days[j]] # Here we shift it

        y=np.array(stocks_FC_close_pred[banks[i]])
        y=y[:-forecast_days[j]]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        ## LINEAR REGESSION
        linMod = LinearRegression()
        linMod.fit(X_train.reshape(-1, 1),y_train)
        linReg_R2.append(linMod.score(X_test.reshape(-1, 1),y_test))
        R2_values[i,0+3*j]=linMod.score(X_test.reshape(-1,1),y_test)

        ## SVR
        SVRmodel = SVR(kernel='rbf', C=1e3, gamma=0.1)
        SVRmodel.fit(X_train.reshape(-1, 1),y_train)
        SVR_R2.append(SVRmodel.score(X_test.reshape(-1,1),y_test))
        R2_values[i,1+3*j]=SVRmodel.score(X_test.reshape(-1,1),y_test)
        
        ## SVR WITH GRID SEARCH
        param_grid={'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.01,0.001,0.0001]}

        GridSVR = GridSearchCV(SVR(), param_grid,verbose=0) ## One can change verbose to 3 if you need info over the GS
        GridSVR.fit(X_train.reshape(-1,1),y_train)
        SVR_GS_R2.append(GridSVR.score(X_test.reshape(-1,1),y_test))
        R2_values[i,2+3*j]=GridSVR.score(X_test.reshape(-1,1),y_test)
        
        

    R2_values[i+1,0+3*j]=np.mean(linReg_R2)
    R2_values[i+1,1+3*j]=np.mean(SVR_R2)
    R2_values[i+1,2+3*j]=np.mean(SVR_GS_R2)

## PRODUCE DATAFRAME
col_names = [np.array(['3', '3', '3', '10', '10', '10', '20', '20' , '20']),np.array(['linReg', 'SVR', 'SVR_GS', 'linReg', 'SVR', 'SVR_GS','linReg', 'SVR', 'SVR_GS'])]
R2_DF_FC = pd.DataFrame(R2_values,columns=col_names,index=['ING', 'DDB', 'HSB', 'CIT', 'GMS', 'JPM', 'Average'])
R2_DF_FC.columns.names=['n_predictionDays','PredictionMethod']
print(R2_DF_FC)

## NOW FOR THE COVID CRISIS
stocks_COVID=pd.read_pickle('stocks_COVID.csv')

ax=sns.heatmap(stocks_COVID.isna(),cmap='Greens')
ax.set_title('I am happy - no empties')

print(stocks_COVID.xs('close', level='Stock', axis=1).describe())
## BOX PLOTS
stocks_COVID_close = stocks_COVID.xs('close', level='Stock', axis=1)
plt.figure(figsize=(12,5))
ax1=sns.boxplot(data=stocks_COVID_close[['EU','US']],width=0.3)
ax1.set_title('Boxplot - Absolute stock price for the six banks during the COVID')

stocks_COVID_close_change = stocks_COVID_close.pct_change()*100

plt.figure(figsize=(12,5))
ax1=sns.boxplot(data=stocks_COVID_close_change.droplevel('Region', axis=1),width=0.3)
ax1=sns.boxplot(data=stocks_FC_close_change.droplevel('Region', axis=1),width=0.15,palette='Reds')

ax1.set_ylim([-20,20])
ax1.set_title('Boxplot - Percental stock change for the six banks during COVID')

## PAIRPLOTS
pair_plot=sns.pairplot(stocks_COVID_close_change.droplevel('Region', axis=1), plot_kws=dict(s=30, edgecolor="g", linewidth=2),diag_kws=dict(edgecolor='g',linewidth=2))


for i in range(0,5):
    for j in range(0,5):
        pair_plot.axes[i,j].set_ylim(-20,20)
        pair_plot.axes[i,j].set_xlim(-20,20)

## SOME PRINTS FOR NOTEBOOK
print(stocks_COVID_close_change.min())

print(stocks_COVID_close_change.idxmin())

print(stocks_COVID_close_change.std())

stocks_COVID_close_change.index=pd.to_datetime(stocks_COVID_close_change.index)

dates_COVIDrec = (stocks_COVID_close_change.index > datetime.datetime(2020,3,17)) & (stocks_COVID_close_change.index <= datetime.datetime(2020,7,1))
stocks_COVID_close_change.loc[dates_COVIDrec]
print(stocks_COVID_close_change.loc[dates_COVIDrec].std())

## NOW LETS PLOT FOR THE BANKS AGAIN WITH FC ON TOP
banks='ING DDB HSB CIT GMS JPM'.split()
fig,axes=plt.subplots(nrows=2,ncols=3,figsize=(15,10))
row=0
col=0
for bank in banks:
    sns.distplot(stocks_COVID_close_change.loc[dates_COVIDrec].xs(bank,level='Bank',axis=1),color='green',bins=100,ax=axes[row][col])
    sns.distplot(stocks_FC_close_change.loc[dates_FCrec].xs(bank,level='Bank',axis=1),color='red',bins=100,ax=axes[row][col])

    axes[row][col].set_title(bank)
    axes[row][col].set_xlim([-10, 10])
    axes[row][col].set_ylim([0, 0.4])
    

    if col==2:
        col-=3
        row+=1
    else:
        col+=1
sns.distplot(stocks_COVID_close_change.loc[dates_COVIDrec].xs('ING',level='Bank',axis=1),color='green',bins=100,ax=axes[0][0],label='COVID')
sns.distplot(stocks_FC_close_change.loc[dates_FCrec].xs('ING',level='Bank',axis=1),color='red',bins=100,ax=axes[0][0],label='FC')

axes[0][0].legend()

## CORRELATION PLOTS
fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(20,7))
sns.heatmap(stocks_COVID_close_change.loc[dates_COVIDrec].corr(),annot=True,ax=axes[1],linewidths=2,cmap='Greens')
axes[1].set_title('Correlation of stock prices recovery period of COVID')
sns.heatmap(stocks_COVID_close_change.corr(),annot=True,ax=axes[0],linewidths=2,cmap='Greens')
axes[0].set_title('Correlation of stock prices in 2020',fontsize=15)


ax=plt.figure()
ax=stocks_COVID_close.droplevel('Region', axis=1).iplot()

dates_COVIDrec_2 = (stocks_COVID_close_change.index > datetime.datetime(2020,3,23)) & (stocks_COVID_close_change.index <= datetime.datetime(2020,7,1))
stocks_COVID_close[dates_COVIDrec_2]
ax=plt.figure()
ax=stocks_COVID_close[dates_COVIDrec_2].droplevel('Region', axis=1).iplot()

## LINEAR FITS
stocks_COVID_close = stocks_COVID_close.droplevel(level='Region',axis=1)
colors=['orange','blue','green','purple','red','cyan']
plt.figure(figsize=(20,10))
lin_coef_COVID=[]

for i in range(0,len(banks)):
    price = (stocks_COVID_close[dates_COVIDrec_2][banks[i]].values/stocks_COVID_close[dates_COVIDrec_2][banks[i]].values[0]-1)*100
    days_from_start = np.arange(0,len(price))
    linFit = np.polyfit(days_from_start,price,1)
    lin_coef_COVID.append(linFit[0])
    
    plt.scatter(days_from_start,price,c=colors[i],s=1)
    plt.plot([days_from_start[0],days_from_start[-1]],[linFit[0]*days_from_start[0]+linFit[1],linFit[0]*days_from_start[-1]+linFit[1]],c=colors[i],label=banks[i])
    plt.legend(fontsize=20)
    plt.ylabel('Stock price increase since recovery started [%]', fontsize=20)
    plt.xlabel('Days of recovery', fontsize=20)
    
## COMPARE RECOVERY RATES
linCoeffs=pd.DataFrame(np.hstack((np.reshape(lin_coef_FC,(-1,1)), np.reshape(lin_coef_COVID,(-1,1)))),index=banks,columns=['FC','COVID'])
print(linCoeffs)

## PREDICT COVID
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

linReg_R2=[]
SVR_R2=[]
SVR_GS_R2=[]

forecast_days = np.array([3, 10, 20])
R2_values=np.zeros((len(banks)+1,3*len(forecast_days)))

for j in range(0,len(forecast_days)):
    stocks_COVID_close_pred=stocks_COVID_close[dates_COVIDrec_2].shift(-forecast_days[j])



    for i in range(0,len(banks)):
        X=np.array(stocks_COVID_close[dates_COVIDrec_2][banks[i]])
        X=X[:-forecast_days[j]]

        y=np.array(stocks_COVID_close_pred[banks[i]])
        y=y[:-forecast_days[j]]
        np.shape(y)


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        linMod = LinearRegression()
        linMod.fit(X_train.reshape(-1, 1),y_train)
        linReg_R2.append(linMod.score(X_test.reshape(-1, 1),y_test))
        R2_values[i,0+3*j]=linMod.score(X_test.reshape(-1,1),y_test)


        SVRmodel = SVR(kernel='rbf', C=1e3, gamma=0.1)
        SVRmodel.fit(X_train.reshape(-1, 1),y_train)
        SVR_R2.append(SVRmodel.score(X_test.reshape(-1,1),y_test))
        R2_values[i,1+3*j]=SVRmodel.score(X_test.reshape(-1,1),y_test)

        param_grid={'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.01,0.001,0.0001]}

        GridSVR = GridSearchCV(SVR(), param_grid,verbose=0)
        GridSVR.fit(X_train.reshape(-1,1),y_train)
        SVR_GS_R2.append(GridSVR.score(X_test.reshape(-1,1),y_test))
        R2_values[i,2+3*j]=GridSVR.score(X_test.reshape(-1,1),y_test)
        
        

    R2_values[i+1,0+3*j]=abs(np.mean(linReg_R2))
    R2_values[i+1,1+3*j]=abs(np.mean(SVR_R2))
    R2_values[i+1,2+3*j]=abs(np.mean(SVR_GS_R2))
    
np.shape(R2_values)

col_names = [np.array(['3', '3', '3', '10', '10', '10', '20', '20' , '20']),np.array(['linReg', 'SVR', 'SVR_GS', 'linReg', 'SVR', 'SVR_GS','linReg', 'SVR', 'SVR_GS'])]
R2_DF_COVID = pd.DataFrame(abs(R2_values),columns=col_names,index=['ING', 'DDB', 'HSB', 'CIT', 'GMS', 'JPM', 'Average'])
R2_DF_COVID.columns.names=['n_predictionDays','PredictionMethod']
print(R2_DF_COVID)

## COMPARISON
DF_R2=pd.concat([R2_DF_FC, R2_DF_COVID], keys=['FC', 'COVID'])
print(DF_R2)












