## IMPORTS
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('whitegrid')
sns.set_context('notebook')
sns.set(font='times new roman',font_scale=1,palette='Greens')
import warnings
warnings.filterwarnings('ignore')

## READ ORIGINAL
spendings_boxcox=pd.read_pickle('spendings_boxcox')
Products=spendings_boxcox.columns
spendings_boxcox.head()

## TUKEY
## First I calculate the first and third quartile
Q1 = np.percentile(spendings_boxcox['Milk'],25)
Q3 = np.percentile(spendings_boxcox['Milk'],75)

spendings_boxcox['Outlier_milk_IQR']=0 # For plotting

## We consider it an outlier if the step is 1.5 Q1-Q3:
change = (Q3-Q1) * 1.5
   
outlier=spendings_boxcox['Milk'][(spendings_boxcox['Milk'] <= Q1 - change) | (spendings_boxcox['Milk'] >= Q3 + change)]
print('Outliers at indices:', outlier.index.values)
spendings_boxcox['Outlier_milk_IQR'].iloc[outlier.index]=1
print('Verification using value counts: ')
spendings_boxcox['Outlier_milk_IQR'].value_counts()

plt.figure()
sns.scatterplot(x=spendings_boxcox.index,y='Milk',data=spendings_boxcox,hue='Outlier_milk_IQR',style='Outlier_milk_IQR',c='green',palette='RdYlGn',alpha=1)


## Z SCORE
spendings_boxcox['Outlier_milk_Z']=0 # For plotting

z_values=(spendings_boxcox['Milk']-spendings_boxcox['Milk'].mean())/spendings_boxcox['Milk'].std()
outliers=spendings_boxcox['Milk'][abs(z_values)>3]

print('Outliers at indices:', outlier.index.values)
spendings_boxcox['Outlier_milk_Z'].iloc[outlier.index]=1
print('Verification using value counts: ')
spendings_boxcox['Outlier_milk_Z'].value_counts()
plt.figure()
sns.scatterplot(x=spendings_boxcox.index,y='Milk',data=spendings_boxcox,hue='Outlier_milk_Z',style='Outlier_milk_Z',c='green',palette='RdYlGn',alpha=1)

## ISO FOREST
from sklearn.ensemble import IsolationForest

spendings_boxcox['Outlier_milk_IsoFor']=1

IsoFor=IsolationForest(max_samples=440, random_state=101,contamination=.01)
IsoFor.fit(spendings_boxcox['Milk'].values.reshape(-1,1))
IsoFor_outlier = IsoFor.predict(spendings_boxcox['Milk'].values.reshape(-1,1))
spendings_boxcox['Outlier_milk_IsoFor']=spendings_boxcox['Outlier_milk_IsoFor']*IsoFor_outlier<1
plt.figure()
sns.scatterplot(x=spendings_boxcox.index,y='Milk',data=spendings_boxcox,hue='Outlier_milk_IsoFor',style='Outlier_milk_IsoFor',c='green',palette='RdYlGn',alpha=1)


## ELLIPTIC ENVOLOPE
from sklearn.covariance import EllipticEnvelope


spendings_boxcox['Outlier_milk_EE']=1

EE=EllipticEnvelope(random_state=101,contamination=.01)
EE.fit(spendings_boxcox['Milk'].values.reshape(-1,1))
EE_outlier = EE.predict(spendings_boxcox['Milk'].values.reshape(-1,1))
EE_outlier
spendings_boxcox['Outlier_milk_EE']=spendings_boxcox['Outlier_milk_EE']*IsoFor_outlier<1
plt.figure()
sns.scatterplot(x=spendings_boxcox.index,y='Milk',data=spendings_boxcox,hue='Outlier_milk_EE',style='Outlier_milk_EE',c='green',palette='RdYlGn',alpha=1)


## MODEL COMPARISON
models='IQR Z IsoFor EE'.split()
fig,axes=plt.subplots(nrows=2,ncols=2,figsize=(15,10))

row=0
col=0
for Model in models:
    sns.scatterplot(x=spendings_boxcox.index,y='Milk',data=spendings_boxcox,hue='Outlier_milk_'+Model,style='Outlier_milk_'+Model,c='green',palette='RdYlGn',alpha=1,ax=axes[row][col])

    axes[row][col].set_title(Model)
    if col==1:
        col-=2
        row+=1
    else:
        col+=1
        
        
## FULL OUTLIER DETECTION
outliers=[]
for Product in Products:
    Q1 = np.percentile(spendings_boxcox[Product],25)
    Q3 = np.percentile(spendings_boxcox[Product],75)

    ## We consider it an outlier if the step is 1.5 Q1-Q3:
    change = (Q3-Q1) * 1.5

    outlier=spendings_boxcox[Product][(spendings_boxcox[Product] <= Q1 - change) | (spendings_boxcox[Product] >= Q3 + change)]
    for i in range(0,len(outlier)): ## Append the selected outliers for each feature to athe total list
        outliers.append(outlier.index[i])
outliers=np.array(outliers)
outliers=np.unique(outliers) # select all uniques
print('Outlier indices: ',outliers)
print('# of outliers: ',len(outliers))
spendings_boxcox_IQR=pd.read_pickle('spendings_boxcox')
spendings_boxcox_IQR.drop(outliers)
print('So we should drop 22 of the 440 this if verified by the length of the remaining: ',len(spendings_boxcox_IQR))
#spendings_boxcox_IQR.to_pickle('spendings_boxcox_IQR')

outliers=[]
IsoFor=IsolationForest(max_samples=440, random_state=101,contamination=.01)
for Product in Products:
    IsoFor.fit(spendings_boxcox[Product].values.reshape(-1,1))
    IsoFor_outlier = IsoFor.predict(spendings_boxcox[Product].values.reshape(-1,1))
    outlier=np.where(IsoFor_outlier<0)

    for i in range(0,len(outlier[0])):
        outliers.append(outlier[0][i])
        
outliers=np.array(outliers)
outliers=np.unique(outliers)
print('Outlier indices: ',outliers)
print('# of outliers: ',len(outliers))

spendings_boxcox_IsoFor=pd.read_pickle('spendings_boxcox')
spendings_boxcox_IsoFor.drop(outliers)
#spendings_boxcox_IsoFor.to_pickle('spendings_boxcox_IsoFor')
