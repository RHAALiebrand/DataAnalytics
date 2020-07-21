## IMPORTS
# Analysis
import pandas as pd
import numpy as np

# Visualisation
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('whitegrid')
sns.set_context('notebook')
sns.set(font='times new roman',font_scale=1,palette='Greens')


## INITIAL DATA EXPL

spendings = pd.read_csv("Wholesale customers data.csv")
spendings.head()
spendings['Region'].value_counts()

channels = spendings['Channel'].apply(lambda x: x-1)
spendings.drop(['Region', 'Channel'], axis = 1, inplace = True)
print(channels)

print(spendings.info())

## KDES
fig,axes=plt.subplots(nrows=2,ncols=3,figsize=(15,10))
row=0
col=0
Products = spendings.columns
for Product in Products:
    sns.distplot(spendings[Product],color='green',bins=100,ax=axes[row][col])
    axes[row][col].set_title(Product)
    axes[row][col].set_xlim([0, 100000])
    axes[row][col].set_ylim([0,0.0005])
    

    if col==2:
        col-=3
        row+=1
    else:
        col+=1
        

## DATA SCALING
from scipy.stats import boxcox

spendings_log = spendings.apply(lambda x: np.log(x))
spendings_boxcox = spendings.apply(lambda x: boxcox(x))
type(spendings_boxcox)

fig,axes=plt.subplots(nrows=2,ncols=3,figsize=(15,10))
row=0
col=0
Products = spendings.columns
for Product in Products:
    sns.distplot(spendings_log[Product],color='green',bins=100,ax=axes[row][col])
    axes[row][col].set_title(Product)
    #axes[row][col].set_xlim([0, 100000])
    #axes[row][col].set_ylim([0,0.0005])
    

    if col==2:
        col-=3
        row+=1
    else:
        col+=1

fig,axes=plt.subplots(nrows=2,ncols=3,figsize=(15,10))
row=0
col=0
for Product in Products:
    sns.distplot(spendings_boxcox[Product][0],color='green',bins=100,ax=axes[row][col])
    axes[row][col].set_title(Product)
    #axes[row][col].set_xlim([0, 100000])
    #axes[row][col].set_ylim([0,0.0005])
    

    if col==2:
        col-=3
        row+=1
    else:
        col+=1
        
spendings['Total']=spendings.sum(axis=1)
spendings['Total'].describe()

plt.figure()
sns.distplot(spendings['Total'],color='green',bins=100)
spendings

## GUESS OF CUSTOMER SEGMENTS
spendings_relative=pd.DataFrame()
#Products=['Fresh']
for Product in Products:
    spendings_relative[Product]=round(spendings[Product]/spendings['Total']*100)
spendings_relative.describe()


print('Table 1')
print(spendings_relative.sort_values('Fresh',ascending=False).iloc[10:20])

print('Table 2')
print(spendings_relative.sort_values('Grocery',ascending=False).iloc[10:20])

print('Table 3')
print(spendings_relative.sort_values('Milk',ascending=False).iloc[10:20])
spendings_corr = spendings.drop('Total',axis=1).corr()
sns.heatmap(spendings_corr,annot=True,linewidths=2,cmap='Greens')


## FEATURE RELEVANCE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

LR_score=[]
DT_score=[]
for Product in Products:
    spendings_drop = spendings.drop([Product,'Total'],axis=1)
    X_train, X_test, y_train, y_test = train_test_split(spendings_drop,spendings[Product],test_size=0.25,random_state=101)
    LinReg = LinearRegression()
    LinReg.fit(X_train,y_train)

    DecTree=DecisionTreeRegressor()
    DecTree.fit(X_train,y_train)

    LR_score.append(round(LinReg.score(X_test,y_test),2))
    DT_score.append(round(DecTree.score(X_test,y_test),2))

feature_pred = pd.DataFrame(np.vstack((LR_score,DT_score)),index=['LinReg', 'DecTree'],columns=Products)
feature_pred

spendings_boxcox=pd.DataFrame(np.vstack((spendings_boxcox['Fresh'][0],spendings_boxcox['Milk'][0],spendings_boxcox['Grocery'][0],spendings_boxcox['Frozen'][0],spendings_boxcox['Detergents_Paper'][0],spendings_boxcox['Delicassen'][0])).transpose(),columns=Products)

plt.figure()
sns.pairplot(spendings_boxcox,plot_kws=dict(s=30, edgecolor="g", linewidth=2),diag_kws=dict(edgecolor='g',linewidth=2))

## OUTLEIR DETECTION
#spendings_boxcox.to_pickle('DataFrames/spendings_boxcox') ## This is the input of the outlier program
#spendings_log.to_pickle('DataFrames/spendings_log') ## This is the input of the outlier program
#channels.to_pickle('DataFrames/channels')


spendings_boxcox_IQR=pd.read_pickle('DataFrames/spendings_boxcox_IQR')
spendings_boxcox_IsoFor=pd.read_pickle('DataFrames/spendings_boxcox_IsoFor')

spendings_log_IQR=pd.read_pickle('DataFrames/spendings_log_IQR')
spendings_log_IsoFor=pd.read_pickle('DataFrames/spendings_log_IsoFor')

size=10
axs=sns.jointplot('Milk','Grocery',data=spendings_log,color='green',label='Original',alpha=1,s=size)
axs.ax_joint.scatter('Milk','Grocery',data=spendings_log_IQR,color='red',label='IQR',alpha=1,s=size)
axs.ax_joint.scatter('Milk','Grocery',data=spendings_log_IsoFor,color='orange',label='IsoFor',alpha=1,s=size)
axs.ax_joint.legend()
plt.savefig('Figures/Outlier_log_1.png',quality=50,format='png')

axs=sns.jointplot('Frozen','Delicassen',data=spendings_log,color='green',label='Original',alpha=1,s=size)
axs.ax_joint.scatter('Frozen','Delicassen',data=spendings_log_IQR,color='red',label='IQR',alpha=1,s=size)
axs.ax_joint.scatter('Frozen','Delicassen',data=spendings_log_IsoFor,color='orange',label='IsoFor',alpha=1,s=size)
axs.ax_joint.legend()
plt.savefig('Figures/Outlier_log_2.png',quality=50,format='png')

print('# Customers original = ',len(spendings_log))
print('# Customers IQR log= ',len(spendings_log_IQR))
print('# Customers IsoFor log= ',len(spendings_log_IsoFor))

size=10
axs=sns.jointplot('Milk','Grocery',data=spendings_boxcox,color='green',label='Original',alpha=1,s=size)
axs.ax_joint.scatter('Milk','Grocery',data=spendings_boxcox_IQR,color='red',label='IQR',alpha=1,s=size)
axs.ax_joint.scatter('Milk','Grocery',data=spendings_boxcox_IsoFor,color='orange',label='IsoFor',alpha=1,s=size)
axs.ax_joint.legend()
plt.savefig('Figures/Outlier_boxcox_1.png',quality=50,format='png')

axs=sns.jointplot('Frozen','Delicassen',data=spendings_boxcox,color='green',label='Original',alpha=1,s=size)
axs.ax_joint.scatter('Frozen','Delicassen',data=spendings_boxcox_IQR,color='red',label='IQR',alpha=1,s=size)
axs.ax_joint.scatter('Frozen','Delicassen',data=spendings_boxcox_IsoFor,color='orange',label='IsoFor',alpha=1,s=size)
axs.ax_joint.legend()
plt.savefig('Figures/Outlier_boxcox_2.png',quality=50,format='png')
print('# Customers original = ',len(spendings_boxcox))
print('# Customers IQR = ',len(spendings_boxcox_IQR))
print('# Customers IsoFor = ',len(spendings_boxcox_IsoFor))

plt.figure()
sns.pairplot(spendings_boxcox_IQR,plot_kws=dict(s=30, edgecolor="g", linewidth=2),diag_kws=dict(edgecolor='g',linewidth=2))

## PRINCINPLE COMPONENT ANALYSIS
from IPython.display import Image
Image(filename='Figures/PCA_dist_log.png') 

Image(filename='Figures/PCA_boxcox.png') 

spendings_log_IQR_red2=pd.read_pickle('DataFrames/spendings_log_IQR_red2')
spendings_log_IQR_red3=pd.read_pickle('DataFrames/spendings_log_IQR_red3')
spendings_boxcox_IQR_red2=pd.read_pickle('DataFrames/spendings_boxcox_IQR_red2')


## CLUSTERING

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

N_clus=np.arange(2,10)
N_clus

KM_scores=[]
GM_scores=[]

for N in N_clus:
    KMmod = KMeans(n_clusters=N)
    GMmod = GaussianMixture(n_components=N)
    
    KMmod.fit(spendings_log_IQR_red2)
    GMmod.fit(spendings_log_IQR_red2)
    
    KM_log_red2_pred = KMmod.predict(spendings_log_IQR_red2)
    GM_log_red2_pred = GMmod.predict(spendings_log_IQR_red2)

    KM_scores.append(silhouette_score(spendings_log_IQR_red2,KM_log_red2_pred))
    GM_scores.append(silhouette_score(spendings_log_IQR_red2,GM_log_red2_pred))

N_clus_DF = pd.DataFrame(np.vstack((N_clus,KM_scores,GM_scores)).transpose(),columns=['N_clus','KMeans','GM'])
print(N_clus_DF)

KM_scores=[]
GM_scores=[]
### Boxcox data - 2 dimensions

for N in N_clus:
    KMmod = KMeans(n_clusters=N)
    GMmod = GaussianMixture(n_components=N)
    
    KMmod.fit(spendings_boxcox_IQR_red2)
    GMmod.fit(spendings_boxcox_IQR_red2)
    
    KM_boxcox_red2_pred = KMmod.predict(spendings_boxcox_IQR_red2)
    GM_boxcox_red2_pred = GMmod.predict(spendings_boxcox_IQR_red2)

    KM_scores.append(silhouette_score(spendings_boxcox_IQR_red2,KM_boxcox_red2_pred))
    GM_scores.append(silhouette_score(spendings_boxcox_IQR_red2,GM_boxcox_red2_pred))
N_clus_DF = pd.DataFrame(np.vstack((N_clus,KM_scores,GM_scores)).transpose(),columns=['N_clus','KMeans','GM'])

### Log data - 2 versus 3 dimenions
GM_2_scores=[]
GM_3_scores=[]

for N in N_clus:
    GMmod_2 = GaussianMixture(n_components=N)
    GMmod_3 = GaussianMixture(n_components=N)
    
    GMmod_2.fit(spendings_log_IQR_red2)
    GMmod_3.fit(spendings_log_IQR_red3)
    
    GM_log_red2_pred = GMmod_2.predict(spendings_log_IQR_red2)
    GM_log_red3_pred = GMmod_3.predict(spendings_log_IQR_red3)

    GM_2_scores.append(silhouette_score(spendings_log_IQR_red2,GM_log_red2_pred))
    GM_3_scores.append(silhouette_score(spendings_log_IQR_red3,GM_log_red3_pred))
    
N_clus_DF = pd.DataFrame(np.vstack((N_clus,GM_2_scores,GM_3_scores)).transpose(),columns=['N_clus','70% Expl Var','90% Expl Var'])
print(N_clus_DF)


N=2

GMmod_2 = GaussianMixture(n_components=N)
GMmod_3 = GaussianMixture(n_components=N)

GMmod_2.fit(spendings_log_IQR_red2)
GMmod_3.fit(spendings_log_IQR_red3)

GM_log_red2_pred = GMmod_2.predict(spendings_log_IQR_red2)
GM_log_red3_pred = GMmod_3.predict(spendings_log_IQR_red3) ## With 3 min cluster 0 = cluster 1, why? No idea.
where_0 = np.where(GM_log_red3_pred == 0)
where_1 = np.where(GM_log_red3_pred == 1)

GM_log_red3_pred[where_0] = 1
GM_log_red3_pred[where_1] = 0


channels_log = pd.read_pickle('DataFrames/channels_log_IQR').reset_index()
channels_log.drop('index',axis=1,inplace=True)

# For 2 dim
GM_log_red2_pred = pd.DataFrame(GM_log_red2_pred, columns = ['Cluster'])
GM_log_2 = pd.concat([GM_log_red2_pred,channels_log,spendings_log_IQR_red2], axis = 1)

# For 3 dim
GM_log_red3_pred = pd.DataFrame(GM_log_red3_pred, columns = ['Cluster'])
GM_log_3 = pd.concat([GM_log_red3_pred,channels_log,spendings_log_IQR_red3], axis = 1)


import math

fig, ax = plt.subplots(figsize = (14,8))
# Looping over points makes the plots more controlable at this moment
for i in range(0,len(GM_log_2)):
    if GM_log_2['Cluster'].iloc[i]==GM_log_2['Channel'].iloc[i]:
        if GM_log_2['Cluster'].iloc[i]==0:
            ax.scatter(x=GM_log_2['Dim1'].iloc[i],y=GM_log_2['Dim2'].iloc[i],c='green',s=13,alpha=0.8)
        else:
            ax.scatter(x=GM_log_2['Dim1'].iloc[i],y=GM_log_2['Dim2'].iloc[i],c='black',s=13,alpha=0.8)
    else:
         ax.scatter(x=GM_log_2['Dim1'].iloc[i],y=GM_log_2['Dim2'].iloc[i],c='red',s=13,alpha=0.8)
  

GMmod_2_centers = GMmod_2.means_
# Plot centers with indicators
for i, c in enumerate(GMmod_2_centers):
    ax.scatter(x = c[0], y = c[1], color = 'white', edgecolors = 'black',alpha = 1, linewidth = 2, marker = 'o', s=200);
    ax.scatter(x = c[0], y = c[1], marker='$%d$'%(i), alpha = 1, s=80,c='black');
    
# Finsihing touch    
ax.scatter(x=math.nan,y=math.nan,c='green',s=13,label='Cluster 0',alpha=0.8)
ax.scatter(x=math.nan,y=math.nan,c='black',s=13,label='Cluster 1',alpha=0.8)
ax.scatter(x=math.nan,y=math.nan,c='red',s=13,label='Wrongly predicted',alpha=0.8)
ax.scatter(x = math.nan, y = math.nan, color = 'white', edgecolors = 'black',alpha = 1, linewidth = 2, marker = 'o', s=100,label='Cluster centre')

       
ax.set_xlabel("Dim1", fontsize=14)
ax.set_ylabel("Dim2", fontsize=14)
ax.set_title("PC plane with GM and original clusters (2-Dim, 70% explained var)", fontsize=16);
ax.legend(fontsize=14)

from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d import Axes3D


fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(111, projection='3d')

for i in range(0,len(GM_log_3)):
    if GM_log_3['Cluster'].iloc[i]==GM_log_3['Channel'].iloc[i]:
        if GM_log_3['Cluster'].iloc[i]==0:
            x=GM_log_3['Dim1'].iloc[i]
            y=GM_log_3['Dim2'].iloc[i]
            z=GM_log_3['Dim3'].iloc[i]
            ax.scatter(x, y, z, c='green', marker='o',s=13,alpha=0.8)     
        else:
            x=GM_log_3['Dim1'].iloc[i]
            y=GM_log_3['Dim2'].iloc[i]
            z=GM_log_3['Dim3'].iloc[i]
            ax.scatter(x, y, z, c='black', marker='o',s=13,alpha=0.8)    
    else:
        x=GM_log_3['Dim1'].iloc[i]
        y=GM_log_3['Dim2'].iloc[i]
        z=GM_log_3['Dim3'].iloc[i]
        ax.scatter(x, y, z, c='red', marker='o',s=13,alpha=0.8)
         #ax.scatter(x=GM_log_3['Dim1'].iloc[i],,,c='red',s=10)
  
GMmod_3_centers = GMmod_3.means_
# Plot centers with indicators
for i, c in enumerate(GMmod_3_centers):
    x=c[0]
    y=c[1]
    z=c[2]
    ax.scatter(x,y,z, color = 'white', edgecolors = 'black',alpha = 1, linewidth = 2, marker = 'o', s=200);
    if i==1:
        ax.scatter(x,y,z, marker='$%d$'%(0), alpha = 1, s=80,c='black');
    else:
        ax.scatter(x,y,z, marker='$%d$'%(1), alpha = 1, s=80,c='black');


x=math.nan
y=math.nan
z=math.nan
ax.scatter(x,y,z,c='green',s=20,label='Cluster 0',alpha=0.8)
ax.scatter(x,y,z,c='black',s=20,label='Cluster 1',alpha=0.8)
ax.scatter(x,y,z,c='red',s=20,label='Wrongly predicted',alpha=0.8)
ax.scatter(x,y,z, color = 'white', edgecolors = 'black',alpha = 1, linewidth = 2, marker = 'o', s=200,label='Cluster centre')

       
ax.set_xlabel("Dim1", fontsize=14)
ax.set_ylabel("Dim2", fontsize=14)
ax.set_ylabel("Dim3", fontsize=14)
ax.set_title("PC plane with GM and original clusters (3-Dim, 90% explained var)", fontsize=16);
lgnd=ax.legend(fontsize=14)

#change the marker size manually for both lines
lgnd.legendHandles[0]._sizes=[30]
lgnd.legendHandles[1]._sizes=[30]
lgnd.legendHandles[2]._sizes=[30]
lgnd.legendHandles[3]._sizes=[50]

# plt.xlim([-3, 3])
# plt.ylim([-3, 3])
# plt.zlim([-3, 3])
plt.show()

from sklearn.metrics import classification_report,confusion_matrix,f1_score

print(classification_report(GM_log_2['Channel'].values,GM_log_2['Cluster'].values))
print(classification_report(GM_log_3['Channel'].values,GM_log_3['Cluster'].values))

GM_log_2_f1=f1_score(GM_log_2['Channel'].values,GM_log_2['Cluster'].values)
GM_log_3_f1=f1_score(GM_log_3['Channel'].values,GM_log_3['Cluster'].values)


### Log data - KMeans versus Gaussian
N=2

KMmod_2 = KMeans(n_clusters=N)
KMmod_3 = KMeans(n_clusters=N)

KMmod_2.fit(spendings_log_IQR_red2)
KMmod_3.fit(spendings_log_IQR_red3)

KM_log_red2_pred = KMmod_2.predict(spendings_log_IQR_red2)
KM_log_red3_pred = KMmod_3.predict(spendings_log_IQR_red3)

KM_log_red2_pred = pd.DataFrame(KM_log_red2_pred, columns = ['Cluster'])
KM_log_2 = pd.concat([KM_log_red2_pred,channels_log,spendings_log_IQR_red2], axis = 1)

KM_log_red3_pred = pd.DataFrame(KM_log_red3_pred, columns = ['Cluster'])
KM_log_3 = pd.concat([KM_log_red3_pred,channels_log,spendings_log_IQR_red3], axis = 1)

import math

fig, ax = plt.subplots(figsize = (14,8))

for i in range(0,len(KM_log_2)):
    if KM_log_2['Cluster'].iloc[i]==KM_log_2['Channel'].iloc[i]:
        if KM_log_2['Cluster'].iloc[i]==0:
            ax.scatter(x=KM_log_2['Dim1'].iloc[i],y=KM_log_2['Dim2'].iloc[i],c='green',s=13,alpha=0.8)
        else:
            ax.scatter(x=KM_log_2['Dim1'].iloc[i],y=KM_log_2['Dim2'].iloc[i],c='black',s=13,alpha=0.8)
    else:
         ax.scatter(x=KM_log_2['Dim1'].iloc[i],y=KM_log_2['Dim2'].iloc[i],c='red',s=13,alpha=0.8)
  

KMmod_2_centers = KMmod_2.cluster_centers_
# Plot centers with indicators
for i, c in enumerate(KMmod_2_centers):
    ax.scatter(x = c[0], y = c[1], color = 'white', edgecolors = 'black',alpha = 1, linewidth = 2, marker = 'o', s=200);
    ax.scatter(x = c[0], y = c[1], marker='$%d$'%(i), alpha = 1, s=80,c='black');
    
    
ax.scatter(x=math.nan,y=math.nan,c='green',s=13,label='Cluster 0',alpha=0.8)
ax.scatter(x=math.nan,y=math.nan,c='black',s=13,label='Cluster 1',alpha=0.8)
ax.scatter(x=math.nan,y=math.nan,c='red',s=13,label='Wrongly predicted',alpha=0.8)
ax.scatter(x = math.nan, y = math.nan, color = 'white', edgecolors = 'black',alpha = 1, linewidth = 2, marker = 'o', s=100,label='Cluster centre')

       
ax.set_xlabel("Dim1", fontsize=14)
ax.set_ylabel("Dim2", fontsize=14)
ax.set_title("PC plane with KM and original clusters (2-Dim, 70% explained var)", fontsize=16);
ax.legend(fontsize=14)

from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d import Axes3D


fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(111, projection='3d')

for i in range(0,len(GM_log_2)):
    if GM_log_3['Cluster'].iloc[i]==GM_log_3['Channel'].iloc[i]:
        if GM_log_3['Cluster'].iloc[i]==0:
            x=GM_log_3['Dim1'].iloc[i]
            y=GM_log_3['Dim2'].iloc[i]
            z=GM_log_3['Dim3'].iloc[i]
            ax.scatter(x, y, z, c='green', marker='o',s=13,alpha=0.8)     
        else:
            x=GM_log_3['Dim1'].iloc[i]
            y=GM_log_3['Dim2'].iloc[i]
            z=GM_log_3['Dim3'].iloc[i]
            ax.scatter(x, y, z, c='black', marker='o',s=13,alpha=0.8)    
    else:
        x=GM_log_3['Dim1'].iloc[i]
        y=GM_log_3['Dim2'].iloc[i]
        z=GM_log_3['Dim3'].iloc[i]
        ax.scatter(x, y, z, c='red', marker='o',s=13,alpha=0.8)
         #ax.scatter(x=GM_log_3['Dim1'].iloc[i],,,c='red',s=10)
  
GMmod_3_centers = GMmod_3.means_
# Plot centers with indicators
for i, c in enumerate(GMmod_3_centers):
    x=c[0]
    y=c[1]
    z=c[2]
    ax.scatter(x,y,z, color = 'white', edgecolors = 'black',alpha = 1, linewidth = 2, marker = 'o', s=200);
    if i==1:
        ax.scatter(x,y,z, marker='$%d$'%(0), alpha = 1, s=80,c='black');
    else:
        ax.scatter(x,y,z, marker='$%d$'%(1), alpha = 1, s=80,c='black');

    

x=math.nan
y=math.nan
z=math.nan
ax.scatter(x,y,z,c='green',s=20,label='Cluster 0',alpha=0.8)
ax.scatter(x,y,z,c='black',s=20,label='Cluster 1',alpha=0.8)
ax.scatter(x,y,z,c='red',s=20,label='Wrongly predicted',alpha=0.8)
ax.scatter(x,y,z, color = 'white', edgecolors = 'black',alpha = 1, linewidth = 2, marker = 'o', s=200,label='Cluster centre')

       
ax.set_xlabel("Dim1", fontsize=14)
ax.set_ylabel("Dim2", fontsize=14)
ax.set_ylabel("Dim3", fontsize=14)
ax.set_title("PC plane with GM and original clusters (3-Dim, 90% explained var)", fontsize=16);
lgnd=ax.legend(fontsize=14)

#change the marker size manually for both lines
lgnd.legendHandles[0]._sizes=[30]
lgnd.legendHandles[1]._sizes=[30]
lgnd.legendHandles[2]._sizes=[30]
lgnd.legendHandles[3]._sizes=[50]

# plt.xlim([-3, 3])
# plt.ylim([-3, 3])
# plt.zlim([-3, 3])
plt.show()


from sklearn.metrics import classification_report,confusion_matrix,f1_score

print(classification_report(KM_log_2['Channel'].values,KM_log_2['Cluster'].values))
print(classification_report(KM_log_3['Channel'].values,KM_log_3['Cluster'].values))

KM_log_2_f1=f1_score(KM_log_2['Channel'].values,KM_log_2['Cluster'].values)
KM_log_3_f1=f1_score(KM_log_3['Channel'].values,KM_log_3['Cluster'].values)


F1s = pd.DataFrame(np.array([GM_log_2_f1,GM_log_3_f1,KM_log_2_f1,KM_log_3_f1]).reshape(1,-1),columns=['GM 70%' ,'GM 90%','KM 70%' ,'KM 90%'])
print(F1s)


        
