## IMPORTS
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sns.set_style('whitegrid')
sns.set_context('notebook')
sns.set(font='times new roman',font_scale=1,palette='Greens')
import warnings
warnings.filterwarnings('ignore')

spendings_boxcox_IQR=pd.read_pickle('DataFrames/spendings_boxcox_IQR')
spendings_log_IQR=pd.read_pickle('DataFrames/spendings_log_IQR')
## MODELS
from sklearn.decomposition import PCA

PCAmod_log = PCA()
PCAmod_log.fit(spendings_log_IQR)
PCAdata_log = PCAmod_log.transform(spendings_log_IQR)

PCAmod_boxcox = PCA()
PCAmod_boxcox.fit(spendings_boxcox_IQR)
PCAdata_boxcox = PCAmod_boxcox.transform(spendings_boxcox_IQR)

## LOG SCALED DATA PCA
Explained_var_log=PCAmod_log.explained_variance_ratio_
Explained_var_log.sum() ## This verifies

DFcomp_log=pd.DataFrame(np.hstack((PCAmod_log.components_,Explained_var_log.reshape(-1,1))),columns=['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen','Explained Var'])
DFcomp_log = DFcomp_log.rename(index={0:'Dim 1',1:'Dim 2',2:'Dim 3',3:'Dim 4',4:'Dim 5',5:'Dim 6'})
DFcomp_log

fig, ax = plt.subplots(figsize = (14,8))
DFcomp_log.drop('Explained Var',axis=1).plot(ax = ax, kind = 'bar');

for i in range(0,len(Explained_var_log)):
    ax.text(i-0.40, ax.get_ylim()[1] + 0.05, "Explained Variance\n          %.4f"%(Explained_var_log[i]))
    
plt.savefig('Figures/PCA_dist_log.png',quality=50,format='png')

## DIM REDUCTION 2
PCAmod_log_red2 = PCA(n_components=2)
PCAmod_log_red2.fit(spendings_log_IQR)
PCAdata_log_red2 = PCAmod_log_red2.transform(spendings_log_IQR)

print('Shape reduced data =',np.shape(PCAdata_log_red2))
print('This is perfect, since the dimension is now 2 iso 6 while the amount of data-point remained the same. But is it an array now, I want to have it as a DF')

spendings_log_IQR_red2 = pd.DataFrame(PCAdata_log_red2,columns='Dim1 Dim2'.split())
spendings_log_IQR_red2.to_pickle('DataFrames/spendings_log_IQR_red2')
print(spendings_log_IQR_red2.head())

fig, ax = plt.subplots(figsize = (14,8))

# scatterplot of the reduced data    
ax.scatter(x=spendings_log_IQR_red2.loc[:, 'Dim1'], y=spendings_log_IQR_red2.loc[:, 'Dim2'], 
facecolors='g', edgecolors='g', s=50, alpha=0.5)

feature_vectors = PCAmod_log_red2.components_.T

# we use scaling factors to make the arrows easier to see
arrow_size, text_pos = 5.0, 6.0,

# projections of the original features
for i in range(0,len(feature_vectors)):
    ax.arrow(0, 0, arrow_size*feature_vectors[i][0], arrow_size*feature_vectors[i][1], 
              head_width=0.2, head_length=0.2, linewidth=2, color='black')
    ax.text(feature_vectors[i][0]*text_pos, feature_vectors[i][1]*text_pos, spendings_log_IQR.columns[i], color='black', 
             ha='center', va='center', fontsize=18)

ax.set_xlabel("Dim1", fontsize=14)
ax.set_ylabel("Dim", fontsize=14)
ax.set_title("PC plane with original feature projections.", fontsize=16);

plt.savefig('Figures/PCA_log_red2.png',quality=50,format='png')

## DIM REDUCTION 3
PCAmod_log_red3 = PCA(n_components=3)
PCAmod_log_red3.fit(spendings_log_IQR)
PCAdata_log_red3 = PCAmod_log_red3.transform(spendings_log_IQR)

print('Shape reduced data =',np.shape(PCAdata_log_red3))
print('This is perfect, since the dimension is now 3 iso 6 while the amount of data-point remained the same. But is it an array now, I want to have it as a DF')


spendings_log_IQR_red3 = pd.DataFrame(PCAdata_log_red3,columns='Dim1 Dim2 Dim3'.split())
print(spendings_log_IQR_red3.head())

spendings_log_IQR_red3.to_pickle('DataFrames/spendings_log_IQR_red3')

## 3 D PLOTTING
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform

class Arrow3D(FancyArrowPatch):
    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._xyz = (x,y,z)
        self._dxdydz = (dx,dy,dz)

    def draw(self, renderer):
        x1,y1,z1 = self._xyz
        dx,dy,dz = self._dxdydz
        x2,y2,z2 = (x1+dx,y1+dy,z1+dz)

        xs, ys, zs = proj_transform((x1,x2),(y1,y2),(z1,z2), renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        super().draw(renderer)
        
def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    '''Add an 3d arrow to an `Axes3D` instance.'''

    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)

setattr(Axes3D,'arrow3D',_arrow3D)

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(111, projection='3d')


# scatterplot of the reduced data  
x=spendings_log_IQR_red3.loc[:, 'Dim1'].values.tolist()
y=spendings_log_IQR_red3.loc[:, 'Dim2'].values.tolist()
z=spendings_log_IQR_red3.loc[:, 'Dim3'].values.tolist()
ax.scatter(x, y, z, c='g', marker='o',s=50)

ax.set_xlabel('Dim1')
ax.set_ylabel('Dim2')
ax.set_zlabel('Dim3')
ax.set_title("PC space with original feature projections", fontsize=16);

feature_vectors = PCAmod_log_red3.components_.T
arrow_size, text_pos = 6.0, 8.0

for i in range(0,len(feature_vectors)):

    x_text= feature_vectors[i][0]*text_pos
    y_text= feature_vectors[i][1]*text_pos
    z_text= feature_vectors[i][2]*text_pos

    ax.text(x_text, y_text,z_text ,spendings_log_IQR.columns[i], color='black',ha='center', va='center', fontsize=18)

    dx_arrow=feature_vectors[i][0] * arrow_size
    dy_arrow=feature_vectors[i][1]* arrow_size
    dz_arrow=feature_vectors[i][2] * arrow_size

    ax.arrow3D(0,0,0,dx_arrow,dy_arrow,dz_arrow,
               mutation_scale=20,
               arrowstyle="-|>",
               ec ='black',
               fc='black')
plt.savefig('Figures/PCA_log_red3.png',quality=50,format='png')

## BOXCOX DATA
Explained_var_boxcox=PCAmod_boxcox.explained_variance_ratio_
Explained_var_boxcox.sum()

DFcomp_boxcox=pd.DataFrame(np.hstack((PCAmod_boxcox.components_,Explained_var_boxcox.reshape(-1,1))),columns=['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen','Explained Var'])
DFcomp_boxcox = DFcomp_boxcox.rename(index={0:'Dim 1',1:'Dim 2',2:'Dim 3',3:'Dim 4',4:'Dim 5',5:'Dim 6'})
print(DFcomp_boxcox)


fig, ax = plt.subplots(figsize = (14,8))
DFcomp_boxcox.drop('Explained Var',axis=1).plot(ax = ax, kind = 'bar');

for i in range(0,len(Explained_var_boxcox)):
    ax.text(i-0.40, ax.get_ylim()[1] + 0.05, "Explained Variance\n          %.4f"%(Explained_var_boxcox[i]))
plt.savefig('Figures/PCA_boxcox.png',quality=50,format='png')

PCAmod_boxcox_red2 = PCA(n_components=2)
PCAmod_boxcox_red2.fit(spendings_boxcox_IQR)
PCAdata_boxcox_red2 = PCAmod_boxcox_red2.transform(spendings_boxcox_IQR)

print('Shape reduced data =',np.shape(PCAdata_boxcox_red2))
print('This is perfect, since the dimension is now 2 iso 6 while the amount of data-point remained the same. But is it an array now, I want to have it as a DF')


spendings_boxcox_IQR_red2 = pd.DataFrame(PCAdata_log_red2,columns='Dim1 Dim2'.split())
spendings_boxcox_IQR_red2.to_pickle('DataFrames/spendings_boxcox_IQR_red2')


fig, ax = plt.subplots(figsize = (14,8))

# scatterplot of the reduced data    
ax.scatter(x=spendings_boxcox_IQR_red2.loc[:, 'Dim1'], y=spendings_boxcox_IQR_red2.loc[:, 'Dim2'], 
facecolors='g', edgecolors='g', s=50, alpha=0.5)

feature_vectors = PCAmod_boxcox_red2.components_.T

# we use scaling factors to make the arrows easier to see
arrow_size, text_pos = 5.0, 6.0,

# projections of the original features
for i in range(0,len(feature_vectors)):
    ax.arrow(0, 0, arrow_size*feature_vectors[i][0], arrow_size*feature_vectors[i][1], 
              head_width=0.2, head_length=0.2, linewidth=2, color='black')
    ax.text(feature_vectors[i][0]*text_pos, feature_vectors[i][1]*text_pos, spendings_boxcox_IQR.columns[i], color='black', 
             ha='center', va='center', fontsize=18)

ax.set_xlabel("Dim1", fontsize=14)
ax.set_ylabel("Dim", fontsize=14)
ax.set_title("PC plane with original feature projections.", fontsize=16);
plt.savefig('Figures/PCA_boxcox_red2.png',quality=50,format='png')




