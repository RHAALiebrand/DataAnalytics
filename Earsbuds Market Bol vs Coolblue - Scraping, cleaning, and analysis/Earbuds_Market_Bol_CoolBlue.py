# Analysis
import pandas as pd
import numpy as np

# Visualisation
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('whitegrid')
sns.set_context('notebook')
sns.set(font='times new roman',font_scale=1,palette='Greens')

# NLP
from nltk.corpus import stopwords
stop_words=stopwords.words('dutch')
import string as string_module

# Cut warnings for now
import warnings
warnings.filterwarnings('ignore')


## FUNCTIONS
def simplify_string(string,remove_stops,remove_earbud_words):
    nopunc = [char for char in string if char not in string_module.punctuation]
    nopunc=''.join(nopunc) 
    nopunc=nopunc.lower()
    if remove_stops:
        stop_words=stopwords.words('dutch')
        clean_string = [word for word in nopunc.split() if word.lower() not in stop_words]
        nopunc=' '.join(clean_string)
    if remove_earbud_words:
        earbud_words='volledige volledig oordopjes draadloos draadloze sport titaniumzwart wit zwart titanium goudbeige blauw goud beige koperzwart true wireless earphones donker donkergrijs'.split()
        clean_string = [word for word in nopunc.split() if word.lower() not in earbud_words]
        nopunc=' '.join(clean_string)   
    return nopunc


def levenshtein_ratio_and_dist(string1, string2, ratio_calc = False):
    import numpy as np
    # Initialize matrix of zeros
    R = len(string1)+1
    C = len(string2)+1
    dist = np.zeros((R,C),dtype = int)

    # Populate matrix of zeros with the indeces of each character of both strings
    for i in range(1, R):
        for k in range(1,C):
            dist[i][0] = i
            dist[0][k] = k

    # Iterate over the matrix to compute the cost of deletions,insertions and/or substitutions    
    for col in range(1, C):
        for row in range(1, R):
            if string1[row-1] == string2[col-1]:
                cost = 0 # If the characters are the same in the two strings in a given position [i,j] then the cost is 0
            else:
                if ratio_calc == True:
                    cost = 2
                else:
                    cost = 1
            dist[row][col] = min(dist[row-1][col] + 1,      # Cost of deletions
                                 dist[row][col-1] + 1,          # Cost of insertions
                                 dist[row-1][col-1] + cost)     # Cost of substitutions
    if ratio_calc == True:
        # Computation of the Levenshtein dist Ratio
        Ratio = ((len(string1)+len(string2)) - dist[row][col]) / (len(string1)+len(string2))
        return Ratio
    else:
        return "The strings are {} edits away".format(dist[row][col])
    

    
def stars_to_rating(star,cut):
    if star<cut: # This is set as 4.0 to have # low=95 and # high 153. Choice is arbitrary
        return 'Low'
    else:
        return 'High'
    

## READING WEBSCRAPES    
EarBuds_Bol=pd.read_pickle('EarBuds_Bol.csv')
EarBuds_CoolBlue=pd.read_pickle('EarBuds_CoolBlue.csv')

print('Number of manufacturers @ Bol: ',len(EarBuds_Bol['Manufacturer'].unique()))
print('Number of manufacturers @ CoolBlue: ',len(EarBuds_CoolBlue['Manufacturer'].unique()))

print('Average number of products per manufacturer @ Bol: ',len(EarBuds_Bol['Manufacturer'])/len(EarBuds_Bol['Manufacturer'].unique()))
print('Average number of products per manufacturer @ CoolBlue: ',len(EarBuds_CoolBlue['Manufacturer'])/len(EarBuds_CoolBlue['Manufacturer'].unique()))

plt.figure(figsize=(22,5))
sns.countplot(x='Manufacturer',data=EarBuds_Bol,order=EarBuds_Bol.Manufacturer.value_counts().iloc[:20].index,palette='Greens')
plt.title('Top 20 most frequently offered manufacturers on Bol')

plt.figure(figsize=(22,5))
sns.countplot(x='Manufacturer',data=EarBuds_CoolBlue,order=EarBuds_CoolBlue.Manufacturer.value_counts().iloc[:20].index,palette='Greens')
plt.title('Top 20 most frequently offered manufacturers on CoolBlue')


## LETS DISCOVER MANUFACTURERS
Name_count_Bol=EarBuds_Bol.groupby('Manufacturer').count()
Name_count_Bol=Name_count_Bol['Name']

Grouped_Man_Bol = pd.concat([EarBuds_Bol.groupby('Manufacturer').mean(),Name_count_Bol], axis=1, sort=False)
Grouped_Man_Bol['Price']=Grouped_Man_Bol['Price [EUR]']
Grouped_Man_Bol=Grouped_Man_Bol.sort_values('Name',ascending=False)
print('Bol:')
print(Grouped_Man_Bol.head())

## NOW FOR COOLBLUE
Name_count_CoolBlue=EarBuds_CoolBlue.groupby('Manufacturer').count()
Name_count_CoolBlue=Name_count_CoolBlue['Name']

Grouped_Man_CoolBlue = pd.concat([EarBuds_CoolBlue.groupby('Manufacturer').mean(),Name_count_CoolBlue], axis=1, sort=False)
Grouped_Man_CoolBlue['Price']=Grouped_Man_CoolBlue['Price [EUR]']
Grouped_Man_CoolBlue=Grouped_Man_CoolBlue.sort_values('Name',ascending=False)

print('CoolBlue:')
print(Grouped_Man_CoolBlue.head())

plt.figure(figsize=(22,5))
sns.barplot(x=Grouped_Man_Bol.index[0:20],y=Grouped_Man_Bol.Price.iloc[0:20],palette='Greens')
plt.ylim([0,200])

plt.figure(figsize=(22,5))
sns.barplot(x=Grouped_Man_CoolBlue.index[0:20],y=Grouped_Man_CoolBlue.Price.iloc[0:20],palette='Greens')
plt.ylim([0,200])


## LETS NOW COMPARE PRICES. FIRST PART IS CODE VERIFICATION
# Let's consider JBL
Manuf='JBL'

## Selection of comparison 
EarBuds_Bol_Compare = EarBuds_Bol[EarBuds_Bol['Manufacturer']==Manuf]
EarBuds_CoolBlue_Compare = EarBuds_CoolBlue[EarBuds_CoolBlue['Manufacturer']==Manuf] 

## Simplify names
EarBuds_Bol_Compare['Name']=EarBuds_Bol_Compare['Name'].map(lambda x: simplify_string(x,True,True))
EarBuds_CoolBlue_Compare['Name']=EarBuds_CoolBlue_Compare['Name'].map(lambda x: simplify_string(x,True,True))

## Remove duplicates from Bol since we are looping over Bol rows
EarBuds_Bol_Compare.drop_duplicates(subset ="Name",keep = 'first', inplace = True)

EarBuds_Bol_Compare['Name']


Name_corr = np.zeros((len(EarBuds_Bol_Compare['Name']),len(EarBuds_CoolBlue_Compare['Name'])))  # This will be 'the correlation matrix'. Aka the matrix which contains the results of the levenshtein_ratio_and_distance 
Product_combinations=[]
for j in range(0,len(EarBuds_Bol_Compare['Name'])):
    for i in range(0,len(EarBuds_CoolBlue_Compare['Name'])):
        Name_corr[j,i]=levenshtein_ratio_and_dist(EarBuds_Bol_Compare['Name'].iloc[j],EarBuds_CoolBlue_Compare['Name'].iloc[i],True)
    CoolBlue_row=np.argmax(Name_corr[j,:])
    if np.max(Name_corr[j,:]) > 0.85:
        Price_Bol=EarBuds_Bol_Compare['Price [EUR]'].iloc[j]    
        Price_CoolBlue=EarBuds_CoolBlue_Compare['Price [EUR]'].iloc[CoolBlue_row]    
        Product_combinations.append([Manuf,EarBuds_Bol_Compare['Name'].iloc[j],Price_Bol,Price_CoolBlue])
#with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
#    print(pd.DataFrame(Name_corr,index=EarBuds_Bol_Compare['Name'],columns=EarBuds_CoolBlue_Compare['Name']))


## PRICE COMPARISON
Product_combinations=[]
Product_combinations_diff=[]

Manufs='JBL Jabra Sony Bose Huawei'.split()
count=0
for Manuf in Manufs:
    ## Selection of comparison 
    EarBuds_Bol_Compare = EarBuds_Bol[EarBuds_Bol['Manufacturer']==Manuf]
    EarBuds_CoolBlue_Compare = EarBuds_CoolBlue[EarBuds_CoolBlue['Manufacturer']==Manuf] 
    
    ## Simplify names
    EarBuds_Bol_Compare['Name']=EarBuds_Bol_Compare['Name'].map(lambda x: simplify_string(x,True,True))
    EarBuds_CoolBlue_Compare['Name']=EarBuds_CoolBlue_Compare['Name'].map(lambda x: simplify_string(x,True,True))

    ## Remove duplicates from Bol since we are looping over Bol rows
    EarBuds_Bol_Compare.drop_duplicates(subset ="Name",keep = 'first', inplace = True)
    
    Name_corr = np.zeros((len(EarBuds_Bol_Compare['Name']),len(EarBuds_CoolBlue_Compare['Name'])))  # This will be 'the correlation matrix'. Aka the matrix which contains the results of the levenshtein_ratio_and_distance 

    for j in range(0,len(EarBuds_Bol_Compare['Name'])):
        for i in range(0,len(EarBuds_CoolBlue_Compare['Name'])):

            Name_corr[j,i]=levenshtein_ratio_and_dist(EarBuds_Bol_Compare['Name'].iloc[j],EarBuds_CoolBlue_Compare['Name'].iloc[i],True)
        CoolBlue_row=np.argmax(Name_corr[j,:])
        if np.max(Name_corr[j,:]) > 0.7:
            Price_Bol=EarBuds_Bol_Compare['Price [EUR]'].iloc[j]    
            Price_CoolBlue=EarBuds_CoolBlue_Compare['Price [EUR]'].iloc[CoolBlue_row]    
            Product_combinations.append([Manuf,EarBuds_Bol_Compare['Name'].iloc[j],Price_Bol,'Bol'])
            Product_combinations.append([Manuf,EarBuds_Bol_Compare['Name'].iloc[j],Price_CoolBlue,'CoolBlue'])
            Product_combinations_diff.append([Manuf,EarBuds_Bol_Compare['Name'].iloc[j],(Price_CoolBlue-Price_Bol)/Price_CoolBlue*100])
Product_combi = pd.DataFrame(Product_combinations,columns=['Manuf','Name','Price','Retailer'])
print(Product_combi.head(15))
Product_combi.drop([8,9],inplace=True)

Product_combi_diff = pd.DataFrame(Product_combinations_diff,columns=['Manuf','Name','Discount compared to CoolBlue'])
print(Product_combi_diff.head(15))
Product_combi_diff.drop([4],inplace=True)

## PLOT RESULTS
fig = plt.figure(figsize=(20,20))
ax=[0,0,0,0,0]
ax[0] = fig.add_subplot(321)
ax[1] = fig.add_subplot(323)
ax[2] = fig.add_subplot(325)
ax[3] = fig.add_subplot(222)
ax[4] = fig.add_subplot(224)
k=0
for Manuf in Manufs:
    #ax[k].set_title(Manuf)
    ax[k]=sns.barplot(x='Name', y='Price', hue='Retailer', data=Product_combi[Product_combi['Manuf']==Manuf],ax=ax[k])
    k+=1

fig = plt.figure(figsize=(20,20))
ax=[0,0,0,0,0]
ax[0] = fig.add_subplot(321)
ax[1] = fig.add_subplot(323)
ax[2] = fig.add_subplot(325)
ax[3] = fig.add_subplot(222)
ax[4] = fig.add_subplot(224)
k=0
for Manuf in Manufs:
    #ax[k].set_title(Manuf)
    ax[k]=sns.barplot(x='Name', y='Discount compared to CoolBlue', data=Product_combi_diff[Product_combi_diff['Manuf']==Manuf],ax=ax[k])
    k+=1
    
## DISCOUNT RATES
EarBuds_Bol['Discount']=EarBuds_Bol['Discount'].apply(lambda x: int(x[:-1]))
EarBuds_CoolBlue['Discount']=EarBuds_CoolBlue['Discount'].apply(lambda x: int(x[:-1]))
EarBuds_Bol.head()

EarBuds_Bol_disc=EarBuds_Bol[EarBuds_Bol['Discount']>0]
EarBuds_CoolBlue_disc=EarBuds_CoolBlue[EarBuds_CoolBlue['Discount']>0]

print('# of discounted items Bol: ',len(EarBuds_Bol_disc))
print('# of discounted items CoolBlue: ',len(EarBuds_CoolBlue_disc))


print('Percentage discount items Bol: ',round(len(EarBuds_Bol_disc)/len(EarBuds_Bol)*100,1),'%')
print('Percentage discount items CoolBlue: ',round(len(EarBuds_CoolBlue_disc)/len(EarBuds_CoolBlue)*100,1),'%')

EarBuds_Bol_description = EarBuds_Bol[['Manufacturer', 'Name','Stars [x/5.0]', 'S_count', 'Description', 'Pros', 'Cons']]
EarBuds_Bol_description=EarBuds_Bol_description[EarBuds_Bol_description['Stars [x/5.0]'] != 'No stars'] # Remove the no stars ratings
EarBuds_Bol_description['Stars [x/5.0]']=EarBuds_Bol_description['Stars [x/5.0]'].apply(float) 
EarBuds_Bol_description['S_count']=EarBuds_Bol_description['S_count'].apply(int)
EarBuds_Bol_description.head()


plt.figure(figsize=(20,5))
sns.countplot(EarBuds_Bol_description['S_count'],color='green')
plt.figure()
sns.distplot(EarBuds_Bol_description['Stars [x/5.0]'],kde=True,color='green')

## START OF NLP PROCESS
EarBuds_Bol_description['Rating']=EarBuds_Bol_description['Stars [x/5.0]'].apply(lambda x: stars_to_rating(x,4.0))
EarBuds_Bol_description['Description']=EarBuds_Bol_description['Description'].apply(lambda x: simplify_string(x,True,False))

## THIS WILL BE A LONG UNIT TEST, PIPELINES ARE USED LATER
from sklearn.feature_extraction.text import CountVectorizer

VectCount = CountVectorizer()
VectCount.fit(EarBuds_Bol_description['Description'])
print('Total amount of unique words = ',len(VectCount.vocabulary_))

## VECTORISE
FirstDescr_Bol = EarBuds_Bol_description['Description'][0]
FirstDescr_Bol=FirstDescr_Bol[0:200]
FirstDescr_Bol_trans = VectCount.transform([FirstDescr_Bol])
print(FirstDescr_Bol_trans)


## NORMALISE
from sklearn.feature_extraction.text import TfidfTransformer

TfidfTrans_first = TfidfTransformer()
TfidfTrans_first.fit(FirstDescr_Bol_trans)
FirstDescr_Bol_trans_norm=TfidfTrans_first.transform(FirstDescr_Bol_trans)
print(FirstDescr_Bol_trans_norm)

Descr_Bol_trans = VectCount.transform(EarBuds_Bol_description['Description'])
print(Descr_Bol_trans)


TfidfTrans = TfidfTransformer()
TfidfTrans.fit(Descr_Bol_trans)
Descr_Bol_trans_norm=TfidfTrans.transform(Descr_Bol_trans)

## APPLY MODEL
from sklearn.model_selection import train_test_split

descr_train, descr_test, star_train, star_test = train_test_split(Descr_Bol_trans_norm, EarBuds_Bol_description['Rating'], test_size=0.3,random_state=42)

from sklearn.linear_model import LogisticRegression
logRes = LogisticRegression()
logRes.fit(descr_train, star_train)
star_pred=logRes.predict(descr_test)

from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score

precisions=[]
recalls=[]
fscores=[]
precision,recall,fscore,support=score(star_test,star_pred,average='weighted')
precisions.append(precision)
recalls.append(recall)
fscores.append(fscore)


print(classification_report(star_test, star_pred))

## (LOOP) OVER MODELS
pred_models=['LogRes','SVM','NaiveBayes']

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

descr_train, descr_test, star_train, star_test = train_test_split(EarBuds_Bol_description['Description'], EarBuds_Bol_description['Rating'], test_size=0.3,random_state=42)

pipeline = Pipeline([
    ('bow', CountVectorizer()),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', SVC()),  # train on TF-IDF vectors w/ SVM classifier
])

pipeline.fit(descr_train,star_train)
star_pred = pipeline.predict(descr_test)

precision,recall,fscore,support=score(star_test,star_pred,average='weighted')
precisions.append(precision)
recalls.append(recall)
fscores.append(fscore)

print(classification_report(star_test, star_pred))


from sklearn.naive_bayes import MultinomialNB
pipeline = Pipeline([
    ('bow', CountVectorizer()),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])

pipeline.fit(descr_train,star_train)
star_pred = pipeline.predict(descr_test)

precision,recall,fscore,support=score(star_test,star_pred,average='weighted')
precisions.append(precision)
recalls.append(recall)
fscores.append(fscore)

print(classification_report(star_test, star_pred))

## GATHER RESULTS
describ_star_results_Bol=pd.DataFrame(np.vstack((precisions,recalls,fscores)),index=['Precision','Recall','Fscore'],columns=pred_models)

describ_star_results_Bol=describ_star_results_Bol.transpose()
describ_star_results_Bol=describ_star_results_Bol.reset_index()
describ_star_results_Bol.columns=['Models', 'Precision', 'Recall', 'Fscore']
print(describ_star_results_Bol)


## PLOT RESULTS
fig=plt.figure()
axes=fig.add_axes([0.1,0.1,0.8,0.8])
sns.scatterplot(x='Models', y='Precision',data=describ_star_results_Bol,color='green',label='Precision',s=100, marker="s")
sns.scatterplot(x='Models', y='Recall',data=describ_star_results_Bol,color='green',label='Recall',s=100, marker="+")
sns.scatterplot(x='Models', y='Fscore',data=describ_star_results_Bol,color='green',label='Recall',s=100, marker="d")

axes.set_ylabel('Value')


## NOW DO THE SAME FOR COOLBLUE
EarBuds_CoolBlue_description = EarBuds_CoolBlue[['Manufacturer', 'Name','Stars [x/5.0]', 'S_count', 'Description', 'Pros', 'Cons']]
EarBuds_CoolBlue_description=EarBuds_CoolBlue_description[EarBuds_CoolBlue_description['Stars [x/5.0]'] != 'No review']
EarBuds_CoolBlue_description['Stars [x/5.0]']=EarBuds_CoolBlue_description['Stars [x/5.0]'].apply(float)
EarBuds_CoolBlue_description['S_count']=EarBuds_CoolBlue_description['S_count'].apply(int)

plt.figure()
sns.countplot(EarBuds_CoolBlue_description['S_count'],color='green')
plt.figure()
sns.distplot(EarBuds_CoolBlue_description['Stars [x/5.0]'],color='green',kde=False)

EarBuds_CoolBlue_description['Rating']=EarBuds_CoolBlue_description['Stars [x/5.0]'].apply(lambda x: stars_to_rating(x,4.1)) # This is set as 4.0 to have # low=28 and # high 119. Choice is arbitrary 
EarBuds_CoolBlue_description['Description']=EarBuds_CoolBlue_description['Description'].apply(lambda x: simplify_string(x,True,False))


descr_train, descr_test, star_train, star_test = train_test_split(EarBuds_CoolBlue_description['Description'], EarBuds_CoolBlue_description['Rating'], test_size=0.3,random_state=42)

pipeline = Pipeline([
    ('bow', CountVectorizer()),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', LogisticRegression()),  # train on TF-IDF vectors w/ LogRes classifier
])

pipeline.fit(descr_train,star_train)
star_pred = pipeline.predict(descr_test)

precisions=[]
recalls=[]
fscores=[]
precision,recall,fscore,support=score(star_test,star_pred,average='weighted')
precisions.append(precision)
recalls.append(recall)
fscores.append(fscore)

print(classification_report(star_test, star_pred))

descr_train, descr_test, star_train, star_test = train_test_split(EarBuds_CoolBlue_description['Description'], EarBuds_CoolBlue_description['Rating'], test_size=0.3,random_state=42)

pipeline = Pipeline([
    ('bow', CountVectorizer()),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', SVC()),  # train on TF-IDF vectors w/ SVM classifier
])

pipeline.fit(descr_train,star_train)
star_pred = pipeline.predict(descr_test)

precision,recall,fscore,support=score(star_test,star_pred,average='weighted')
precisions.append(precision)
recalls.append(recall)
fscores.append(fscore)

print(classification_report(star_test, star_pred))


descr_train, descr_test, star_train, star_test = train_test_split(EarBuds_CoolBlue_description['Description'], EarBuds_CoolBlue_description['Rating'], test_size=0.3,random_state=42)

pipeline = Pipeline([
    ('bow', CountVectorizer()),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ NB classifier
])

pipeline.fit(descr_train,star_train)
star_pred = pipeline.predict(descr_test)

precision,recall,fscore,support=score(star_test,star_pred,average='weighted')
precisions.append(precision)
recalls.append(recall)
fscores.append(fscore)

print(classification_report(star_test, star_pred))


## REVIEW PREDICTIONS
# Initialise lists
Names_list=[]
Review_list=[]
Star_list=[]

Earbuds_Bol_review = EarBuds_Bol[['Name','Reviews']]
Earbuds_CoolBlue_review = EarBuds_CoolBlue[['Name','Reviews']]

# Read all bol data into the different lists
for i in range(0,len(Earbuds_Bol_review)):
    for j in range(0,len(Earbuds_Bol_review['Reviews'].iloc[i])):
        review=Earbuds_Bol_review['Reviews'].iloc[i][j]
        review_split=review.split(',')
        text=review_split[0]
        if len(review_split)>1: # Sometimes commas are used in the review so we have to merge these
            text=' '.join(review_split[:-1])
        Review_list.append(text)
        Star_list.append(float(review_split[-1]))
        Names_list.append(Earbuds_Bol_review['Name'].iloc[i])

# Read all coolblue data into the different lists
for i in range(0,len(Earbuds_CoolBlue_review)):
    for j in range(0,len(Earbuds_CoolBlue_review['Reviews'].iloc[i])):
        review=Earbuds_CoolBlue_review['Reviews'].iloc[i][j]
        review_split=review.split(',')
        text=review_split[0]
        if len(review_split)>1:
            text=' '.join(review_split[:-1])
        Review_list.append(text)
        if round(float(review_split[-1])/2,1)<1.0:
            Star_list.append(1.0)
        else:
            Star_list.append(round(float(review_split[-1])/2,1))
        Names_list.append(Earbuds_Bol_review['Name'].iloc[i])
len(Names_list)


Review_data = pd.DataFrame(np.vstack((Star_list,Review_list,Names_list)).transpose(),columns=['Stars','Review','Name'])
Review_data['Stars']=Review_data['Stars'].apply(float)
Review_data['Rating']=Review_data['Stars'].apply(lambda x:stars_to_rating(x,3.5))
Review_data=Review_data.reindex(columns='Stars Rating Review Name'.split())
Review_data.head()

print(Review_data['Rating'].value_counts())

plt.close('all')
sns.distplot(Review_data['Stars'],kde=True,color='green',bins=20)

descr_train, descr_test, star_train, star_test = train_test_split(Review_data['Review'], Review_data['Rating'], test_size=0.3,random_state=42)


pipeline = Pipeline([
    ('bow', CountVectorizer()),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', LogisticRegression()),  # train on TF-IDF vectors w/ LogRes classifier
])

pipeline.fit(descr_train,star_train)
star_pred = pipeline.predict(descr_test)

precisions=[]
recalls=[]
fscores=[]
precision,recall,fscore,support=score(star_test,star_pred,average='weighted')
precisions.append(precision)
recalls.append(recall)
fscores.append(fscore)

print(classification_report(star_test, star_pred))


pipeline = Pipeline([
    ('bow', CountVectorizer()),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', SVC()),  # train on TF-IDF vectors w/ LogRes classifier
])

pipeline.fit(descr_train,star_train)
star_pred = pipeline.predict(descr_test)

precision,recall,fscore,support=score(star_test,star_pred,average='weighted')
precisions.append(precision)
recalls.append(recall)
fscores.append(fscore)

print(classification_report(star_test, star_pred))


pipeline = Pipeline([
    ('bow', CountVectorizer()),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ LogRes classifier
])

pipeline.fit(descr_train,star_train)
star_pred = pipeline.predict(descr_test)

precision,recall,fscore,support=score(star_test,star_pred,average='weighted')
precisions.append(precision)
recalls.append(recall)
fscores.append(fscore)

print(classification_report(star_test, star_pred))

