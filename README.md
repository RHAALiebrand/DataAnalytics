# DataAnalytics
This folder contains several data-related projects I worked on or currently working on. Note that this folder is work in progress as I wrote all my scripts in Spyder and would like to covert them to Jupyter for visualisation and presentation purposes. The reason I code in spyder is purely for efficiently reasons - unit testing and coding itself is much more practicle in Spyder for me.

## Content
Below the most important topics I have worked on are listed together with the required Python tooling. The current Python version I am working with is 3.7.

*General tools: Spyder, Jupyter, Numpy, Pandas, Matplotlib*
### Machine Learning
[**Bank stocks: The Financial crisis versus COVID**](https://github.com/RHAALiebrand/DataAnalytics/blob/master/Bank%20Stocks%20FC%20vs%20COVID%20-%20API%2C%20visualisation%2C%20and%20supervised%20learning/Bank_Stocks_FC_COVID.ipynb): A supervised learning model utilising to analyse the recovery rate in the of EU and US banks stocks in the COVID crisis relative and the Financial Crisis. Furthermore, the predictability of the banks stocks was determined by using:
1. Linear regression
2. Epsilon-Support Vector Regression
3. Epsilon-Support Vector Regression in combination with a grid search algorithm

This resulted in remarkable observations which can be valuable to investors. Preditability and recovery are summarized in this figure:
 <img src="https://github.com/RHAALiebrand/DataAnalytics/blob/master/Bank%20Stocks%20FC%20vs%20COVID%20-%20API%2C%20visualisation%2C%20and%20supervised%20learning/Figures/KDE_FC_COVID.png" alt="KDE COVID FC" width="600"/>

[**Earbuds market: Bol versus CoolBlue**](https://github.com/RHAALiebrand/DataAnalytics/blob/master/Earsbuds%20Market%20Bol%20vs%20Coolblue%20-%20Scraping%2C%20cleaning%2C%20and%20analysis/Earbuds_Market_Bol_vs_CoolBlue.ipynb): A supervised learning model which predicts the total customer product rating based on the product description - it also extract product features which seem to be important to customers. As an extention, the same algorithms were trained on a much larger, but less consitent data-set: individual customer review with corresponding customer rating. Results are very much supprising and this project uncovers the need for reliable data. Since I approached this as a classification task, the employed algorithms are:
1. Logistic Regression
2. Supported Vector Machines
3. Naive Bayes
 
 [**Groceries: Customer Segments**](https://github.com/RHAALiebrand/DataAnalytics/blob/master/Groceries%20Customer%20Segments%20-%20Feature%20Engineering%2C%20PCA%2C%20and%20unsupervised%20learning/Groceries_Customer_Segments.ipynb): Two unsupervised learning models,
1. KMeans Clustering,
2. Gaussian Mixture Clustering,

are created which attempts to identify customer categories within an annual wholesale spending dataset. Data scaling is compared, logarithmic vs boxcox. Further, outliers are removed by means of [several methods](http://localhost:8888/notebooks/Desktop/PYTHON_FOR_DATA_SCIENCE/Projects/Groceries%20Customer%20Segments/Outlier_Detection.ipynb). A PCA is applied to reduce the dimensionality to an explained variance of 70 and 90% respectively, for visualisation purposes:
 
 <img src="https://github.com/RHAALiebrand/DataAnalytics/blob/master/Groceries%20Customer%20Segments%20-%20Feature%20Engineering%2C%20PCA%2C%20and%20unsupervised%20learning/Figures/Clusters_3_GM.png" alt="Clusters" width="400"/>

*Tools: Scikit, Seaborn, Datetime, Pipeline*

### Web-scraping
[**API Stock reader**](https://github.com/RHAALiebrand/DataAnalytics/blob/master/Bank%20Stocks%20FC%20vs%20COVID%20-%20API%2C%20visualisation%2C%20and%20supervised%20learning/API_Stock_Reader.ipynb): This data reader utilises API tools to acquire stock data from https://www.alphavantage.co/. The code reads the data for six large banks:

*European*
1. ING, Dutch (Rabobank is not AEX-listed)
2. Deutsche Bank, German
3. HSBC, English

*American*
1. CitiGroup
2. Goldman Sachs
3. JPMorgan Chase

[**CoolBlue websraping tool**](https://github.com/RHAALiebrand/DataAnalytics/blob/master/Earsbuds%20Market%20Bol%20vs%20Coolblue%20-%20Scraping%2C%20cleaning%2C%20and%20analysis/WebScrape_CoolBlue.ipynb) + [**Bol websraping tool**](https://github.com/RHAALiebrand/DataAnalytics/blob/master/Earsbuds%20Market%20Bol%20vs%20Coolblue%20-%20Scraping%2C%20cleaning%2C%20and%20analysis/WebScrape_Bol.ipynb): These codes extract all information of the offered earbuds on Bol and CoolBlue. Extracted information includes:
1. Manufacturer
2. Product Name
3. Price (Current, Retail, and discount)
4. Pros and Cons
5. Description
6. Rating
7. Reviews
 
The eventual dataframes were saved and read by other parts of the code to analyse it.
*Tools: Pandas data-reader, BeautifulSoup, Requests*

### Natural Language Processing
[**Levenshtein calculator**](https://github.com/RHAALiebrand/DataAnalytics/blob/master/Earsbuds%20Market%20Bol%20vs%20Coolblue%20-%20Scraping%2C%20cleaning%2C%20and%20analysis/WebScrape_CoolBlue.ipynb): An NLP technique to calculate the similarity of two strings. The algorithm deterimes:
1. The 'distance' of the strings, i.e. how many changes are needed to convert them
2. The Levenshtein ratio, which is a measure how similar they are based on the distance and original string metrics
The latter is used to deteremine product similarity in the earbuds project. 

[**Earbuds market: Bol versus CoolBlue**](https://github.com/RHAALiebrand/DataAnalytics/blob/master/Earsbuds%20Market%20Bol%20vs%20Coolblue%20-%20Scraping%2C%20cleaning%2C%20and%20analysis/Earbuds_Market_Bol_vs_CoolBlue.ipynb): In this project the Levenshtein calculator is employed to match product for both websites and create a price comparison. All text is first pre-processed and cleaned to make predictions as accurate as possible. Furthermore, the TDI-FD technique is employed to convert product descriptions and reviews in . Pipelines are used to make the script most efficient. A general pipeline is build as:
1. Text vectorizer
2. Vector scaler / normaliser
3. Predictive model, in this case a classifier
*Tools: CountVectorizer, NLTK, Tfidf*

### Preprocessing, Analysis and visualisation
[**Bank stocks: The Financial crisis versus COVID**](https://github.com/RHAALiebrand/DataAnalytics/blob/master/Bank%20Stocks%20FC%20vs%20COVID%20-%20API%2C%20visualisation%2C%20and%20supervised%20learning/Bank_Stocks_FC_COVID.ipynb): Time series of stock data are plotted and analysed using Matplotlib and Seaborn. Sample plots are, line, box, heatmaps, pair, histograms, scatter, kde, correlations maps. Furthermore, interactive plotting by means of Pyplot in combinations with Cufflinks is esthablished. 

[**911 Calls: What, why, and where**](https://github.com/RHAALiebrand/DataAnalytics/blob/master/Bank%20Stocks%20FC%20vs%20COVID%20-%20API%2C%20visualisation%2C%20and%20supervised%20learning/Bank_Stocks_FC_COVID.ipynb): The 911 calls in Montgomery County are analysed by means of visualisation. Geographical and reason dependent visualisation were the main topics. To create geographical maps, basemap was employed. Ideally I was aiming to create some choropleth maps, however, the considered area was to small for these. I will do this for another project.

[**Customer outlier detection**](http://localhost:8888/notebooks/Desktop/PYTHON_FOR_DATA_SCIENCE/Projects/Groceries%20Customer%20Segments/Outlier_Detection.ipynb): This program identifies and removes the outliers from a given data-set. Four different algorithms are implemented and tested:
*Self implemented*
1. [Tukey's method](https://sphweb.bumc.bu.edu/otlt/mph-modules/bs/bs704_summarizingdata/bs704_summarizingdata7.html) 
2. [Z-Scores](https://support.hach.com/ci/okcsFattach/get/1008007_4) 

*Sci-kit*
1. [Isolation Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html) 
2. [Elliptic Envelope](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.EllipticEnvelope.html)

<img src="https://github.com/RHAALiebrand/DataAnalytics/blob/master/Groceries%20Customer%20Segments%20-%20Feature%20Engineering%2C%20PCA%2C%20and%20unsupervised%20learning/Figures/Outlier_log_2.png" alt="Outliers" width="400"/>


[**Customer Principle Component Analysis (PCA)**](https://github.com/RHAALiebrand/DataAnalytics/blob/master/Groceries%20Customer%20Segments%20-%20Feature%20Engineering%2C%20PCA%2C%20and%20unsupervised%20learning/Customer_Principle%20Components_Analysis.ipynb) In this piece, PCA is applied on log and boxcox scaled data to identify the differences. A dimensionality reduction from 6 to 2 and 3, to respectively achieve a explained variance of 70% and 90%:
<img src="https://github.com/RHAALiebrand/DataAnalytics/blob/master/Groceries%20Customer%20Segments%20-%20Feature%20Engineering%2C%20PCA%2C%20and%20unsupervised%20learning/Figures/PCA_dist_log.png" alt="Explained Variance" width="600"/>

 *Tools: Pyplot,Cufflinks, Basemap, IsolationForest, EllipticEnvelope, PCA, 3DAxes *



## Educational Background and Contact
These projects were fully volotarily and purely done because of my interest and curiosity. The knowledge required to complete these courses was gathered by means of several resources:

1. Recent TU Delft courses provided by the [Interative Intelligence of TU Delft's math faculty (EWI)](https://www.tudelft.nl/ewi/over-de-faculteit/afdelingen/intelligent-systems/interactive-intelligence/) 
   - [Machine Learning 1](https://studiegids.tudelft.nl/a101_displayCourse.do?course_id=51391)
   - [Machine Learning 2](https://studiegids.tudelft.nl/a101_displayCourse.do?course_id=51392) 
   - [Deep Learning](https://studiegids.tudelft.nl/a101_displayCourse.do?course_id=51998) 
2. Books such as
   - An Introduction to Statistical Learning by Gareth James
   - Data Science from Scratch by Joel Grus
   - Data Science for Business What you need to know about data mining and data-analytic thinking by Foster Provost
3. Courses provided by Udemy and Coursera

All required math and statistics was of course already present due to my background in aerospace engineering. Further, my first programming language is Python and was taught in my first year at Delft University of Technology. I expended this by learning Matlab, R, and STATA. Besides, I have experience in object oriented languages (C++ and Fortran) due to my internships. 

If you have any questions or comments still, please do not hesitate to contact me as I am happy to share my experience / codes.
- rensliebrand@gmail.com
- [LinkedIn](https://www.linkedin.com/in/rensliebrand/) 

