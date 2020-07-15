# DataAnalytics
This folder contains several data-related projects I did the last year. Note that this folder is work in progress as I wrote all my scripts in Spyder and would like to covert them to Jupyter for visualisation and presentation purposes. The reason I code in spyder is purely for efficiently reasons - unit testing and coding itself is much more practicle in Spyder for me.

## Content
Below the most important topics I have worked on are listed together with the required Python tooling. The current Python version I am working with is 3.7.

*General tools: Spyder, Jupyter, Numpy, Pandas, Matplotlib*
#### Machine Learning
[**Bank stocks: The Financial crisis versus COVID**](https://github.com/RHAALiebrand/DataAnalytics/blob/master/Bank%20Stocks_FC%20vs%20COVID%20-%20Gathering%2C%20visualisation%2C%20and%20analysis/Bank_Stocks_FC_COVID.ipynb): A supervised learning model utilising to analyse the recovery rate in the of EU and US banks stocks in the COVID crisis relative and the Financial Crisis. Furthermore, the predictability of the banks stocks was determined by using:
   - Linear regression
   - Epsilon-Support Vector Regression
   - Epsilon-Support Vector Regression in combination with a grid search algorithm

[**Earbuds market: Bol versus CoolBlue**](https://github.com/RHAALiebrand/DataAnalytics/blob/master/Earsbuds%20Market%20Bol%20vs%20Coolblue%20-%20Scraping%2C%20cleaning%2C%20and%20analysis/Earbuds_Market_Bol_vs_CoolBlue.ipynb): A supervised learning model which predicts the total customer product rating based on the product description - it also extract product features which seem to be important to customers. As an extention, the same algorithms were trained on a much larger, but less consitent data-set: individual customer review with corresponding customer rating. Results are very much supprising and this project uncovers the need for reliable data. Since I approached this as a classification task, the employed algorithms are:
   - Logistic Regression
   - Supported Vector Machines
   - Naive Bayes
   
*Tools: Scikit, Seaborn, Datetime, Pyplot + Cufflinks (interactive plotting), Pipeline*

#### Web-scraping
[**CoolBlue websraping tool**](https://github.com/RHAALiebrand/DataAnalytics/blob/master/Earsbuds%20Market%20Bol%20vs%20Coolblue%20-%20Scraping%2C%20cleaning%2C%20and%20analysis/WebScrape_CoolBlue.ipynb) 

[**CoolBlue websraping tool**](https://github.com/RHAALiebrand/DataAnalytics/blob/master/Earsbuds%20Market%20Bol%20vs%20Coolblue%20-%20Scraping%2C%20cleaning%2C%20and%20analysis/WebScrape_CoolBlue.ipynb) + [**Bol websraping tool**](https://github.com/RHAALiebrand/DataAnalytics/blob/master/Earsbuds%20Market%20Bol%20vs%20Coolblue%20-%20Scraping%2C%20cleaning%2C%20and%20analysis/WebScrape_Bol.ipynb): These codes extract all information of the offered earbuds on Bol and CoolBlue. Extracted information includes:
 - Manufacturer
 - Product Name
 - Price (Current, Retail, and discount)
 - Pros and Cons
 - Description
 - Rating
 - Reviews
The eventual dataframes were saved and read by other parts of the code to analyse it.
*Tools: BeautifulSoup, Requests, *
#### Natural Language Processing

#### Analysis and visualisation







These projects were fully volotarily and purely done because of my interest and curiosity. The knowledge required to complete these courses was gathered by means of several resources:

1. Recent TU Delft courses list item
2. Books such as
   - An Introduction to Statistical Learning by Gareth James
   - Data Science from Scratch by Joel Grus
   - Data Science for Business What you need to know about data mining and data-analytic thinking by Foster Provost
3. Courses provided by Udemy and Coursera

All required math and statistics was of course already present due to my background in aerospace engineering. Further, my first programming language is Python and was taught in my first year at TU Delft. I expended this by learning Matlab, R, and STATA. Besides, I have experience in object oriented languages (C++ and Fortran) due to my internships. 

