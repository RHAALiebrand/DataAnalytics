import os
from bs4 import BeautifulSoup
import requests
import numpy as np
import pandas as pd
import re
from math import nan

## REQUIRED FUNCTIONS
def join_string(word_list):
    string=''
    for word in word_list:
        string=string+' '+word
    return string[1:]

def price_list_to_price(price_list):
    if len(price_list)==1:
        return price_list[0]
    else:
        return price_list[0]+price_list[1]/100

def discount_list_to_discount(price_list):
    if len(price_list)==1:
        return price_list[0]
    else:
        return price_list[0]*10+price_list[1]

def get_digits(str1):
    c = ""
    for i in str1:
        if i.isdigit():
            c += i
    return c
    
    
## INITIALISE VARIABLES
Bol_data=[]
url = "https://www.bol.com/nl/l/volledig-draadloze-oordopjes/N/8440+44349/?page="
endpage=13
pages=np.arange(1,endpage+1)
#pages=[3]   # For debugging
for Npage in pages:   # Loop over pages
    print('Page=',Npage)
    ## READ PAGES
    page_read = BeautifulSoup(requests.get(url+str(Npage)).text, 'html5lib')
    products = page_read.findAll("li",{"class":'product-item--row'})
    
    #products=products[3:4]  # For debugging
    for product in products:   # Loop over products per page
        ## MANUFACTURER AND PRODUCT NAME
        # Sometimes manufacurer is given separately, sometimes in title. Let's check this:
        Man_name=product.find('ul','product-creator').a.text
        Prod_name=product.find('div','product-title--inline').a.text
        
        if Man_name == 'Merkloos':
            Man_name=Prod_name.split()[0]
        
        if Prod_name.split()[0]==Man_name:
            Prod_name=join_string(Prod_name.split()[1:-1])
        
        ## PRICING
        try:
            Price_string=product.find('span','promo-price').text
            Price_list=[int(s) for s in Price_string.split() if s.isdigit()] #Convert to ints
            Price=price_list_to_price(Price_list) #Concat int list to a price
        except:
            Price=nan
        ## Let's identify the discount items and calculate the normal retail price
        try:
            Discount_string=product.select('p[class*="product-prices"]')[0].strong.text
            Discount=Discount_string.split()[-1]
            
            Discount_list=[int(i) for i in list(Discount)[:-1]] 
            Retail_price=100*Price/(100-discount_list_to_discount(Discount_list))
        except IndexError:
            Discount = '0%'
            Retail_price=Price
        
        ## STAR RATING
        Stars=product.find('div',{'class':'star-rating'}).get('title').split()[1]
        Star_count=product.find('div',{'class':'star-rating'}).get('data-count')
        
         # Sometimes no reviews are given yet
        if Stars == 'zijn':
            Stars = 'No stars'
            Star_count=0
        
        ## DESCRIPTION
        product_url=product.find('p',{'class':'medium--is-visible'}).a.get('href')
        product_page_read = BeautifulSoup(requests.get('https://www.bol.com/' +product_url).text, 'html5lib')
        
        try:
            descr=product_page_read.find('div',{'data-test':'description'}).text.split()
            descr=' '.join(descr)
        except:
            descr='No description'
            
        ## PROS
        plusses=product_page_read.find_all('li',{'class':'pros-cons-list__item pros-cons-list__item--pro'})
        plus_list=[]
        for plus in plusses:
            plus=plus.text.split()
            plus_list.append(' '.join(plus))

        ## CONS
        minusses=product_page_read.find_all('li',{'class':'pros-cons-list__item pros-cons-list__item--con'})
        minus_list=[]
        for minus in minusses:
            minus=minus.text.split()
            minus_list.append(' '.join(minus))
        
        ## REVIEWS         
        reviews=product_page_read.find_all('div',{'class':'review__body'})
        review_list=[]
        
        review_stars=product_page_read.find_all('div',{'class':'star-rating'})

        for i in range(0,len(reviews)):
            review=reviews[i].p.text.split()
            review_star=float(get_digits(review_stars[i+12].span.get('style')))/20
            review_list.append(' '.join(review)+', '+ str(review_star))


        Bol_data.append([Man_name,Prod_name,Price,Discount,round(Retail_price,2),Stars,Star_count,descr,plus_list,minus_list,review_list])


# Create DF
col_names=['Manufacturer','Name','Price [EUR]','Discount','Ret P [EUR]','Stars [x/5.0]', 'S_count','Description','Pros','Cons','Reviews']
EarBuds_Bol = pd.DataFrame(np.array(Bol_data),columns=col_names)
EarBuds_Bol['Price [EUR]']=EarBuds_Bol['Price [EUR]'].apply(lambda x: float(x))
EarBuds_Bol['Ret P [EUR]']=EarBuds_Bol['Ret P [EUR]'].apply(lambda x: float(x))
EarBuds_Bol['S_count']=EarBuds_Bol['S_count'].apply(lambda x: int(x))

EarBuds_Bol.to_pickle('EarBuds_Bol')
