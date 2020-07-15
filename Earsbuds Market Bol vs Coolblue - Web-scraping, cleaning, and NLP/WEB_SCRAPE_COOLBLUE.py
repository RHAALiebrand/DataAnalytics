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

def get_digits(str1):
    c = ""
    for i in str1:
        if i.isdigit():
            c += i
    return c

## INITIALISE VARIABLES
CoolBlue_data=[]
url = 'https://www.coolblue.nl/oordopjes/draadloos?pagina='
endpage=9
pages=np.arange(1,endpage+1)
# pages=[1]
for Npage in pages: # Loop over pages
    print('Page=',Npage)
    ## READ PAGES
    page_read = BeautifulSoup(requests.get(url+str(Npage)).text, 'html.parser')
    products = page_read.find_all('div', {'class': 'product-card grid-container grid-container-xs--gap-4--x grid-container-xs--gap-2--y grid-container-l--gap-4--y js-product grid-container-l--gap-1--x'})
    # products=products[2:3]
    for product in products: # Loop over products per page
        ## MANUFACTURER AND PRODUCT NAME
        Prod_name=join_string(product.h3.text.split()[1:])
        Man_name=product.h3.text.split()[0]
        print(Prod_name)
        ## PRICING
        try:
            Price_string=product.find('span','sales-price sales-price--small js-sales-price').strong.text
            Price_list=[int(s) for s in Price_string.split(',') if s.isdigit()] #Convert to ints
            Price=price_list_to_price(Price_list) #Concat int list to a price
            
            Retail_Price=Price
        # Sometimes an 'advies' prijs and current price is given, lets get them    
        except:  
            Retail_Price_string=product.find('span','sales-price__former-price').text
            Retail_Price_list=Retail_Price_string.split(',')
            Retail_Price_list_ints=[]
            for Retail_Price_str_el in Retail_Price_list:
                element=get_digits(Retail_Price_str_el)
                if len(element)>0:
                    Retail_Price_list_ints.append(int(element))
            Retail_Price=price_list_to_price(Retail_Price_list_ints)
    
            
            Price_string=product.find('strong','sales-price__current').text
            Price_list=[int(s) for s in Price_string.split(',') if s.isdigit()] #Convert to ints
            Price=price_list_to_price(Price_list) #Concat int list to a price    print(Prod_name) 
        
        # Determine discount rates
        Discount=-int((Price-Retail_Price)/Retail_Price*100)
        Discount=str(Discount)+'%'
    
       ## STAR RATING
        Review=product.find('div',{'class':'review-rating__icons'}).get('title').split()
        Stars=float(Review[0])
        Star_count=int(Review[-2])
        if Star_count==0:
            Stars='No review'
            
        ## DESCRIPTION
        product_url=product.find('div',{'class':'product-card__title'}).a.get('href')
        product_page_read = BeautifulSoup(requests.get('https://www.coolblue.nl' +product_url).text, 'html5lib')
        
        try:
            descr=product_page_read.find('div',{'class':'js-product-description'}).p.text.split()
            descr=' '.join(descr)
        except:
            descr='No description'
        
         ## PROS 
        points=product_page_read.find_all('div',{'class':'js-pros-and-cons'})[0]
        labels=points.find_all('span',{'class':'screen-reader-only'})
        points_list=points.find_all('div')
        plus_list=[]
        minus_list=[]
        for i in range(0,len(labels)):
            label=labels[i].text
            if label == 'Pluspunt:':
                plus=points_list[2+3*i].text.split()
                plus_list.append(' '.join(plus))
            elif label == 'Minpunt:':
                minus=points_list[2+3*i].text.split()
                minus_list.append(' '.join(minus))
        
        ## REVIEWS       
        reviews=product_page_read.select('p[class*="grid-section-xs--gap-3"]')
        review_stars=product_page_read.find_all('span',{'class':'review-rating__reviews overflow--ellipsis'})
        
        got_expert_review=product_page_read.text.find('Onze specialistenreview') # Sometimes the first review is an expert review which should be removed
        if got_expert_review>0:
            reviews=reviews[1:]
        
        # Load star ratings in list and split this list to only include reviews given in the subwindow
        star_list=[]
        for review_star in review_stars:
            if len(get_digits(review_star.text))==2:
                star_for_review=int(get_digits(review_star.text))
                if star_for_review==10:
                    star_list.append(star_for_review)
                else:
                    star_list.append(star_for_review/10)
        
        # Remove rest of star_ratings
        for k in range(int(len(star_list)/2),len(star_list)):
            star_list[k]='Double review'  # Rather do it this way for code robustness
        
        # Put them together in a review list
        star_list=np.flip(star_list)    
        review_list=[]
        for i in range(0,int(len(reviews)/2)):
              review=reviews[i].text.split()
              review_list.append(' '.join(review)+', '+ str(star_list[len(star_list)-1-i]))#
              if str(star_list[len(star_list)-1-i])=='Double review':
                  review_list=review_list[:-1]
                  
        CoolBlue_data.append([Man_name,Prod_name,Price,Discount,round(Retail_Price,2),Stars,Star_count,descr,plus_list,minus_list,review_list])


# Create DF
col_names=['Manufacturer','Name','Price [EUR]','Discount','Ret P [EUR]','Stars [x/5.0]', 'S_count','Description','Pros','Cons','Reviews']
EarBuds_CoolBlue = pd.DataFrame(np.array(CoolBlue_data),columns=col_names)
EarBuds_CoolBlue['Price [EUR]']=EarBuds_CoolBlue['Price [EUR]'].apply(lambda x: float(x))
EarBuds_CoolBlue['Ret P [EUR]']=EarBuds_CoolBlue['Ret P [EUR]'].apply(lambda x: float(x))
EarBuds_CoolBlue['S_count']=EarBuds_CoolBlue['S_count'].apply(lambda x: int(x))

EarBuds_CoolBlue.to_pickle('EarBuds_CoolBlue')
