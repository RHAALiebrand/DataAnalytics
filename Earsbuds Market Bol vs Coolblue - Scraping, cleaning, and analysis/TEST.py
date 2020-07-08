# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 11:40:22 2020

@author: rensl
"""


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


pages=[1]
for Npage in pages: # Loop over pages
    print('Page=',Npage)
    ## READ PAGES
    page_read = BeautifulSoup(requests.get(url+str(Npage)).text, 'html.parser')
    products = page_read.find_all('div', {'class': 'product-card grid-container grid-container-xs--gap-4--x grid-container-xs--gap-2--y grid-container-l--gap-4--y js-product grid-container-l--gap-1--x'})
    products=products[0:5]
    for product in products: # Loop over products per page
        ## MANUFACTURER AND PRODUCT NAME
        Prod_name=join_string(product.h3.text.split()[1:])
        Man_name=product.h3.text.split()[0]
        print(Prod_name)
        
        
        ## DESCRIPTION
        product_url=product.find('div',{'class':'product-card__title'}).a.get('href')
        product_page_read = BeautifulSoup(requests.get('https://www.coolblue.nl'+product_url).text, 'html.parser')
        # ## REVIEWS        
        #reviews=product_page_read.select('div[class*="grid-container-xs--gap-4--x"]')
        reviews=product_page_read.find_all('p',{'class':"grid-section-xs--gap-2 no-padding-bottom"})
        
        print(len(reviews))
        review_list=[]
        # for i in range(0,len(reviews)):
        #       review=reviews[i+12].p.text.split()
        #       review_star=reviews[i+12].find('span',{'class':'review-rating__reviews overflow--ellipsis'}).text
        #       review_star=int(get_digits(review_star))/10/2
        #       review_list.append(' '.join(review)+', '+ str(review_star))
        
        