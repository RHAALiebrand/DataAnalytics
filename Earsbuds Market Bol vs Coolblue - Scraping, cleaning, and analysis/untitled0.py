# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 11:53:33 2020

@author: rensl
"""


import os
from bs4 import BeautifulSoup
import requests
import numpy as np
import pandas as pd
import re
from math import nan


product_page_read = BeautifulSoup(requests.get('https://www.coolblue.nl/product/852042/apple-airpods-pro-met-draadloze-oplaadcase.html').text, 'html.parser')

print(product_page_read)
reviews2=product_page_read.select('p[class*="grid-section-xs--gap-3"]')

print(len(reviews2))