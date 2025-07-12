# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 16:54:53 2025

@author: librarypc
"""
import pandas as pd
from pandas import read_html
import requests
import numpy as np

data = 'https://stockanalysis.com/ipos/2023/'
url_2023 = requests.get(data)
ipos_2023 = read_html(url_2023.text)
df_2023 = ipos_2023[0]              #df_2023 will contain primary IPO data for 2023

# Cleaning data and converting from string to numeric
clean_data = {
    'Return' : r'[%]', 
    'IPO Price' : r'[$]', 
    'Current' : r'[$]'
    }
for col, pattern in clean_data.items():
    df_2023[col] = pd.to_numeric(df_2023[col].replace(pattern, '', regex=True), errors='coerce')


# Here, I'm calculating summary stats for Return Rates, IPO Price, and Current Price 
# Then placing the results in a dataframe for easier readability
cols = ['Return', 'IPO Price', 'Current']
summary = {}
for col in cols:
    data = df_2023[col].dropna().T
    summary[col] = {
        'Number of valid integers' : len(data), 
        'Average' : round(data.mean(), 2), 
        'Median' : data.median(), 
        'Min' : data.min(), 
        'Max' : data.max(), 
        'Std. Dev' : round(data.std(), 2),
        'Skewness' : round(data.skew(), 2), 
        'Kurtosis' : round(data.kurtosis(), 2)
        }
df_summary = pd.DataFrame(summary)

# Here, I'm calculating the percentage of companies that had negative returns in 2023
returns = df_2023['Return'].dropna()
neg_returns = (returns < 0).sum()
total_returns = returns.count()
perc_negative = (neg_returns / total_returns) * 100
print(total_returns)
print(round(perc_negative, 2))
# 67.32% of the 153 companies that offered prices in 2023 experienced negative returns.

# Next, I'll plot the summary stats for all three variables on a histogram to better visualize the distribution
import matplotlib.pyplot as plt
import seaborn as sns
import os

output_location = r'c:\code_and_data\code_and_data\python\python_homework'

histogram_filepath = os.path.join(output_location, '2023_IPO_Distributions.png')

cols = ['Return', 'IPO Price', 'Current']
plt.figure(figsize=(18,6))
for i, col in enumerate(cols):
    plt.subplot(1,3,i+1)
    sns.histplot(df_2023[col].dropna(), bins=50, kde=False, color='forestgreen', edgecolor='black')
    
    skew = df_summary[col]['Skewness']
    kurt = df_summary[col]['Kurtosis']
    
    plt.title(f"2023 {col} Distribution \nSkewness {skew} \nKurtosis {kurt}")
    plt.xlabel(col)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(histogram_filepath)
plt.show()


# Summary Interpretation: The dataset is comprised of 154 company IPOs
# Return Analysis: 
        # Average return= -21.51%, Median= -45.8%
            # company's typically lost almost half of their value
        # Min= -99.95%, Max= 497.5%
            # Some companies went close to zero, 
            # while a small number of positive outliers inflated the average
        # Standard Deviation= 85.46
            # High negative variability and high risk for investors
        #Skewness= 2.66, Kurtosis= 10.48
            # Few outliers have dragged the distribution's average up,
            # while the majority of the dataset's returns are low
# IPO Price Analysis: 
        #Average price= $9.86, Median= $7.50
            # Fairly low priced IPO market
            # could be early stage companies
        #Skewness= 2.48, Kurtosis= 7.8
            # A small number of IPOs are priced very high,
            # but most remind within the $5-$10 range
# Current Price Analysis:
    # Average current price= $10.65, Median current price= $3.19
        # The average is priced slightly above IPO levels,
        # but the median tells that the typical stock is worth less than half its IPO price
    # Max= $148.55
        # Caused by a small number of high outliers distorting the average
    #Skewness= 3.99, Kurtosis= 20.73
        # The majority of stocks fell while very few performed exceptionally
    

# Next, I'm splitting the dataframe (df_2023) into two separate dataframes:
    # top thirty companies with the highest returns(df_top_thirty), and the rest of the datase(df_rest)
# I want to calculate the summary stats for each dataset for comparison to identify the influence of posible outliers
df_top_thirty = df_2023.sort_values(by='Return', ascending=False).head(30)
df_rest = df_2023[~df_2023.index.isin(df_top_thirty.index)]

import scipy.stats as stats

def summary_stats(df):
    returns = df['Return'].dropna()
    return {
        'Number of companies' : len(returns),
        'Average Return' : round(returns.mean(), 2),
        'Median Return' : returns.median(), 
        'Std. Dev' : round(returns.std(), 2), 
        'Skewness' : round(returns.skew(), 2),
        'Kurtosis' : round(returns.kurtosis(), 2),
        'Min Return' : returns.min(), 
        'Max Return' : returns.max()
        }
top_thirty_stats = summary_stats(df_top_thirty)
rest_stats = summary_stats(df_rest)

df_comparison = pd.DataFrame ({
    'Top Thirty' : top_thirty_stats,
    'Rest of Dataset' : rest_stats
    })

# Here, I'll create a boxplot to visualize the different nature of return rate distributions between the Top Thirty companies and the Rest of the dataset
boxplot_filepath = os.path.join(output_location, '2023_Return_Comparison_Boxplot')

df_top_thirty = df_top_thirty.dropna(subset=['Return'].copy())        
df_top_thirty['Group'] = 'Top 30'

df_rest = df_rest.dropna(subset=['Return'].copy())
df_rest['Group'] = 'Rest of Dataset'

df_combine = pd.concat ([
    df_top_thirty[['Return', 'Group']],
    df_rest[['Return', 'Group']]
    ],ignore_index=True
    )

plt.figure(figsize=(8,6))
sns.boxplot(data=df_combine, x='Group', y='Return', hue='Group', palette='pastel', legend=False)
plt.title('2023 Return Distribution: Top 30 Companies vs. Rest of Companies')
plt.xlabel('')
plt.ylabel('Return %')
plt.tight_layout()
plt.savefig(boxplot_filepath)
plt.show()

# IPO Return Rate Comparison (Top Thirty Companies vs Rest of the Dataset):
# Top Thirty Earners:
    # Average Return= 104.23%, Median return= 57.45%
        # The average IPO doubled it's value
        # The typical IPO experience consistent gains 
    # Min return= 14.7%, Max= 497.5%
        # Every company in this group had positive returns
        # The highest return multiplied its value 5 times
    # Skewness= 1.94, Kurtosis= 4.45
        # Overall, this group is consistently outperforming
# Rest of the Dataset:
    # Average Return: -53.19%, Median= -62.10%
        # Companies typically lost about 62% of their value
    # Min return= -99.95%, Max return= 14.31%
        # The highest return for this group was less than the lowest return in the Top Thirty Group
    # Skewness= 0.54, Kurtosis= -1.02
        # Tight cluster of negative returns with no positive outliers.
