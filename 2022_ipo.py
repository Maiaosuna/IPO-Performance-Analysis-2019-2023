# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 22:09:25 2025

@author: librarypc
"""

import numpy as np
import pandas as pd
from pandas import read_html
import requests

data = 'https://stockanalysis.com/ipos/2022/'
url_2022 = requests.get(data)
ipos_2022 = pd.read_html(url_2022.text)
# df_2022 will contain the ipo data for the year 2022
df_2022 = ipos_2022[0]

clean_data = {
    'Return' : r'[%]',
    'IPO Price' : r'[$]', 
    'Current' : r'[$]'
    }
for col, pattern in clean_data.items():
    df_2022[col] = pd.to_numeric(df_2022[col].replace(pattern, '', regex=True), errors='coerce')

# Here I'll generate summary statistics for Return Rate, IPO Price, and Current Price
# and create a dataframe for easier comparison and readability
cols = ['Return', 'IPO Price', 'Current']
summary = {}
for col in cols:
    data = df_2022[col].dropna().T
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

# Here, I'm calculating the percentage of companies that experienced negative returns in 2022
returns = df_2022['Return'].dropna()
neg_returns = (returns < 0).sum()
total_returns = returns.count()
perc_negative = (neg_returns / total_returns) * 100
print(total_returns)
print(round(perc_negative, 2))
# 56.14% of the 171 companies that offered prices experienced negative returns in 2022

# Here, I'll plot the stats of all three variables (Return Rate, IPO Price, and Current Price) on a histogram 
import seaborn as sns
import matplotlib.pyplot as plt
import os

output_location = r'c:\code_and_data\code_and_data\python\python_homework'

histogram_filepath = os.path.join(output_location, '2022_IPO_Distributions.png')

cols = ['Return', 'IPO Price', 'Current']
plt.figure(figsize=(18,6))
for i, col in enumerate(cols):
    plt.subplot(1, 3, 1+i)
    sns.histplot(df_2022[col].dropna(), bins=50, kde=False, color='forestgreen', edgecolor='black')
    
    skew = df_summary[col]['Skewness']
    kurt = df_summary[col]['Kurtosis']
    
    plt.title(f"2022 {col} Distribution \nSkewness {skew} \nKurtosis {kurt}")
    plt.xlabel(col)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(histogram_filepath)
plt.show()

# Summary Stat Overview:
    # Generated using valid integers within each Variable's column
        # Number of valid integers/companies are as follows:
            # Return: 171 , IPO Price: 180 , Current: 173
# Return Analysis:
    # Average return= -8.81% , median= -24%
        # This indicates that companies typically lost nearly a quarter of it's value.
    # Min= -99.63% , Max= 941%
        # Some companies were a complete loss, while a small number multiplied their valued up to 9 times.
    # Standard Deviation= 119.4
        # High volatility and unpredicatable
    # Skewness= 5.67 , Kurtosis= 40.48
        # As seen in the graph, these numbers mean the data includes outliers
        # that are distorting the average.
        # Most stocks are underperforming but the few exceptional companies are causing a right skew
# IPO Price Analysis:
    # Average price=$9.22, Median= $10
        #typical pricing
    # Standard deviation= 4.53, Skewness= 1.58 , Kurtosis= 4.12
        # The graph shows a slight right
        # Relatively stable compared to returns.
# Current Price Analysis: 
    #Average current price= $8.86 , Median= $6.51
        # Prices have dropped overall
    # Standard Deviation=11.94, Skewness=4.24, Kurtosis= 23.5
        # Same pattern as the Return variable
        # Small number of stocks performed very well while most fell below their IPO price 



# Here, I'm splitting the dataframe (df_2022) into 2 separate dataframes:
    # top thirty return rates (df_top_thirty) and the rest of the dataset (df_rest)
# Then i'll combine the results of each dataframe into another dataframe (df_stat_comp) for easier performance comparison
    # I want to determine the influence  positive outliers have on the rest of the dataset
df_top_thirty = df_2022.sort_values(by=['Return'], ascending=False).head(30)
df_rest = df_2022[~df_2022.index.isin(df_top_thirty.index)]

import scipy.stats as stats

def summary_stats(df):
    returns = df['Return'].dropna()
    return {
        'Number of Companies' : len(returns),
        'Average Return' : round(returns.mean(), 2), 
        'Median Return' : returns.median(),
        'Std_Dev' : round(returns.std(), 2),
        'Skewness' : round(returns.skew(), 2),
        'Kurtosis' : round(returns.kurtosis(), 2),
        'min' : returns.min(),
        'max' : returns.max()
        }
top_thirty_stats = summary_stats(df_top_thirty)
rest_stats = summary_stats(df_rest)   

df_stat_comp = pd.DataFrame ({
    'Top Thirty' : top_thirty_stats,
    'Rest of Dataset' : rest_stats
    })

# I'll plot the raw data for each dataframe (df_top_thirty, df_rest) on a boxplot 
# to visualize the different nature of Return rate distributions between the two datasets.
# First I'll create a copy of each dataframe's Return column, 
# and add a 'Group' column to label and organize the data for the boxplot
boxplot_filepath = os.path.join(output_location, '2022_Return_Comparison_Boxplot.png')

df_top_thirty = df_top_thirty.dropna(subset=['Return'].copy())
df_top_thirty['Group'] = 'Top 30'

df_rest = df_rest.dropna(subset=['Return'].copy())
df_rest['Group'] = 'Rest of Dataset'

df_combine = pd.concat([
    df_top_thirty[['Return','Group']],
    df_rest[['Return', 'Group']]
    ], ignore_index=True
    )
plt.figure(figsize=(8, 6))
sns.boxplot(data=df_combine, x='Group', y='Return', hue='Group', palette='pastel', legend=False)
plt.title('2022 Return Distribution: Top Thirty Companies vs Rest of Companies')
plt.xlabel('')
plt.ylabel('Return %')
plt.tight_layout()
plt.savefig(boxplot_filepath)
plt.show()


                        # Summary Stat Comparison of both Groups(Return Rate):
# Top Thirty IPOs:
    # Average return= 131.04%, Median= 60.14%
        # This group performed quite well and returns remained positive.
    # Min return= 18.5%, Max Return= 941%
    # Standard Deviation= 226.49
        # High volatility, but remained positive so no companies experience negative returns.
    # Skewness= 2.99, Kurtosis= 8.57
        # Implies a right skew
        # Few stocks drove up the average from exteme returns
        # but the whole group still performed exceptionally.
# Rest of The Dataset: 
    # Average return= -38.38%, Median= -47.20%
        # Most companies lost almost half of their value
    # Max= 18.5%
        # This equal to the Top Thirty Group's minimum return
    # Standard Deviation= 40.3
        # Low volatility compared to the Top Thirty Group, but the distribution is concentrated downward
    # Skewness= 0.09, Kurtosis= -1.5
        # Most negative returns are clustered towards hard losses and the group has no significant outliers
        