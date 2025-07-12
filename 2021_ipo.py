# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 23:05:52 2025

@author: librarypc
"""

import pandas as pd
from pandas import read_html
import numpy as np
import requests

data = 'https://stockanalysis.com/ipos/2021/'
url_2021 = requests.get(data) 
ipos_2021 = read_html(url_2021.text)
df_2021 = ipos_2021[0]

# Clean and convert data into numeric integers
clean_data = {
'Return' : r'[%]',
'IPO Price' :r'[$]',
'Current' : r'[$]'
}

for col, pattern in clean_data.items():
    df_2021[col] = pd.to_numeric(df_2021[col].replace(pattern, '', regex=True), errors='coerce').dropna()

# Generate summary stats using data from df_2021 for Return Rate, IPO Price, and Current Price 
cols = ['Return', 'IPO Price', 'Current']
summary = {}

for col in cols:
    data = df_2021[col].dropna().T
    summary[col] = {
        'Number of Companies' : len(data),
        'Average' : round(data.mean(), 2),
        'Median' : data.median(), 
        'Std. Dev.' : round(data.std(), 2), 
        'Skewness' : round(data.skew(), 2), 
        'Kurtosis' : round(data.kurtosis(), 2),
        'Min' : data.min(), 
        'Max' : data.max()
        }
summary_df = pd.DataFrame(summary)
print(summary_df)

# Here, I'm graphing that data on a histogtam to visualize the distribution of Return Rate, IPO Price, and Current Price
import matplotlib.pyplot as plt
import seaborn as sns
import os

output_location = r'c:\code_and_data\code_and_data\python\python_homework'

histogram_filepath = os.path.join(output_location, '2021_IPO_Distributions.png')

cols = ['Return', 'IPO Price', 'Current']
plt.figure(figsize=(18,5))
for i, col in enumerate(cols):
    plt.subplot(1, 3, i+1)
    sns.histplot(df_2021[col].dropna(), bins=50, kde=False, color='navy', edgecolor='black')

    skew = summary_df[col]['Skewness']
    kurt = summary_df[col]['Kurtosis']  
    
    plt.title(f"2021{col} Distribution \nskew {skew} \nkurtosis {kurt}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(histogram_filepath)
plt.show()

# Here, I'm calculating the percentage of companies that had negative returns in 2021
returns = df_2021['Return'].dropna()
neg_returns = (returns < 0).sum()
total_returns = returns.count()
perc_negative = (neg_returns / total_returns) * 100
print(total_returns)
print(round(perc_negative, 2))
# 46.64% of the 982 companies that offered in 2021, experienced negative returns.

                            # Summary Stat Analysis:
# Generated using valid integers within each variable's column
     # The number of valid intergers/companies are as follows:
         # Return: 982 , IPO Price: 1022 , Current: 1000
# Return Analysis: 
    #Average return = -11.96% , median= return = 0.6%
        # More than half the market lost value.
        # Which means the majority of companies barely moved from thier IPO price, or dropped below.
    #Skewness = 4.92 , Kurtosis = 64.87 
        #which are noticed on the graph as heavy right skew.
        # This shows that a very small amount of IPOs returned exceptionally (up to 900%), 
        # dragging up the average and distorting the performance of the rest of the dataset
# IPO Price Analysis: 
    # Average price = $13.74 , median price = $10.
        # Data shows that most IPOs are priced between $10-$15, 
        # but the average is influenced by outliers.
    # Skewness = 9.17 , Kurtosis = 144.01
        # The outliers have drastically distrorted the summary stats and created a strong right skew and heavy tail.
        # Some IPOs priced as high as $250 while others are as low as $3.
# Current Price Analysis:
    # Average current price = $12.34 , median = $10.15
        # The current price has barely moved from the IPO price
        # Considering the negative returns, the data shows that most stocks didn't make any improvement since IPO.
    # Skewness = 11.75 , kurtosis = 159.93
        # Again, this is the result of exceptional outliers.
        # Very few high performing stocks while the majority of the stocks barely changed.


# Here, I'll split the data separating top thirty earners (df_top_thirty) from the rest of the dataset (df_rest)
# And calculate the indidual summary stats for return rates
# Then combine both dictionaries into a dataframe for easier comparison
df_top_thirty = df_2021.sort_values(by=['Return'], ascending=False).head(30)
df_rest = df_2021[~df_2021.index.isin(df_top_thirty.index)]         # [~df_2021.index.isnin...] to grab all data opposite of top thirty data

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

# Here I'll plot the two datasets (df_top_thirty, df_rest) on a boxplot to visualize the nature of the different return rate distributions between them:
import pandas as pd 
 
boxplot_filepath = os.path.join(output_location, '2021_Return_Comparison_Boxplot.png')  

df_top_thirty = df_top_thirty.dropna(subset=['Return'].copy())
df_top_thirty['Group'] = 'Top Thirty'
df_rest = df_rest.dropna(subset=['Return'].copy())
df_rest['Group'] = 'Rest of Dataset'

df_combined = pd.concat ([
    df_top_thirty[['Return', 'Group']],
    df_rest[['Return', 'Group']]
    ], ignore_index=True
    )

plt.figure(figsize=(8, 6))
sns.boxplot(data=df_combined, x ='Group', y='Return', hue='Group', palette='pastel', legend=False)
plt.title('2021 Return Distribution: Top 30 vs. Rest of 2020 IPOs', fontsize=14)
plt.ylabel('Return (%)')
plt.xlabel('')
plt.tight_layout()
plt.savefig(boxplot_filepath)
plt.show()

            # Summary Stat Comparison Analysis (Return Rate):
# Top Thirty IPOs:  
    # Average return = 200.94%, Median Return = 149.63%
    #Minimum return = 101.53%
        # Every company nearly doubled in value
    #Standard Deviation = 158.38
        # High volatility, but each company was distributed positively
    # Skewness = 3.97 , Kurtosis = 18.27
        # The graph and data show that there were a few high earners,
# Rest of Dataset:
    # Average return = -18.67% , Median return = 0.4%
        # This group lost value for investors.
        # Most IPOs either lost value or didn't move.
    # Standard Deviation = 40.88
        # Lower volatility than the Top Thirty group
            # but still had risk due to underperforming stocks.
    # Skewness = -0.39 , Kurtosis = -0.46
        # Most returns are clumped closer to zero or negative.
    # Highest return = 94.32%
        # Which is stil far less than the lowest return than the Top Thirty group (101.53%)
        























