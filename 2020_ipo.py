# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 16:51:05 2025

@author: librarypc
"""
import numpy as np
import pandas as pd
from pandas import read_html
import requests


data_2020 = 'https://stockanalysis.com/ipos/2020/'
url_2020 = requests.get(data_2020)

ipos_2020 = read_html(url_2020.text)

# Extract dataframe of interest into dataframe called df_2020
df_2020 = ipos_2020[0]

# Clean data in df_2020 and convert into numeric integers
clean_data = {
    'Return' : r'[%]',
    'IPO Price' : r'[$]',
    'Current' : r'[$]'
    }
for col, pattern in clean_data.items():
    df_2020[col] = pd.to_numeric(df_2020[col].replace(pattern, '', regex=True), errors='coerce')

# Evaluate the summary statistics for Return Rate, IPO Price, and Current Price
import scipy.stats as stats

cols = ['Return', 'IPO Price', 'Current']
summary = {}
for col in cols:
    data = df_2020[col].dropna().T
    summary[col] = {
        'Number of Companies' : len(data),
        'Average' : round(data.mean(), 2),
        'median' : data.median(),
        'std_dev' : round(data.std(), 2), 
        'skewness' : round(data.skew(), 2), 
        'kurtosis' : round(data.kurtosis(), 2),
        'min' : data.min(),
        'max' : data.max()
        }
# Convert results into dataframe for clean output and easier readability
summary_df = pd.DataFrame(summary)

# Here I'll plot the data on a histogram to visualize the distibution of Return, IPO Price, and Current Price
import seaborn as sns
import matplotlib.pyplot as plt
import os
    
output_location = r'c:\code_and_data\code_and_data\python\python_homework'
histogram_filepath = os.path.join(output_location, '2020_IPO_Distributions.png')    

cols = ['Return', 'IPO Price', 'Current']
plt.figure(figsize=(18,5))
for i, col in enumerate(cols):
        plt.subplot(1, 3, i + 1)
        sns.histplot(df_2020[col].dropna(), bins=50, kde=False, color='skyblue', edgecolor='black')
        
        skew = summary_df[col]['skewness']
        kurt = summary_df[col]['kurtosis']
        
        plt.title(f"2020 {col} Distribution \nSKewness {skew} \nKurtosis {kurt}")
        plt.xlabel(col)
        plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(histogram_filepath)
plt.show()

# Next, I'll calculate the percentage of companies whose return rates were negative
returns = df_2020['Return'].dropna()
neg_returns = (returns < 0).sum()
total_returns = returns.count()
perc_negative = (neg_returns / total_returns) * 100
print(round(perc_negative, 2))
print(total_returns)
# 53.08% of the 454 companies within the dataset experienced negative returns in the year 2020.

                            # Summary Stat Analysis:
# Generated using valid integers within each variable's column
    # The number of valid integers/company are as follows:
        # Return: 454 , IPO Price: 477 , Current: 460
# Returns show underperformance. 
    # The average return (-4.85%) and median return (-2.1%) are both negative, proving more than half of the IPOs lost value.
    # Standard deviation (75.01) shows dispersion and suggests volatility.
    # Skewness (2.18) and kurtosis (9.26) provide a sign of heavy, right tail distribution, which can be seen in the graph.
        # A small number of companies performed exceptionally while most underperformed.
    # The highest return was 532.4%, while the lowest return was -100%.
# IPO Prices were highly skewed (skew=6.02) due to moderate prices of most IPOs with the exception of a small number of IPOs priced very high.
    # Kurtosis (53.45) is quite high which confirms there are rare, but extreme outliers.
    # The average price ($14.02) is greater than the median price ($10), meaning the outliers pulled up the average as the typical price should be around $10.
# Current Prices have a strong right tail with skewness at 6.4 and kurtosis at 54.23.
    # The median ($10.03) and average price (13.88) are similar to the IPO price which tells that there was little growth post-IPO.
    # However, the high current price was $238.24 which means some companies did quite well.
    # Althoug, many companies still lost value (min= $0.03).



# Next, i'll split the dataframe into two separate datasets:
    # Top Thirty return earners (df_top_thirty), and the rest of the companies (df_rest)
df_top_thirty = df_2020.sort_values(by='Return', ascending=False).head(30)
df_rest = df_2020[~df_2020.index.isin(df_top_thirty.index)]

import scipy.stats as stats

def summary_stats(df):
    returns = df['Return'].dropna()
    return {
        'Number of Companies' : len(returns),
        'Average Return' : returns.mean(), 
        'Median Return' : returns.median(),
        'Std_Dev' : round(returns.std(), 2),
        'Skewness' : round(returns.skew(), 2),
        'Kurtosis' : round(returns.kurtosis(), 2),
        'min' : returns.min(),
        'max' : returns.max()
        }
top_thirty_stats = summary_stats(df_top_thirty)
rest_stats = summary_stats(df_rest)
# Convert the summary stats into a dataframe for easier comparison
df_comparison = pd.DataFrame({
    'Top Thirty' : top_thirty_stats,
    'Rest of Dataset' : rest_stats
    })
print(df_comparison)


# Next, I'll make plot the data on a boxplot
# to visualize the nature of the different return rate distributions between the Top Thirty Group and Rest of Dataset Group
# First I'll label the dataset then combine into dataframe before creating boxplot

boxplot_filepath = os.path.join(output_location, '2020_Return_Comparison_Boxplot.png')

df_top_thirty = df_top_thirty.dropna(subset=['Return'].copy())         #.copy() so pandas knows this is an independent set
df_top_thirty['Group'] = 'Top 30'

df_rest = df_rest.dropna(subset=['Return'].copy())
df_rest['Group'] = 'Rest of Dataset'

df_combined = pd.concat([
    df_top_thirty[['Return', 'Group']],
    df_rest[['Return', 'Group']]
    ], ignore_index=True)

plt.figure(figsize=(8, 6))
sns.boxplot(data=df_combined, x='Group', y='Return', hue='Group', palette='pastel', legend=False)
plt.title('2020 Return Distribution: Top 30 vs. Rest of 2020 IPOs', fontsize=14)
plt.ylabel('Return (%)')
plt.xlabel('')
plt.savefig(boxplot_filepath)
plt.tight_layout()
plt.show()

# Summary Stat Comparison of Return Rates:
# The Top Thirty group showed exceptional performance.
    # The average return was 189.23% and the median return was 157.26%.
        # Meaning most companies in this group had high returns
    # The standard deviation was 95.92, so returns were quite dispersed
    # Skewness was 1.85, right tailed distribution due to a few large return rates
    # Kurtosis was 3.82, which is expected as this group consistently outperforms.
# The Rest of the dataset underperformed. 
    # The average return was -19.22% and the median was also negative at -5.15%.
        # Implying most IPOs lost value
    # Standard deviation for the Rest group was 47.57, as most returns were clustered on the lower end of the distribution.
    # Skewness was -0.01 and kurtosis was -0.53, which gives a more symmetric distribution since there are no outliers


