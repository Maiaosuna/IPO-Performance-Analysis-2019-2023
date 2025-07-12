# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 14:55:02 2025

@author: librarypc
"""
import pandas as pd
from pandas import DataFrame
from pandas import read_html
import numpy as np
import requests

data = 'https://stockanalysis.com/ipos/2019/'
url_2019 = requests.get(data)
ipos_2019 = read_html(url_2019.text)

# Main IPO data will be in the dataframe (df_2019)
df_2019 = ipos_2019[0]

# Start by cleaning the data and convert to numeric integers:
cols = ['Return', 'IPO Price', 'Current']
clean_data = {
    'Return' : r'[%]', 
    'IPO Price' : r'[$]', 
    'Current' : r'[$]'
    }
for col, pattern in clean_data.items():
    df_2019[col] = pd.to_numeric(df_2019[col].replace(pattern, '', regex=True), errors='coerce')
    
# Next, I'll calculate the summary stats for Return Rates, IPO Price, and Current Price
col = ['Return', 'IPO Price', 'Current']
summary = {}
for col in cols:
    data = df_2019[col].dropna()
    summary[col] = {
        'Number of companies' : len(data), 
        'Average' : round(data.mean(), 2), 
        'Median' : round(data.median(), 2), 
        'Std. Dev.' : round(data.std(), 2), 
        'Skewness' : round(data.skew(), 2), 
        'Kurtosis' : round(data.kurtosis(), 2), 
        'Min' : data.min(), 
        'Max' : data.max()
        }
df_summary = pd.DataFrame(summary)

# Here, I'll graph the data on a histogram to visualize the distribution of Return Rate, IPO Price, and Current Price side-by-side
import seaborn as sns
import matplotlib.pyplot as plt
import os

output_location = r'c:\code_and_data\code_and_data\python\python_homework'

histogram_filepath = os.path.join(output_location, '2019_IPO_Distributions.png')

cols = ['Return', 'IPO Price', 'Current']
plt.figure(figsize=(18,6))
for i, col in enumerate(cols):
    plt.subplot(1,3,i+1)
    sns.histplot(data=df_2019[col].dropna(), bins=50, kde=False, color='forestgreen', edgecolor='black')
    
    skew = df_summary[col]['Skewness']
    kurt = df_summary[col]['Kurtosis']
    
    plt.title(f"2019 {col} Distribution \nSkewness {skew} \nKurtosis {kurt}")
    plt.xlabel(col)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(histogram_filepath)
plt.show()

# Here, I'll check to see the percentage of companies that has negative returns for the year 2019.
returns = df_2019['Return'].dropna()       
neg_returns = (returns < 0).sum()
total_returns = returns.count()
perc_negative = (neg_returns / total_returns) * 100
print(neg_returns)
print(round(perc_negative, 2))
# There are 117 companies that experienced negative returns in 2019, 
# making up 54.67% of the companies that initially offered in this year.

                        # Summary Stat Analysis: 
# Generated using valid integers within each variable's column
        # Number of valid integers/companies are as follows:
            # Return: 214 , IPO Price: 225, Current: 225
# The mean return is 26.23%, but median return is negative (-7.79%),
    # meaning more than half of companies lost revenue since IPO.
    # The standard deviation for returns is 154.18, which tells that the dataset is quite volatile.
    # Skewness of 2.53 suggests that there is a small number of exceptional earners within the dataset that are dragging up the average.
    # Kurtosis of 8.0 shows a slightly heavy tail so outliers dominate the statistics.
    # Minimum return is -100%, which means some companies went to zero and were a complete loss for investors.
# IPO price skewness is 2.95, which is moderate but still has a right tail.
    # The mean price ($14.34) is greater than the median price ($13), 
    # while some IPOs were priced much higher causing high kurtosis (15.23)
    # Standard deviation of 7.84 is moderate compared to the mean of $14.34
# Current Price is the most extreme of all three.
    # The mean ($25.89) is more than double the median ($11.41), in which shows a significant skew on the graph.
    # Standard deviation is 487.11, which is quite large. While many companies are clusted,
    # some experienced major gains post-IPO.
    # Skewness (5.33) and kurtosis(35.62) are the result of extreme outliers within the data.
    # Some stocks are near zero (min= $0.07), while some have performed exceptionally.
# The summary statistics and the graph show the dataset to represent high risk and low predictability.




#Next, I'll separate the data into two dataframe:
    # Top Thirty Companies with the most returns (df_top_thirty), and the rest of the companies (rest_df)
# So I can compare summary stats for both in order to determine the influence the outliers have on the average return rate
df_top_thirty = df_2019.sort_values(by=['Return'], ascending=False).head(30)
df_rest = df_2019[~df_2019.index.isin(df_top_thirty.index)]

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
# Place both datasets into a dataframe for easier comparison
df_comparison = pd.DataFrame ({
    'Top Thirty' : top_thirty_stats,
    'Rest of Dataset' : rest_stats
    })
  
# Next, I'll combine the data from df_top_thirty and df_rest
# and plot them on a boxplot to visualize the nature of the different return rate distributions

boxplot_filepath = os.path.join(output_location, '2019_Return_Comparison_Boxplot.png')


df_top_thirty = df_top_thirty.dropna(subset=['Return'].copy())
df_top_thirty['Group'] = 'Top Thirty'

df_rest = df_rest.dropna(subset=['Return'].copy())
df_rest['Group'] = 'Rest of Dataset'

df_combine = pd.concat ([
     df_top_thirty[['Return', 'Group']],
    df_rest[['Return', 'Group']],
    ], ignore_index=True
    )

plt.figure(figsize=(8,6))
sns.boxplot(data=df_combine, x='Group', y='Return', hue='Group', palette='pastel', legend=False)
plt.title('2019 Return Distribution: Top Thirty Companies vs. Rest of The Companies')
plt.xlabel('')
plt.ylabel('Return %')
plt.tight_layout()
plt.savefig(boxplot_filepath)
plt.show()

                    #Summary Stat Comparison of Return Rates:
# Average and Median Returns:
# Top Thirty: Average=306.22%, Median=265.5%:
    # Exceptional results and signigicantly higher than the Rest Group
# Rest: Average= -21.05%, Median= -20.24%:
    # negative results suggest these companies lost value and underperformed since intitial offering

# Standard Deviation:
# Top Thirty: Std. Dev.= 173.63
# Rest: Std. Dev.= 59.45
    # To Top Thirty Groupd showed much higher returns but at the cost of higher volatility.
    # The Top Thirty Group is more prone to risk but with the probability of higher reward
    # The Rest Group is more tightly clustered around poor average.
    
# Skewness: 
# Top Thirty Group: Skewness= 1.15
# Rest: Skewness= 0.41
    # Both distributions are right skewed, but the Top Thirty Group is further right
    # which means there are fewer outliers as all the companies in this group exceptional performers

# Kurtosis: 
# Top Thirty Group: Kurtosis= 1.04
# Rest: Kurtosis= -0.9
    # Top Thirty Group kurtosis indicates more exteme outcomes
    # The Rest Group has a more normal distribution and fewer outliers that are concetrated towards loss.
    
