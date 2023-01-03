#!/usr/bin/env python
# coding: utf-8

# # Case Study #2

# ## 1. Case Study description

# For each year we need the following information:
# * Total revenue for the current year
# * Total Customers Current Year / Total Customers Previous Year
# * New Customers
# * New Customer Revenue e.g. new customers not present in previous year only
# * Existing Customer Revenue Current Year / Existing Customer Revenue Prior Year
# * Existing Customer Growth. To calculate this, use the Revenue of existing customers for current year –(minus) Revenue of existing customers from the previous year
# * Revenue lost from attrition
# * Lost Customers
# 
# Additionally, generate a few unique plots highlighting some information from the dataset. 
# Are there any interesting observations?
# 

# ## 2. Dataset

# A csv file with 3 years worth of customer orders. 
# There are 4 columns in the csv dataset: 
# * index, 
# * CUSTOMER_EMAIL(unique identifier as hash), 
# * Net_Revenue, and 
# * Year.
# 

# ## 3. Analysis

# ### 3.0 Import libraries

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# In[2]:


filename = 'casestudy.csv'
df = pd.read_csv(filename, sep=',')


# In[3]:


df.head()


# In[4]:


df.shape


# ### 3.1 Total revenue for the current/previous year

# In[5]:


all_years = df['year'].unique()


# In[6]:


total_net_revenue = np.empty([len(all_years)], dtype=np.float64)
for index, current_year in enumerate(all_years):
    total_net_revenue[index] = df['net_revenue'][df['year']==current_year].sum()
    print(f"Current year is {current_year}.")
    print(f"Current net_revenue is {total_net_revenue[index]:.2f}.")
    print()


# ### 3.2 Total Customers

# In[7]:


customers_per_year = {}
for index, current_year in enumerate(all_years):
    customers_per_year[current_year] = set(df['customer_email'][df['year']==current_year])
    
total = 0
for index, (key, value) in enumerate(customers_per_year.items()):
    total += len(value)
    print(f"In current year={key}, there are {len(value)} customers.")
print(f"Total customers: {total}. Total revenue: {total_net_revenue.sum()}")


# ### 3.3 New Customers / New Customer Revenue

# In[8]:


new_customer_net_revenue = np.empty([len(all_years)], dtype=np.float64)

new_customers_per_year = {}
for index, current_year in enumerate(all_years):
    if current_year-1 != all_years[index-1]:
        new_customers_per_year[current_year] = list(customers_per_year[current_year])
    else:
        new_customers_per_year[current_year] = list(customers_per_year[current_year]-customers_per_year[current_year-1])
        
    new_customer_net_revenue[index] = df['net_revenue'][df['year']==current_year][df['customer_email'].isin(new_customers_per_year[current_year])].sum()


# In[9]:


total = 0
for index, (key, value) in enumerate(new_customers_per_year.items()):
    total += len(value)
    print(f"In current year={key}, there were {len(value)} new customers. Current new customer revenue: {new_customer_net_revenue[index]:.2f}")
print(f"Total new customers: {total}. Total new revenue: {new_customer_net_revenue.sum()}")


# ### 3.3 Existing Customer Revenue Current/Prior Year

# In[10]:


existing_net_revenue = total_net_revenue - new_customer_net_revenue
for index, current_year in enumerate(all_years):
    print(f"Current year is {current_year}.")
    print(f"Current existing customer net revenue is {existing_net_revenue[index]:.2f}.")
    print()


# ### 3.4 Existing Customer Growth

# Existing Customer Growth.  
# To calculate this, use the Revenue of existing customers for current year –(minus) Revenue of existing customers from the previous year

# In[11]:


existing_customer_growth = np.empty([len(all_years)], dtype=np.float64)

for index, current_year in enumerate(all_years):
    if current_year-1 != all_years[index-1]:
        existing_customer_growth[index] = 0.0
    else:
        existing_customer_growth[index] = existing_net_revenue[index] - existing_net_revenue[index-1]


# In[12]:


for index, current_year in enumerate(all_years):
    print(f"Current year is {current_year}.")
    print(f"Current customer growth is {existing_customer_growth[index]:.2f}.")
    print()


# ### 3.5 Lost Customers

# In[13]:


lost_customers = {}
for index, current_year in enumerate(all_years):
    if current_year-1 != all_years[index-1]:
        lost_customers[current_year] = []
    else:
        lost_customers[current_year] = list(customers_per_year[current_year-1] - customers_per_year[current_year])

total = 0
for index, (key, value) in enumerate(lost_customers.items()):
    total += len(value)
    print(f"In current year={key}, there are {len(value)} customers.")


# ### 3.6 Revenue lost from attrition

# In[14]:


revenue_lost_attrition = np.empty([len(all_years)], dtype=np.float64)
for index, current_year in enumerate(all_years):
    if current_year-1 != all_years[index-1]:
        revenue_lost_attrition[index] = 0.0
    else:
        revenue_lost_attrition[index] = df['net_revenue'][df['year']==current_year-1][df['customer_email'].isin(lost_customers[current_year])].sum()


# In[15]:


for index, current_year in enumerate(all_years):
    print(f"Current year is {current_year}.")
    print(f"Current revenue lost from attrition is {revenue_lost_attrition[index]:.2f}.")
    print()


# ## 4. Observations

# In[16]:


df.head(n=10)


# * Limited years
# * The higher revenue comes from new customers. Not for the existing customers.
# * The customers doesn't buy for muliple years.
# * The company must find a way to attract new customers.
# * In the last year, the revenue from new customers is higher then the revenue lost form attrition.

# In[18]:


df.groupby(['customer_email', 'net_revenue'])['net_revenue'].transform('sum').mean()


# In[19]:


df.groupby(['customer_email', 'net_revenue','year'])['net_revenue'].transform('sum').mean()


# In[20]:


df.groupby(['customer_email', 'year'])['year'].transform('count').sum()


# In[21]:


plt.plot(all_years, total_net_revenue, label='Total net revenue')
plt.plot(all_years, new_customer_net_revenue, label='New customer net revenue')
plt.plot(all_years, existing_net_revenue, label='Existing net revenue')
plt.plot(all_years, -1*revenue_lost_attrition, label='Lost net revenue')
plt.xticks(all_years)
plt.legend(loc='lower left')
plt.show()


# In[22]:


total_lost = []
total_ll = []
total_new = []
for index, (key, value) in enumerate(lost_customers.items()):
    total_lost.append(len(value))
    total_new.append(len(new_customers_per_year[key]))
    total_ll.append(len(customers_per_year[key]))
    
plt.plot(all_years, total_ll, label='Total Customers')
plt.plot(all_years, total_new, label='New customers')
plt.plot(all_years, total_lost, label='Lost Customers')
plt.xticks(all_years)
plt.legend(loc='lower right')
plt.show()


# In[23]:


plt.plot(all_years, existing_customer_growth, label='Lost net revenue')
plt.xticks(all_years)
plt.legend(loc='lower left')
plt.show()


# In[ ]:




