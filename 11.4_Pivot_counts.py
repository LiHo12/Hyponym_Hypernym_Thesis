#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os


# In[15]:


# get string representation of folder
folder = '/path/to/9_FINAL/data/machine_learning/two_class/distance/grouping/Run_3/validation_fold_0/count_sum_'


# In[2]:


# show files in folder
# os.listdir(folder)


# In[18]:


all_data = pd.DataFrame()

for i in range(1,63): # TODO: change number of folders
    # get data for one file
    raw_data = pd.read_csv(folder+ str(i) + '.csv', sep=";")
    
    # concat data
    all_data = pd.concat([all_data, raw_data],
                        ignore_index=True)
    
# pivot data
sums = all_data[['id','pids', 'sum']]

# pivot table
sums = pd.pivot_table(sums, index='id', columns='pids', values='sum', aggfunc='first').fillna(0).reset_index()
sums.to_csv('/path/to/9_FINAL/data/machine_learning/two_class/distance/grouping/Run_3/validation_fold_0/val_fold_0_group_sum.csv',
            sep=";", 
           ignore_index=0)


# In[13]:


# get mean
all_data['mean_count'] = all_data['sum'] /all_data['count']

# pivot data
means = all_data[['id','pids', 'mean_count']]

# pivot table
means = pd.pivot_table(means, index='id', columns='pids', values='mean_count', aggfunc='first').fillna(0).reset_index()
means.to_csv('/media/linda/INTENSO/9_FINAL/data/machine_learning/two_class/distance/grouping/Run_3/validation_fold_0/val_fold_0_group_mean.csv', 
             sep=";",
            ignore_index=0)


# In[ ]:




