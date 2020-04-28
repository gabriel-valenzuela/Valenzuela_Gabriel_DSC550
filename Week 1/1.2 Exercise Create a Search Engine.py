#!/usr/bin/env python
# coding: utf-8

# In[3]:


# File: 1.2 Exercise CreateASearchEngine.ipynb
# Name: Gabriel Valenzuela
# Date: 03/12/2020
# Course: DSC 550 - Data Mining
# Purpose: Creating a 2D array filled with random integers, followed by finding the mean, min, max, and standard
#           deviation. Next, it will sort on a row, on a column, on row 2, and on column 2
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import numpy as np
import pandas as pd

# 1
# Create an array with random integers
rows, cols = (10,50)
arr = [[np.random.randint(1,500) for i in range(cols)] for j in range(rows)]
# Load into DataFrame
df = pd.DataFrame(arr)
df_sum = pd.DataFrame(arr)
print("Randomized Data Frame")
print(df)
print("\n")

# 2
# Sum of each row, column, and all entries
df_sum.loc['Total',:] = df_sum.sum(axis=0)
df_sum.loc[:,'Total'] = df_sum.sum(axis=1)
print("Sum of each Row, Column, and all Values")
print(df_sum)

# 3
# Minimum of all entries
print("\nMinimum Value: " + str(df.values.min()))

# Maximum of all entries
print("Maximum Value: " + str(df.values.max()))

# Average of all entries
print("Average Value: " + str(df.values.mean()))

# Standard Deviation of all entries
print("Standard Deviation: " + str(df.values.std()) + "\n")

# 4
# Sort dataframe by rows
print("Sort by Row")
print(df.sort_values(by=0, axis=1))
print("\n")
# Sort dataframe by columns
print("Sort by Column")
print(df.sort_values(by=0, axis=0))
print("\n")
# Sort dataframe on row 2
print("Sort by Row 2")
print(df.sort_values(by=2, axis=1))
print("\n")
# Sort dataframe on column 2
print("Sort by Column 2")
print(df.sort_values(by=2, axis=0))

