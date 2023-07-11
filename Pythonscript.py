#!/usr/bin/env python
# coding: utf-8

# # Importing the dataset

# In[36]:

import sys
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
print(sys.argv)

print(sys.argv[1])

# In[37]:


filename = sys.argv[1]


# In[38]:


dataset = pd.read_csv(filename)


# In[39]:


dataset.describe()
dataset


# # Fitting Linear Regression to the Dataset

# In[40]:


model = LinearRegression()
model.fit(dataset[['x']], dataset[['y']])


# # Adjusted R-squared




model.score(dataset[['x']], dataset[['y']])


# # Visualizing the Linear Regression results


plt.scatter(dataset[['x']], dataset[['y']], color = 'red')
plt.plot(dataset[['x']], model.predict(dataset[['x']]), color = 'blue')
plt.title('y vs x')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('pythoncombine.png')

plt.scatter(dataset[['x']], dataset[['y']], color = 'red')
plt.title('y vs x')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('pythonog.png')




