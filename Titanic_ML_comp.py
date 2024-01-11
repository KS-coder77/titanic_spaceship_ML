#!/usr/bin/env python
# coding: utf-8

# In[103]:


#import libs

import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle


# Task of competition is to predict whether a passenger was transported to an alternate dimension during the Spaceship Titanic's collision with the spacetime anomaly. 
# 
# Training data has been provided in the file train.csv - Personal records for about two-thirds (~8700) of the passengers, to be used as training data.
# 
# Testing data has been provided in the file test.csv - Personal records for the remaining one-third (~4300) of the passengers, to be used as test data. 
# 
# Your task is to predict the value of Transported for the passengers in this set.

# In[129]:


train_data = pd.read_csv(r'C:\Users\krupa\OneDrive\Desktop\Titanic_ship_ML\data\train.csv')

train_data.head()


# In[105]:


train_data.columns


# In[106]:


train_data.shape


# In[107]:


train_data.describe()


# In[108]:


#check for missing values
print('missing values (%) per column: \n', 100*train_data.isnull().mean())


# In[134]:


#fill the rows with missing values
test_data= test_data.dropna()
test_data


# Columns with integer values include:-
# -Age
# -Room Service (amount billed)
# -Food Court (amount billed)
# -Shopping mall (amount billed)
# -Spa (amount billed)
# -VR Deck (amount billed)
# 

# Non-integer value columns include:-
# -PassengerId 
# -HomePlanet
# -CryoSleep
# -Cabin 
# -Destination
# -VIP
# -Name
# -Transported
# 

# Let's take a look at each non-integer column in more detail. 

# In[110]:


passid=train_data['PassengerId'].nunique()
passid


# In[111]:


hplan=train_data['HomePlanet'].unique()
hplan


# In[142]:


#as there are only three unique values for the home planet column, we can replace them with integer values for ease.

mapping_dict = {'Europa': 1, 'Earth': 2, 'Mars': 3}

train_data['HomePlanet'] = train_data['HomePlanet'].map(mapping_dict)


# In[143]:


train_data['HomePlanet'].head(10)


# In[144]:


train_data['CryoSleep'].unique()


# In[145]:


train_data.dtypes


# In[137]:


train_data.dropna()


# In[147]:


#mapping_dict2 = {'True': 1, 'False': 2}

train_data['CryoSleep'] = train_data['CryoSleep'].fillna(0).astype(int)


# In[148]:


train_data['CryoSleep']


# In[116]:


train_data['Cabin'].nunique()


# In[117]:


train_data['Destination'].unique()


# In[118]:


mapping_dict3 = {'TRAPPIST-1e': 1, 'PSO J318.5-22': 2, '55 Cancri e': 3}


train_data['Destination'] = train_data['Destination'].map(mapping_dict3)


# In[119]:


train_data['Destination']


# In[150]:


train_data['VIP'].unique()


# In[ ]:





# In[151]:


#mapping_dict4 = {'True': 1, 'False': 2}

train_data['VIP'] = train_data['VIP'].fillna(0).astype(int)


# In[152]:


train_data['VIP']


# In[153]:


train_data.loc[train_data['VIP']==1]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# The next question is, which data is important and will have an influence on our final prediction (attribute- Transported). 

# In[ ]:





# In[124]:


predict = 'Transported'


# In[ ]:





# In[125]:


test_data = pd.read_csv(r"C:\Users\krupa\OneDrive\Desktop\Titanic_ship_ML\data\test.csv")

test_data.head()


# In[126]:


test_data.columns

