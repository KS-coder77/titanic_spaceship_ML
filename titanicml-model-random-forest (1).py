#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import libs
import matplotlib.pyplot as plt

import plotly as py
import plotly.express as px

import seaborn as sns

import pandas as pd
import numpy as np

import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle


# # Introduction

# ## Define the Problem ...
#  
# Task of competition is to predict using ML whether a passenger was transported to an alternate dimension during the Spaceship Titanic's collision with the spacetime anomaly. 
# 
# Training data has been provided in the file train.csv - Personal records for about two-thirds (~8700) of the passengers, to be used as training data.
# 
# Testing data has been provided in the file test.csv - Personal records for the remaining one-third (~4300) of the passengers, to be used as test data. 
# 
# Task is to predict the value of Transported for the passengers in the test dataset.

# ## Define the type of Problem ...
# 
# Given that we need to predict the total value of passengers transported, the output will be a category or label (i.e. transported (True) or not transported (False). Therefore, this is a classification type of problem to solve. 
# 
# Next steps ...
# 1. Binary classification, as there are only two possible outcomes, True or False
# 2. Are there any relationships between variables? Is there a decision boundary that seperates different classes? 
# 3. Evaluation metrics for classification include:- 
# * accuracy
# * precision
# * recall
# * F1-score 
# 

# In[2]:


#Gather and prepare training data (data is already split between training and test data)

file_path = "/kaggle/input/spaceship-titanic/train.csv"
train_data = pd.read_csv(file_path)
train_data.head()


# In[3]:


train_data.columns


# In[4]:


train_data.shape


# In[5]:


#generate descriptive statistics
train_data.describe()


# The max. figures for Room Service, Food Court, Shopping Mall, Spa and VR Deck are all total amounts of money spent in these areas and are in the thousands. Whereas, the max. age is 79 and that seems reasonable. Another point to note is that none of the Amenities have a meadian value (i.e. 50th percentile).
# 

# In[6]:


#Let's take a closer look at the spread of the age data 
plt.figure(figsize = (12,8))

sns.boxplot(data=train_data, y ='Age', showmeans=True)

mean_age=train_data['Age'].mean()
median_age=train_data['Age'].median()
plt.axhline(y=mean_age, color='r', linestyle='-')
plt.axhline(y=median_age, color='g', linestyle='-')

plt.title('Age Data')

plt.show()


# In[7]:


train_data.dtypes


# In[8]:


#Clean and pre-process data 
#check and handle missing values, outliers etc. 

print('missing values (%) per column: \n', 100*train_data.isnull().mean())


# In[9]:


#overall percentage of missing values 
total_cells = np.product(train_data.shape)
missing_vals = train_data.isnull().sum()
total_missing_vals = missing_vals.sum()
percent_missing_vals = ((total_missing_vals/total_cells)*100).round(2)
print('Overall percentage of missing values in dataset: ', percent_missing_vals,'%')


# There is a small percentage of missing values in 12 out of 14 of the columns. The only columns without missing data are PassengerId and Transported.
# 
# For the numerical columns we could impute the mean value:-
# * Age
# * RoomService
# * FoodCourt
# * ShoppingMall
# * Spa
# * VRDeck
# 

# In[10]:


#Let's fill the missing numerical values with mean values

mean_age=train_data['Age'].mean()
mean_RS=train_data['RoomService'].mean()
mean_FC=train_data['FoodCourt'].mean()
mean_SM=train_data['ShoppingMall'].mean()
mean_Spa=train_data['Spa'].mean()
mean_VRDeck=train_data['VRDeck'].mean()


train_data['Age'].fillna(mean_age, inplace=True)
train_data['RoomService'].fillna(mean_RS, inplace=True)
train_data['FoodCourt'].fillna(mean_FC, inplace=True)
train_data['ShoppingMall'].fillna(mean_SM, inplace=True)
train_data['Spa'].fillna(mean_Spa, inplace=True)
train_data['VRDeck'].fillna(mean_VRDeck, inplace=True)


# In[11]:


total_cells = np.product(train_data.shape)
missing_vals = train_data.isnull().sum()
total_missing_vals = missing_vals.sum()
updated_percent_missing_vals = ((total_missing_vals/total_cells)*100).round(2)


print('Percentage of missing values following mean value update',updated_percent_missing_vals , '%')


# Now we are down to **less than 1%** of missing values in the remaining columns! Let's take a look at the categorical data:-
# 
# * HomePlanet
# * CryoSleep
# * Cabin
# * Destination
# * VIP
# * Name 
# 
# 

# In[12]:


#how many values are missing in the category columns?

missing_hp_val=train_data['HomePlanet'].isnull().sum()
missing_cs_val=train_data['CryoSleep'].isnull().sum()
missing_cab_val=train_data['Cabin'].isnull().sum()
missing_dest_val=train_data['Destination'].isnull().sum()
missing_vip_val=train_data['VIP'].isnull().sum()
missing_name_val=train_data['Name'].isnull().sum()

print('Missing Home Planet Values:', missing_hp_val)
print('Missing CryoSleep Values:', missing_cs_val)
print('Missing Cabin Values:', missing_cab_val)
print('Missing Destination Values:', missing_dest_val)
print('Missing VIP Values:', missing_vip_val)
print('Missing Name Values:', missing_name_val)


# The missing values in the category column seem to hover around the 200-mark. Let's determine the mode value for each of the categorical data columns.
# 

# In[13]:


#mode method returns the most frequently occurring value and iloc[o] returns the first from a series
mode_hp=train_data['HomePlanet'].mode().iloc[0]
mode_cs=train_data['CryoSleep'].mode().iloc[0]
mode_cab=train_data['Cabin'].mode().iloc[0]
mode_dest=train_data['Destination'].mode().iloc[0]
mode_vip=train_data['VIP'].mode().iloc[0]
mode_name=train_data['Name'].mode().iloc[0]


print('Mode Home Planet Value:', mode_hp)
print('Mode CryoSleep Value:', mode_cs)
print('Mode Cabin Value:', mode_cab)
print('Mode Destination Value:', mode_dest)
print('Mode VIP Value:', mode_vip)
print('Mode Name Value:', mode_name)


# In[14]:


#Let's first fill the home planet values and destination values with the mode
train_data['HomePlanet'].fillna(mode_hp, inplace=True)
train_data['CryoSleep'].fillna(mode_cs, inplace=True)
train_data['Cabin'].fillna(mode_cab, inplace=True)
train_data['Destination'].fillna(mode_dest, inplace=True)
train_data['VIP'].fillna(mode_vip, inplace=True)
train_data['Name'].fillna(mode_name, inplace=True)


# In[15]:


train_data.shape


# In[16]:


total_cells = np.product(train_data.shape)
missing_vals = train_data.isnull().sum()
total_missing_vals = missing_vals.sum()
second_updated_percent_missing_vals = ((total_missing_vals/total_cells)*100).round(2)

print('Percentage of missing values following mode update',second_updated_percent_missing_vals , '%')


# In[17]:


print('missing values (%) per column: \n', 100*train_data.isnull().mean())


# Now the missing values have been dealt with, we can move on to the next step, data analysis,

# # Data Analysis

# In[18]:


#For better readability and data analysis let's split the cabin column into three separate columns for deck, number and side.

train_data[['Cabin Deck', 'Cabin Number', 'Cabin Side']] = train_data['Cabin'].str.split('/', expand=True)


# In[19]:


train_data.head()


# In[20]:


#let's take a closer look at cryo sleep and transported stats 

fig = px.histogram(train_data, x='CryoSleep', title = "CryoSleep Request Histogram", color= train_data['Transported'])
fig.show()


# In[21]:


#let's take a closer look at home planet and transported stats 

fig = px.histogram(train_data, x='HomePlanet', title = "Home Planet Histogram", color= train_data['Transported'])
fig.show()


# In[22]:


#let's take a closer look at cabin side and transported stats 

fig = px.histogram(train_data, x='Cabin Side', title = "Cabin Side (Port or Starboard) Histogram", color= train_data['Transported'])
fig.show()


# In[23]:


#let's take a closer look at deck level and transported stats 

fig = px.histogram(train_data, x='Cabin Deck', title = "Cabin Deck Histogram", color= train_data['Transported'])
fig.show()


# In[24]:


#let's take a closer look at passenger age and transported stats 

fig = px.histogram(train_data, x='Age', title = "Passenger Age Histogram", color= train_data['Transported'])
fig.show()


# In[25]:


#let's take a closer look at VIP and transported stats 

fig = px.histogram(train_data, x='VIP', title = "VIP Service Histogram", color= train_data['Transported'])
fig.show()


# In[26]:


#let's take a closer look at Destination and transported stats 

fig = px.histogram(train_data, x='Destination', title = "Destination Histogram", color= train_data['Transported'])
fig.show()


# From the graphs above, it is evident that the strongest correlation is between cryo sleep  and being transported. Also, a high proportion of passengers located on decks B, C and G were transported. Let's also plot a correlation graph to identify any other patterns. For ease, it could be useful to map categorical data values to integers. 
# 

# In[27]:


#map data in Home Planet, Destination and Cabin Deck columns 
mapping_dict1 = {'Earth':1, 'Europa':2, 'Mars':3}
train_data['HomePlanet'] = train_data['HomePlanet'].map(mapping_dict1)

mapping_dict2 = {'P':1, 'S':2}
train_data['Cabin Side'] = train_data['Cabin Side'].map(mapping_dict2)

mapping_dict3 = {'TRAPPIST-1e': 1, 'PSO J318.5-22': 2, '55 Cancri e': 3}
train_data['Destination'] = train_data['Destination'].map(mapping_dict3)

mapping_dict4 = {'A': 1, 'B': 2, 'C': 3, 'D':4, 'E':5, 'F':6, 'G':7, 'T':8}
train_data['Cabin Deck'] = train_data['Cabin Deck'].map(mapping_dict4)


# In[28]:


train_data.head()


# In[29]:


#let's drop some columns for the correlation graph
reduced_train_data=train_data.drop(columns=['PassengerId', 'Name', 'Cabin', 'Cabin Number'])
reduced_train_data


# In[30]:


reduced_train_data.dtypes


# In[31]:


reduced_train_data.describe(exclude='number')


# In[32]:


reduced_train_data.corr().style.background_gradient(cmap='Oranges')


# r-values greater than 0.7 indicate a strong correlation between two attributes. From the graph above, there doesn't appear to be any strong correlations. However, the graph does confirms what was found in the data analysis, CryoSleep and Transported are highly correlated, with an r-value of 0.46

# In[33]:


train_data.shape


# In[34]:


train_data.columns


# In[35]:


train_data.info()


# In[36]:


train_data.head()


# # Fitting the Model
# 
# As this is a classification problem, let's consider using the Random Forest Classifier ML model.

# In[37]:


from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score, confusion_matrix
from sklearn.metrics import accuracy_score

#define features and target variable
X = train_data[['HomePlanet', 'CryoSleep', 'Destination', 'Age',
       'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
       'Cabin Deck', 'Cabin Number', 'Cabin Side']]
y = train_data['Transported']

#split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[38]:


len(X_train)


# In[39]:


len(X_test)


# In[40]:


from sklearn.ensemble import RandomForestClassifier

#instantiate and fit the Random Forest Classifier 
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

print('score on test: ', end="")
print(str(model.score(X_test, y_test)))
print('score on train: ', end="")
print(str(model.score(X_train, y_train)))


# The accuracy score on the test data is ~79.21%. This is a reasonable score, let's move on to the competition test data.

# # ML Model Evaluation - Test Data

# In[41]:


#run model on test data 
test_data_filepath = "/kaggle/input/spaceship-titanic/test.csv"
test_data = pd.read_csv(test_data_filepath)

test_data.head()


# In[42]:


missing_test_vals = test_data.isnull().sum()
print(missing_test_vals)


# ### Handle missing values 

# In[43]:


#Let's fill the missing numerical values with mean values
mean_age=test_data['Age'].mean()
mean_RS=test_data['RoomService'].mean()
mean_FC=test_data['FoodCourt'].mean()
mean_SM=test_data['ShoppingMall'].mean()
mean_Spa=test_data['Spa'].mean()
mean_VRDeck=test_data['VRDeck'].mean()

test_data['Age'].fillna(mean_age, inplace=True)
test_data['RoomService'].fillna(mean_RS, inplace=True)
test_data['FoodCourt'].fillna(mean_FC, inplace=True)
test_data['ShoppingMall'].fillna(mean_SM, inplace=True)
test_data['Spa'].fillna(mean_Spa, inplace=True)
test_data['VRDeck'].fillna(mean_VRDeck, inplace=True)

#lets fill the missing categorical data with mode values
mode_hp=test_data['HomePlanet'].mode().iloc[0]
mode_cs=test_data['CryoSleep'].mode().iloc[0]
mode_cab=test_data['Cabin'].mode().iloc[0]
mode_dest=test_data['Destination'].mode().iloc[0]
mode_vip=test_data['VIP'].mode().iloc[0]
mode_name=test_data['Name'].mode().iloc[0]

#Let's fill the values with the mode
test_data['HomePlanet'].fillna(mode_hp, inplace=True)
test_data['CryoSleep'].fillna(mode_cs, inplace=True)
test_data['Cabin'].fillna(mode_cab, inplace=True)
test_data['Destination'].fillna(mode_dest, inplace=True)
test_data['VIP'].fillna(mode_vip, inplace=True)
test_data['Name'].fillna(mode_name, inplace=True)

test_data.shape


# In[44]:


print('missing values (%) per column: \n', 100*test_data.isnull().mean())


# ### Re-shape data to suit testing 

# In[45]:


#split cabin column into deck, number and side 
test_data[['Cabin Deck', 'Cabin Number', 'Cabin Side']] = test_data['Cabin'].str.split('/', expand=True)

#map data in Home Planet, Destination and Cabin Deck columns 
mapping_dict1 = {'Earth':1, 'Europa':2, 'Mars':3}
test_data['HomePlanet'] = test_data['HomePlanet'].map(mapping_dict1)

mapping_dict2 = {'P':1, 'S':2}
test_data['Cabin Side'] = test_data['Cabin Side'].map(mapping_dict2)

mapping_dict3 = {'TRAPPIST-1e': 1, 'PSO J318.5-22': 2, '55 Cancri e': 3}
test_data['Destination'] = test_data['Destination'].map(mapping_dict3)

mapping_dict4 = {'A': 1, 'B': 2, 'C': 3, 'D':4, 'E':5, 'F':6, 'G':7, 'T':8}
test_data['Cabin Deck'] = test_data['Cabin Deck'].map(mapping_dict4)


# In[46]:


test_data.head()


# In[47]:


test_data.columns


# In[48]:


X = test_data[['HomePlanet', 'CryoSleep', 'Destination', 'Age',
       'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
       'Cabin Deck', 'Cabin Number', 'Cabin Side']]
passenger_ids = test_data['PassengerId']

test_predictions = model.predict(X)
output_df = pd.DataFrame({'PassengerId' : passenger_ids, 'Transported' : test_predictions})
output_df['Transported'] = output_df['Transported'].astype(bool)
output_df.to_csv('titanic_output.csv', index=False)


# In[49]:


output_df.head(20)


# In[50]:


output_df['Transported'].value_counts()


# In[51]:


output_df.shape


# # Conclusion
# 
# To conclude, the overall accuracy of the ML model is ~79%. This is a good level of accuracy, however, improvements can always be made. 
# 
# To improve the performance of our ML model, we could carry out some of the following options: -  
# - The process of feature selection identifies the top relevant features from a larger set of features in a dataset. This also helps to avoid overfitting the model. 
# - Cross-validation exercise where you evaluate the performance of various models.
# - Vary the number of trees in the forest (i.e. n_estimators). The default value is 100, however this can be amended. 
# 
# Thank you. 
