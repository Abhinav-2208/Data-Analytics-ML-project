#!/usr/bin/env python
# coding: utf-8

# # Importing the libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# # importing the dataset

# In[10]:


df=pd.read_csv("salary simple linear regression.csv")
x=df.iloc[: ,:-1].values
y=df.iloc[:,-1].values


# # Extracting the dependent and independent variable and storing in x and y for further use

# In[11]:


print(x)


# In[12]:


print(y)


# # Splitting the dataset into training and testing set

# In[13]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.3,random_state=42)


# In[14]:


print(x_train)


# In[15]:


print(x_test)


# In[16]:


print(y_train)


# In[17]:


print(y_test)


# # Training on training set using simple linear regression

# In[18]:


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)


# # Predicting on test set

# In[21]:


y_pred=regressor.predict(x_test)
x_pred=regressor.predict(x_train)


# # Visualising the training set

# In[23]:


plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("CTC vs Salary experience")
plt.xlabel("Experience in years")          
plt.ylabel("Salary")
plt.show()


# # Visualising the test set

# In[24]:


plt.scatter(x_test,y_test, color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("CTC vs Experience in year")
plt.xlabel("Exp in years")
plt.ylabel("CTC")
plt.show()


# In[ ]:




