#!/usr/bin/env python
# coding: utf-8

# # Importing libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# # Importing the dataset

# In[2]:


df=pd.read_csv('RBC dataset.csv')
x=df.iloc[:,:-1].values
y=df.iloc[:,1].values


# In[3]:


print(x)


# In[4]:


print(y)


# # Training linear regression on whole dataset

# In[5]:


from sklearn.linear_model import LinearRegression
linear_reg=LinearRegression()
linear_reg.fit(x,y)


# # Training the polynomial regression model on whole dataset

# In[9]:


from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
x_poly=poly_reg.fit_transform(x)


# In[11]:


linear_reg2=LinearRegression()
linear_reg2.fit(x_poly,y)


# # Visualising linear regression results

# In[12]:


plt.scatter(x,y, color='red')
plt.plot(x,linear_reg.predict(x),color='blue')
plt.title("RBCs with linear regression")
plt.xlabel("Age")
plt.ylabel("RBC")
plt.show()


# # Visualising the ploynomial regression results

# In[13]:


plt.scatter(x,y,color='red')
plt.plot(x,linear_reg2.predict(x_poly),color='blue')
plt.title("RBC with polynomial regression")
plt.xlabel("Age")
plt.ylabel("RBC")
plt.show()


# # Predicting a new result with linear regression

# In[14]:


linear_reg.predict([[6.5]])


# # Predicting a new result with polynomial regression

# In[16]:


linear_reg2.predict(poly_reg.fit_transform([[6.5]]))


# In[ ]:




