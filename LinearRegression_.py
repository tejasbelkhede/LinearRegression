#!/usr/bin/env python
# coding: utf-8

# In[43]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns 


# In[44]:


df=pd.read_csv("auto-mpg.csv")


# In[45]:


df.head()


# In[46]:


df.describe()


# In[47]:


df.dtypes


# In[48]:


df['horsepower']=df['horsepower'].replace(to_replace='?', value=np.nan)


# In[49]:


df['horsepower']=df['horsepower'].astype(float)


# In[50]:


df.head()


# In[51]:


df.dtypes


# In[52]:


median1=df['horsepower'].median()


# In[53]:


median1


# In[54]:


df["horsepower"]=df["horsepower"].replace(np.nan,median1)


# In[55]:


df.isnull().sum()


# In[56]:


duplicate=df.duplicated()


# In[57]:


print(duplicate.sum())


# In[58]:


sns.boxplot(x='horsepower',data=df)


# In[59]:


df.isnull().sum()


# In[60]:


df=df.drop(["car name"],axis=1)


# In[61]:


df.head()


# In[62]:


df=pd.get_dummies(df,columns=["origin"])


# In[63]:


df.sample(10)


# # model

# In[64]:


from sklearn.mode1_selection import train_test_split
from sklearn.linear_mode1 import LinearRegression


# In[ ]:


y=df[["mpg"]]
x=df.drop(["mpg"],axis=1)


# In[ ]:


x_train,x_test,y_test=train_test_split(x,y,test_size=0.30,random_state=1)


# In[ ]:


mode1_lr=LinearRegression()


# In[ ]:


mode1_lr.fit(x_train,y_train)


# In[ ]:


mode1_lr.score(x_train,y_train)


# In[ ]:


mode1_lr.score(x_test,y_test)

