#!/usr/bin/env python
# coding: utf-8

# # IMPORT LIBRARIES

# In[76]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# # READING DATA

# In[77]:


data=pd.read_excel("Data_train.xlsx")
data.head()


# In[78]:


data.describe()


# In[79]:


data.info()


# # DATA WRANGLING

# In[80]:


data.isnull().sum()


# In[81]:


sns.heatmap(data.isnull())


# # DEALING CATEGORICAL VARIABLE

# In[82]:


from sklearn.preprocessing import LabelEncoder


# In[83]:


data['Restaurant']=LabelEncoder().fit_transform(data['Restaurant'])
data['Location']=LabelEncoder().fit_transform(data['Location'])
data['Cuisines']=LabelEncoder().fit_transform(data['Cuisines'])
data['Average_Cost']=pd.to_numeric(data['Average_Cost'].str.replace('[^0-9]',''))
data['Minimum_Order']=pd.to_numeric(data['Minimum_Order'].str.replace('[^0-9]',''))
data['Rating']=pd.to_numeric(data['Rating'].apply(lambda x: np.nan if x in ['Temporarily Closed','Opening Soon','-','NEW'] else x))
data['Votes']=pd.to_numeric(data['Votes'].apply(lambda x: np.nan if x=='-' else x))
data['Reviews']=pd.to_numeric(data['Reviews'].apply(lambda x: np.nan if x=='-' else x))
data['Delivery_Time']=pd.to_numeric(data['Delivery_Time'].str.replace('[^0-9]',''))


# In[84]:


data.isnull().sum()


# # DEALING WITH MISSING VALUE 

# In[85]:


data['Rating']=data['Rating'].fillna(data['Rating'].median())
data['Votes']=data['Votes'].fillna(data['Votes'].median())
data['Reviews']=data['Reviews'].fillna(data['Reviews'].median())
data['Average_Cost']=data['Average_Cost'].fillna(data['Average_Cost'].median())
data.head()


# In[86]:


cor = data.corr()

mask = np.zeros_like(cor)
mask[np.triu_indices_from(mask)] = True

plt.figure(figsize=(12,10))

with sns.axes_style("white"):
    sns.heatmap(cor,annot=True,linewidth=2,
                mask = mask,cmap="magma")
plt.title("Correlation between variables")
plt.show()


# In[87]:


plt.figure(figsize=(12,8))
sns.heatmap(round(data.describe()[1:].transpose(),2),linewidth=2,annot=True,fmt="f")
plt.xticks(fontsize=20)
plt.yticks(fontsize=12)
plt.title("Variables summary")
plt.show()


# # TRAIN AND TEST DATA

# In[88]:


y_train=data.Delivery_Time
predictors_col=['Average_Cost','Minimum_Order','Rating','Votes','Reviews']
X_train=data[predictors_col]


# In[89]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.2, random_state = 0)


# In[90]:


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, y_train)


# In[91]:


y_pred = regressor.predict(X_test)


# In[92]:


regressor.score(X_train,y_train)*100


# In[93]:


print(y_pred)


# In[ ]:




