#!/usr/bin/env python
# coding: utf-8

# # IMPORT LIBRARIES

# In[124]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# # READING DATA

# In[125]:


data=pd.read_excel("Data_train.xlsx")
data.head()


# In[126]:


data.describe()


# In[127]:


data.info()


# # DATA WRANGLING

# In[128]:


data.isnull().sum()


# In[129]:


sns.heatmap(data.isnull())


# # DEALING CATEGORICAL VARIABLE

# In[130]:


from sklearn.preprocessing import LabelEncoder


# In[131]:


data['Restaurant']=LabelEncoder().fit_transform(data['Restaurant'])
data['Location']=LabelEncoder().fit_transform(data['Location'])
data['Cuisines']=LabelEncoder().fit_transform(data['Cuisines'])
data['Average_Cost']=pd.to_numeric(data['Average_Cost'].str.replace('[^0-9]',''))
data['Minimum_Order']=pd.to_numeric(data['Minimum_Order'].str.replace('[^0-9]',''))
data['Rating']=pd.to_numeric(data['Rating'].apply(lambda x: np.nan if x in ['Temporarily Closed','Opening Soon','-','NEW'] else x))
data['Votes']=pd.to_numeric(data['Votes'].apply(lambda x: np.nan if x=='-' else x))
data['Reviews']=pd.to_numeric(data['Reviews'].apply(lambda x: np.nan if x=='-' else x))
data['Delivery_Time']=pd.to_numeric(data['Delivery_Time'].str.replace('[^0-9]',''))


# In[132]:


data.isnull().sum()


# # DEALING WITH MISSING VALUE 

# In[133]:


data['Rating']=data['Rating'].fillna(data['Rating'].median())
data['Votes']=data['Votes'].fillna(data['Votes'].median())
data['Reviews']=data['Reviews'].fillna(data['Reviews'].median())
data['Average_Cost']=data['Average_Cost'].fillna(data['Average_Cost'].median())
data.head()


# In[134]:


cor = data.corr()

mask = np.zeros_like(cor)
mask[np.triu_indices_from(mask)] = True

plt.figure(figsize=(12,10))

with sns.axes_style("white"):
    sns.heatmap(cor,annot=True,linewidth=2,
                mask = mask,cmap="magma")
plt.title("Correlation between variables")
plt.show()


# In[135]:


plt.figure(figsize=(12,8))
sns.heatmap(round(data.describe()[1:].transpose(),2),linewidth=2,annot=True,fmt="f")
plt.xticks(fontsize=20)
plt.yticks(fontsize=12)
plt.title("Variables summary")
plt.show()


# # TRAIN AND TEST DATA

# In[136]:


y_train=data.Delivery_Time
predictors_col=['Average_Cost','Minimum_Order','Rating','Votes','Reviews']
X_train=data[predictors_col]


# In[137]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.2, random_state = 0)


# In[138]:


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, y_train)


# In[139]:


y_pred = regressor.predict(X_test)


# In[140]:


regressor.score(X_train,y_train)*100


# In[141]:


print(y_pred)


# # TEST DATA 

# In[142]:


data1=pd.read_excel("Data_test.xlsx")
data1.head()


# In[143]:


# data preprocessing
from sklearn.preprocessing import LabelEncoder
data1['Cuisines']=LabelEncoder().fit_transform(data1['Cuisines'])
data1['Average_Cost']=pd.to_numeric(data1['Average_Cost'].str.replace('[^0-9]',''))
data1['Minimum_Order']=pd.to_numeric(data1['Minimum_Order'].str.replace('[^0-9]',''))
data1['Rating']=pd.to_numeric(data1['Rating'].apply(lambda x: np.nan if x in ['Temporarily Closed','Opening Soon','-','NEW'] else x))
data1['Votes']=pd.to_numeric(data1['Votes'].apply(lambda x: np.nan if x=='-' else x))
data1['Reviews']=pd.to_numeric(data1['Reviews'].apply(lambda x: np.nan if x=='-' else x))


# In[144]:


# fill missing value with meadian imputation
data1['Rating']=data['Rating'].fillna(data1['Rating'].median())
data1['Votes']=data['Votes'].fillna(data1['Votes'].median())
data1['Reviews']=data['Reviews'].fillna(data1['Reviews'].median())
data1['Average_Cost']=data1['Average_Cost'].fillna(data1['Average_Cost'].median())
data1.head()


# In[145]:


X_test=data1[predictors_col]
predictions=regressor.predict(X_test)


# In[146]:


print(predictions)


# In[147]:


out=predictions.astype(int)
out=out.astype(int).astype(str)
for i in range(len(out)):
    out[i]=out[i]+"minutes"


# In[148]:


submission=(pd.DataFrame({'Restaurant':data1.Restaurant,'Delivery_Time':out}))


# In[149]:


submission.to_csv('submission.csv',index=False)


# In[ ]:




