#!/usr/bin/env python
# coding: utf-8

# In[2]:


import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[38]:


hm=pd.read_csv("haberman.csv")
hm


# In[7]:


hm.shape


# In[9]:


hm.columns


# In[39]:


hm.columns=["age","year","axil","survive"]
hm.head()


# In[12]:


hm["survive"].value_counts()


# In[19]:


hm.plot(kind='scatter',x='axil', y='age') 

plt.show()


# In[22]:


sns.set_style("whitegrid")
sns.FacetGrid(hm, hue="survive", size=5)    .map(plt.scatter, "age", "axil")    .add_legend()
plt.show()


# In[24]:


plt.close();
sns.set_style("whitegrid");
sns.pairplot(hm, hue="survive", size=3);
plt.show()


# In[53]:


import numpy as np
surv_ived=hm.loc[hm["survive"]==1]
Non_survived= hm.loc[hm["survive"]==2];


plt.plot(surv_ived["age"], np.zeros_like(surv_ived["age"]),'o')
plt.plot(Non_survived["age"], np.zeros_like(Non_survived["age"]),'o')
#plt.plot(iris_versicolor["petal_length"], np.zeros_like(iris_versicolor['petal_length']), 'o')
#plt.plot(iris_virginica["petal_length"], np.zeros_like(iris_virginica['petal_length']), 'o')


# In[61]:


sns.FacetGrid(hm, hue="survive", size=5)    .map(sns.distplot, "axil")    .add_legend();
plt.show();


# In[ ]:




