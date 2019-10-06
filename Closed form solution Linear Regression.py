#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression


# In[44]:


# Generate DATA set
X,Y=make_regression(n_samples=400,n_features=1,n_informative=1,noise=1.8,random_state=11)


# In[45]:


print(X.shape,Y.shape)


# In[46]:


Y=Y.reshape((-1,1))
print(Y.shape)


# In[47]:


X=(np.mean(X)-X)/np.std(X)


# In[48]:


plt.plot(X,Y)

plt.scatter(X,Y,color='orange')


# In[49]:


plt.scatter(X,Y,color='orange')


# In[50]:


X_one=np.ones((X.shape[0],1))
X_one.shape


# In[84]:


X_br=np.hstack((X,X_one))
X_br.shape


# In[87]:


P=X_br.T
P.shape
X_br.shape


# In[98]:


def predict(X,theta):
    return np.dot(X,theta)

def CFS(X,Y):
    Y=np.mat(Y)
    theta=np.linalg.pinv(np.dot(X.T,X))*np.dot(X.T,Y)
    return theta


# In[99]:


T=CFS(X_br,Y)
print(T)


# In[102]:


Pred=predict(X_br,T)


# In[105]:


plt.scatter(X,Y)
plt.plot(X,Pred,color='orange')


# In[ ]:




