#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


df=pd.read_csv("mushrooms.csv")
df.head()


# In[6]:


#The above is catogorical data
#ENCODE catogorical data into numerical data
#Label encoding

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# In[12]:


LE=LabelEncoder()
df_n=df.apply(LE.fit_transform)
df_n.head(n=5)
df_n.shape


# In[173]:


#break the data into train test
X=df_n.values[:,1:]
Y=df_n.values[:,0]
X


# In[44]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)


# In[45]:


print(X_train.shape,X_test.shape)


# In[121]:


#building our naive bayes classifier
def prior(Y_train,label):
    total_examples=Y_train.shape[0]
    class_examples=np.sum(Y_train==label)
    
    return class_examples/float(total_examples)

def cond_probab(X_train,Y_train,feature_col,feature_value,label):
    
    x_train=X_train[Y_train==label]
    numerator=np.sum(x_train[:,feature_col]==feature_value)
    denominator=np.sum(Y_train==label)
    
    return numerator/float(denominator)

def predict(X_train,Y_train,X_test):
    
    classes=np.unique(Y_train)
    n_features=X_train.shape[1]
   
    post_probs=[]
    
    for label in classes:
        likelihood=1.0
        for f in range(n_features):
            cond_prob=cond_probab(X_train,Y_train,f,X_test[f],label)
            likelihood*=cond_prob
        
        priors=prior(Y_train,label)
        post=likelihood*priors
        post_probs.append(post)
        
    return np.argmax(post_probs)
                
    


# In[122]:


print(predict(X_train,Y_train,X_test[1]))
print(Y_test[10])


# In[144]:


def score(x_train,y_train,x_test,y_test):
    results=[]
    for i in range(x_test.shape[0]):
        result=predict(x_train,y_train,x_test[i])
        results.append(result)
    
    results=np.array(results)
    
    accuracy=np.sum(results==y_test)/y_test.shape[0]
        
    return accuracy
        
        


# In[147]:


print(score(X_train,Y_train,X_test,Y_test))


# In[148]:


p=[[2,3],[5,4]]
p[0][1]


# In[160]:


p=[1,2,3,4,5,9,
  5,6,4,3,5,3,
  0,5,3,2,4,5]


# In[172]:





# In[ ]:




