#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.feature_extraction.text import CountVectorizer


# In[2]:


cv=CountVectorizer()


# In[29]:


corpus=[
    'She is beautiful',
    'World is so cruel',
    'I am vegetarian',
    'Hell I am in love'
]


# In[30]:


vectorized_form=cv.fit_transform(corpus).toarray()
vectorized_form


# In[31]:


print(cv.vocabulary_)


# In[32]:


print(cv.inverse_transform(vectorized_form[3]))


# In[34]:


cv.vocabulary_["she"]


# In[ ]:


# this will actually clean the data using our predefined function of nlp notebook
cv1=CountVectorizer(Tokenizer=nlp) #refer to function nlp notbook
vc=cv1.fit_transform(corpus).toarray()

