#!/usr/bin/env python
# coding: utf-8

# In[47]:


import pandas as pd
import numpy as np


# In[73]:


x1=pd.ExcelFile('Data_Train.xlsx')
print(x1)
df1=x1.parse()
df1


# In[51]:


x2=pd.ExcelFile('Data_Test.xlsx')
print(x1)
df2=x2.parse()
df2.head()


# In[75]:


Story=pd.concat([df1,df2])
Story


# In[52]:


stopwords= set(['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',             'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those',             'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',             'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',             'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very',             's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're',             've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't",             'won', "won't", 'wouldn', "wouldn't"])


# In[53]:


from bs4 import BeautifulSoup


# In[54]:


import re

def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


# In[77]:


from tqdm import tqdm
p_stories = []
# tqdm is for printing the status bar
for sentance in tqdm(Story['STORY'].values):
    sentance = re.sub(r"http\S+", "", sentance)
    sentance = BeautifulSoup(sentance, 'lxml').get_text()
    sentance = decontracted(sentance)
    sentance = re.sub("\S*\d\S*", "", sentance).strip()
    sentance = re.sub('[^A-Za-z]+', ' ', sentance)
    # https://gist.github.com/sebleier/554280
    sentance = ' '.join(e.lower() for e in sentance.split() if e.lower() not in stopwords)
    p_stories.append(sentance.strip())


# In[78]:


len(p_stories)


# In[ ]:





# In[ ]:





# In[79]:


from sklearn.feature_extraction.text import TfidfVectorizer

tf_idf_vect = TfidfVectorizer(ngram_range=(1,2), min_df=10)

tf_idf_vect.fit(p_stories)
print("some sample features(unique words in the corpus)",tf_idf_vect.get_feature_names()[0:10])
print('='*50)

final_tf_idf = tf_idf_vect.transform(p_stories)
print("the type of count vectorizer ",type(final_tf_idf))
print("the shape of out text TFIDF vectorizer ",final_tf_idf.get_shape())
print("the number of unique words including both unigrams and bigrams ", final_tf_idf.get_shape()[1])


# In[ ]:





# In[82]:


X_train=final_tf_idf[:7628]
y_train=df1.SECTION


# In[ ]:





# In[83]:


X_train.shape


# In[ ]:





# In[84]:


y_train.shape


# In[85]:


X_test=final_tf_idf[7628:]
X_test.shape


# In[167]:


from sklearn import svm
clf=svm.SVC(gamma=0.1,C=40)
clf.fit(X_train,y_train)


# In[168]:


pred=clf.predict(X_test)


# In[152]:


pred


# In[169]:


xp=[]
for i in range (len(pred)):
    xp.append(pred[i])


# In[170]:


xp


# In[171]:


len(xp)


# In[172]:


df2.columns


# In[173]:


df2['SECTION']=pd.Series(xp,index=df2.index)


# In[174]:


df2.head


# In[178]:


df2.to_excel(r'Documents\final4.xlsx')


# In[176]:


xl=pd.ExcelFile('final2.xlsx')
print(xl)
xl1=xl.parse()
xl1


# In[177]:


from sklearn.metrics import accuracy_score
acc_bow = accuracy_score(xl1.SECTION, pred) * 100
print('svm accuracy ',acc_bow)


# In[124]:





# In[161]:


px=[]
for i in range(len(xp)):
    px.append(df2.SECTION[i]-xl1.SECTION[i])


# In[162]:


px


# In[1]:


df=pd.DataFrame(px,columns=['diff'])
df=df(index=False)
df


# In[165]:


df.to_excel(r'Documents\diff2.xlsx')


# In[ ]:




