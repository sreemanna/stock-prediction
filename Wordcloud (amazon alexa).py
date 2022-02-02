#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


#step2 importing amazon alexa
df_reviews=pd.read_csv('amazon_alexa.tsv',sep='\t')


# In[4]:


df_reviews.head()


# In[5]:


df_reviews.shape


# In[6]:


df_reviews.info()


# In[7]:


df_reviews.describe()


# In[8]:


df_reviews['date']=pd.to_datetime(df_reviews['date'])


# In[10]:


# display year
df_reviews['date'].dt.year.value_counts()


# In[11]:


df_reviews['date'].min()


# In[12]:


plt.figure(figsize=(15,16))
sns.countplot(x='date',data=df_reviews)
plt.xticks(rotation=90)
plt.show();


# In[13]:


sns.countplot(df_reviews['date'].dt.month)


# In[15]:


df_reviews['date'].dt.month.value_counts()


# In[16]:


sns.countplot(x='rating',data=df_reviews)


# In[18]:


df_reviews.rating.value_counts()


# In[20]:


sns.countplot(x='feedback',data=df_reviews)


# In[22]:


df_reviews['length']=df_reviews['verified_reviews'].apply(lambda x:len(x.split()))
df_reviews.head()
plt.hist(x='length',data=df_reviews,bins=30)
df_reviews.length.describe()


# In[23]:


pip install wordcloud


# In[35]:


#generating wordcloud
neg=df_reviews[df_reviews['feedback']==0]
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
text =neg['verified_reviews'].values
wordcloud=WordCloud(
    width=3000,
    height=2000,
    background_color='black',
    stopwords=STOPWORDS).generate(str(text))
fig=plt.figure(
    figsize=(40,30),
    facecolor='k',
    edgecolor='k')
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis('off')
plt.tick_params(axis='both',labelsize=14)
plt.show()


# In[36]:


df_reviews['variation'].value_counts().plot.bar()


# In[37]:


df_reviews['variation'].value_counts()


# In[38]:


sns.countplot(x='variation',data=df_reviews)
plt.title('Variation Distribution in Alexa')
plt.xlabel('variation')
plt.ylabel('count')
plt.xticks(rotation='vertical')
plt.show


# In[ ]:




