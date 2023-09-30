#!/usr/bin/env python
# coding: utf-8

# In[42]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[43]:


sns.get_dataset_names()


# In[44]:


iris=sns.load_dataset('iris')


# In[45]:


iris.head()


# In[46]:


iris.describe()


# In[47]:


iris.value_counts()


# In[48]:


iris['sepal_length'].nunique()


# In[49]:


iris['petal_length'].nunique()


# In[50]:


iris['species'].nunique()


# In[51]:


iris=iris[['sepal_length','petal_length','species']]

iris.head()


# In[52]:


iris['species'].value_counts()


# In[53]:


iris.isnull().sum()


# In[54]:


sns.pairplot(iris,hue='species')


# In[55]:


from sklearn.preprocessing import LabelEncoder
model = LabelEncoder()


# In[56]:


iris['species'] = model.fit_transform(iris['species'])


# In[57]:


iris.head()


# In[58]:


from sklearn.model_selection import train_test_split


# In[59]:


x_train,x_test,y_train,y_test=train_test_split(iris.drop(columns=['species']),iris['species'], test_size=0.3)
x_train.shape,x_test.shape


# In[60]:


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)


# In[61]:


from sklearn.metrics import accuracy_score
pred=model.predict(x_test)
accuracy_score(y_test,pred)


# In[62]:


from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier()


# In[63]:


model.fit(x_train,y_train)


# In[64]:


pred1=model.predict(x_test)
accuracy_score(y_test,pred1)


# In[65]:


from sklearn.tree import DecisionTreeClassifier
def training_model():
    model=DecisionTreeClassifier()
    trained_model = model.fit(x_train,y_train)
    return trained_model


# In[66]:


pred2=model.predict(x_test)
accuracy_score(y_test,pred2)*100


# In[67]:


pred2


# In[68]:


x_test.head()


# In[69]:


import pickle


# In[70]:


pickle.dump(model,open('model.pkl', 'wb'))


# In[71]:


load_model=pickle.load(open('model.pkl','rb'))


# In[72]:


load_model.predict([[7.7,6.9]])


# In[ ]:





# In[ ]:





# In[ ]:




