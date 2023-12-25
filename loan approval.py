#!/usr/bin/env python
# coding: utf-8

# In[1]:


#loan approval prediction
#importing all the libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import svm


# In[2]:


#import the dataset loan.csv
df=pd.read_csv("C:\\Users\\TUSHAR SAIN\\Downloads\\loan_data_set.csv")
#information about the dataset
df.info()


# In[3]:


df.describe()


# In[4]:


#dataset
df


# In[5]:


plt.hist(df["LoanAmount"],bins=20)


# In[6]:


df['LoanAmount_log']=np.log(df["LoanAmount"])
plt.hist(df["LoanAmount_log"],bins=20)


# In[7]:


df.isnull().sum()


# In[8]:


df["TotalIncome"]=df["ApplicantIncome"]+df["CoapplicantIncome"]
df["TotalIncome_log"]=np.log(df["TotalIncome"])
plt.hist(df["TotalIncome_log"],bins=20)


# In[9]:


df["Gender"].fillna(df["Gender"].mode()[0],inplace=True)
df["Married"].fillna(df["Married"].mode()[0],inplace=True)
df["Self_Employed"].fillna(df["Self_Employed"].mode()[0],inplace=True)
df["Dependents"].fillna(df["Dependents"].mode()[0],inplace=True)

df.LoanAmount=df.LoanAmount.fillna(df.LoanAmount.mean())
df.LoanAmount_log=df.LoanAmount_log.fillna(df.LoanAmount_log.mean())

df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].mode()[0],inplace=True)
df["Credit_History"].fillna(df["Credit_History"].mode()[0],inplace=True)

df.isnull().sum()


# In[10]:


x=df.iloc[:,np.r_[1:5,9:11,13:15]].values
y=df.iloc[:,12].values


# In[11]:


y


# In[12]:


print("percent of missing gender is:",(df["Gender"].isnull().sum()/df.shape[0])*100,"%")


# In[13]:


print("Number of people who take loan as groupe by gender")
print(df["Gender"].value_counts())
sns.countplot(x=df["Gender"])



# In[14]:


print("Number of people who take loan as groupe by maritial status")
print(df["Married"].value_counts())
sns.countplot(x=df["Married"])


# In[15]:


print("Number of people who take loan as groupe by Dependents")
print(df["Dependents"].value_counts())
sns.countplot(x=df["Dependents"])


# In[16]:


print("Number of people who take loan as groupe by Loan_Amount")
print(df["LoanAmount"].value_counts())
plt.figure(figsize=(12,6))
sns.countplot(x=df["LoanAmount"])


# In[17]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[18]:


from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
for i in range(0,5):
    x_train[:,i]=lb.fit_transform(x_train[:,i])
    x_train[:,7]=lb.fit_transform(x_train[:,7])
x_train   


# In[19]:


y_train=lb.fit_transform(y_train)


# In[20]:


y_test=lb.fit_transform(y_test)


# In[21]:


for i in range(0,5):
    x_test[:,i]=lb.fit_transform(x_test[:,i])
    x_test[:,7]=lb.fit_transform(x_test[:,7])


# In[23]:


x_test


# In[24]:


from sklearn.preprocessing import StandardScaler
sd=StandardScaler()
x_train=sd.fit_transform(x_train)
x_test=sd.fit_transform(x_test)


# In[43]:


#by random forest classification
from sklearn.ensemble import RandomForestClassifier
rf_classifier=RandomForestClassifier()
rf_classifier.fit(x_train,y_train)


# In[44]:


y_pred=rf_classifier.predict(x_test)
from sklearn import metrics
print("the Random Forest Classifier accuracy:",metrics.accuracy_score(y_pred,y_test))


# In[45]:


#from the decision tree classifier
from sklearn.tree import DecisionTreeClassifier
clf_tree=DecisionTreeClassifier(criterion="entropy",max_depth=2)
clf_tree.fit(x_train,y_train)
y_predict=clf_tree.predict(x_test)
print("the decision tree classifier accuracy:",metrics.accuracy_score(y_predict,y_test))


# In[46]:


#by logestic regression
from sklearn.linear_model import LogisticRegression
classifier= LogisticRegression()
classifier.fit(x_train,y_train)
y_predict1=classifier.predict(x_test)
print("the logistic regression accuracy:",metrics.accuracy_score(y_predict1,y_test))


# In[47]:


#by k_mean classification
from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=5)
classifier.fit(x_train,y_train)
y_predict2=classifier.predict(x_test)
print("the k_mean accuracy:",metrics.accuracy_score(y_predict2,y_test))


# In[48]:


#naive_bayes
from sklearn.naive_bayes import GaussianNB
classifier= GaussianNB()
classifier.fit(x_train,y_train)
y_predict3=classifier.predict(x_test)
print("the  navie_bayes accuracy:",metrics.accuracy_score(y_predict3,y_test))


# In[ ]:




