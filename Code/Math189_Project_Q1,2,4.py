#!/usr/bin/env python
# coding: utf-8

# # Math 189 Final Project
# ## Vineet Tallavajhala

# ### Import Wall and setup

# In[709]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[733]:


df = pd.read_csv('kaggle_survey_2020_responses.csv')


# In[734]:


df.drop(0, inplace= True)


# In[735]:


df['Time from Start to Finish (seconds)'] = df['Time from Start to Finish (seconds)'].astype(int)


# In[737]:


pd.set_option('display.max_columns', 500)


# ### Question 1 - Summarizing the Data

# In[738]:


demographic_df = df.iloc[:, [i for i in range (1,7)]]


# In[741]:


demographic_df.columns


# In[763]:


demographic_df['Q1'].value_counts().plot(kind = 'bar', title = 'Age Distribution of Kaggle Users', xlabel = 'Age Range', ylabel = 'Count')


# In[780]:


demographic_df['Q2'].value_counts().drop('Prefer to self-describe').plot(kind = 'bar', title = 'Gender Distribution of Kaggle Users', xlabel = 'Gender', ylabel = 'Count')


# In[775]:


demographic_df['Q4'].value_counts().drop('Some college/university study without earning a bachelor’s degree').drop('No formal education past high school').plot(kind = 'bar', title = 'Education Distribution of Kaggle Users', xlabel = 'Degree', ylabel = 'Count')


# In[772]:


demographic_df['Q5'].value_counts().plot(kind = 'bar', title = 'Job Title Distribution of Kaggle Users', xlabel = 'Occupation', ylabel = 'Count')


# In[773]:


demographic_df['Q6'].value_counts().plot(kind = 'bar', title = 'Programming Experience Distribution of Kaggle users', xlabel = 'Years Programming', ylabel = 'Count')


# In[774]:


df['Q24'].value_counts().plot(kind = 'bar', title = 'Yearly Compensation Distribution of Kaggle Users', xlabel = 'Salary Range', ylabel = 'Count')


# ### Question 2, analyzing duration time with different demographics

# In[748]:


# comparing yearly compensation by gender
a = pd.pivot_table(df, values = 'Q3', index = 'Q24', columns = 'Q2', aggfunc = 'count').fillna(0)
shortened_df = a[['Man', 'Woman']]


# In[749]:


# Time from Start to Finish (seconds)
shortened_df.plot.bar(figsize = (20, 15))
plt.title('Comparing the Pay Between Women and Men Data Scientists', size = 20)
plt.xlabel('Salary Range', size = 20)
plt.ylabel('Count', size = 20)
plt.legend(prop = {'size': 30})


# In[783]:


a = pd.pivot_table(df, values = 'Time from Start to Finish (seconds)', index = 'Q24', columns = 'Q2', aggfunc = np.mean)
shortened_df = a[['Man', 'Woman']]
shortened_df.plot.bar(figsize = (20, 15))
plt.title('Comparing the Mean Duration Time Between Women and Men Data Scientists of Different Salaries', size = 20)
plt.xlabel('Salary Range', size = 20)
plt.ylabel('Mean Duration Time', size = 20)
plt.xticks(size = 20)
plt.yticks(size = 15)
plt.legend(prop = {'size': 20})


# In[786]:


a = pd.pivot_table(df, values = 'Time from Start to Finish (seconds)', index = 'Q3', columns = 'Q2', aggfunc = np.mean)
shortened_df = a[['Man', 'Woman']]
shortened_df.plot.bar(figsize = (20, 15))
plt.title('Comparing the Mean Duration Time Between Women and Men Data Scientists from Different Countries', size = 20)
plt.ylabel('Mean Duration Time', size = 20)
plt.xticks(size = 15)
plt.yticks(size = 15)
plt.legend(prop = {'size': 20})


# In[793]:


a = pd.pivot_table(df, values = 'Time from Start to Finish (seconds)', index = 'Q4', columns = 'Q2', aggfunc = np.mean)
shortened_df = a[['Man', 'Woman']]
shortened_df.drop('Some college/university study without earning a bachelor’s degree', inplace= True)
shortened_df.plot.bar(figsize = (15, 10))
plt.title('Comparing the Mean Duration Time Between Women and Men Data Scientists of Different Education Levels', size = 20)
plt.xlabel('Education Level', size = 20)
plt.ylabel('Mean Duration Time', size = 20)
plt.xticks(size = 15)
plt.yticks(size = 15)
plt.legend(prop = {'size': 20})


# In[790]:





# In[794]:


a = pd.pivot_table(df, values = 'Time from Start to Finish (seconds)', index = 'Q5', columns = 'Q2', aggfunc = np.mean)
shortened_df = a[['Man', 'Woman']]
shortened_df.plot.bar(figsize = (20, 15))
plt.title('Comparing the Mean Duration Time Between Women and Men Data Scientists of Different Occupation Titles', size = 20)
plt.xlabel('Occupation Title', size = 20)
plt.ylabel('Mean Duration Time', size = 20)
plt.legend(prop = {'size': 20})


# In[ ]:





# In[666]:


type(df['Time from Start to Finish (seconds)'][1])


# In[758]:


df


# ## Question 4, regression to predict 

# In[667]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score


# ### Preparing the DF for modeling

# - Questions that need to be one hot encoded
#  - 1,2,3,4,5,6, 8, 11, 13, 15, 20, 21, 22, 24, 25, 30, 32, 38, 

# In[668]:


df.drop(0, inplace = True)


# In[669]:


questions_list = [1, 2 ,3,4,5,6, 8, 11, 13, 15, 20, 21, 22, 24, 25, 30, 32, 38]


# In[670]:


for i in questions_list:
    temp = pd.get_dummies(df['Q' + str(i)])
    df = pd.concat([temp, df], axis = 1)
    df.drop(['Q' + str(i)], axis = 1, inplace = True)


# In[671]:


temp = df.copy()


# In[672]:


for i in temp.columns:
    if 'Part' in i or 'OTHER' in i:
        temp[i] = temp[i].where(~df[i].notna(), 1)
        temp[i] = temp[i].fillna(0)


# #### Model Training Begins Here

# In[673]:


def feature_r_squared(df):
    results = {}
    y = df['Time from Start to Finish (seconds)']
    fixed_df = df.drop('Time from Start to Finish (seconds)', axis = 1)
    for i in fixed_df.columns:
        temp_X = pd.DataFrame(df[i])
        X_train, X_test, y_train, y_test = train_test_split(temp_X, y, test_size = 0.25)
        reg = LinearRegression()
        reg.fit(X_train, y_train)
        score = reg.score(X_test, y_test)
        results[i] = score
    return results


# In[674]:


r_squared_dict = feature_r_squared(temp)


# In[675]:


r_squared_df = pd.DataFrame.from_dict(r_squared_dict, orient = 'index')


# In[676]:


r_squared_df.sort_values(by = 0, ascending = False)


# In[677]:


sorted(r_squared_dict.values())


# In[680]:


y = df['Time from Start to Finish (seconds)']
X = temp.drop('Time from Start to Finish (seconds)', axis = 1)


# In[681]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)


# In[682]:


reg = LinearRegression()
reg.fit(X_train, y_train)


# In[683]:


print(reg.score(X_test, y_test))


# In[684]:


y_pred = reg.predict(X_test)


# In[685]:


r2_score(y_test, y_pred)


# ### Modeling with fewer features

# In[686]:


df


# In[687]:


demographic_df


# In[688]:


demographic_df.drop(0, inplace = True)


# In[689]:


for i in demographic_df.columns:
    temp = pd.get_dummies(demographic_df[i])
    demographic_df = pd.concat([temp, demographic_df], axis = 1)
    demographic_df.drop([i], axis = 1, inplace = True)


# In[690]:


demographic_df


# In[691]:


y = df['Time from Start to Finish (seconds)']
X = demographic_df[['Man', 'Woman']]


# In[693]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)


# In[694]:


reg = LinearRegression()
reg.fit(X_train, y_train)


# In[695]:


print(reg.score(X_test, y_test))


# In[696]:


X = temp.drop('Time from Start to Finish (seconds)', axis = 1)


# In[ ]:




