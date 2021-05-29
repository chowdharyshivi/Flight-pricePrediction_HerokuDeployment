#!/usr/bin/env python
# coding: utf-8

# In[1]:


pwd


# In[2]:


import pandas as pd      #data cleaning,exploring
import numpy as np       #scientific computing and multidimensional arrays
import matplotlib.pyplot as plt #Matplotlib is a cross-platform, data visualization and graphical plotting library for Python and its numerical extension NumPy.
import seaborn as sns #uses Matplotlib underneath to plot graphs. It will be used to visualize random distributions.


# In[3]:


train_data=pd.read_excel(r'flight_price_prediction.xlsx')
train_data.head()


# In[5]:


train_data.info()


# In[6]:


train_data.isnull().sum()


# In[7]:


train_data.dropna() #inplace=true means dataframe makes permanent changes


# In[8]:


train_data.isnull().sum()


# In[9]:


train_data.dropna(inplace=True) #inplace=true means dataframe makes permanent changes
train_data.isnull().sum()


# In[10]:


train_data.dtypes


# In[11]:


def change_into_datetime(col):
    train_data[col]=pd.to_datetime(train_data[col])


# In[12]:


train_data.columns


# In[13]:


for i in ['Date_of_Journey','Dep_Time','Arrival_Time']:
    change_into_datetime(i)
    


# In[14]:


train_data.dtypes


# In[15]:


train_data['Journey_Day']=train_data['Date_of_Journey'].dt.day
train_data['Journey_Month']=train_data['Date_of_Journey'].dt.month


# In[16]:


#train_data.drop(columns=['Date_of_Journey'],inplace=True)
train_data


# In[17]:


def extract_hour(df,col):
    df[col+"_hour"]=df[col].dt.hour
    
def extract_min(df,col):
    df[col+"_minute"]=df[col].dt.minute
    
def drop_col(df,col):
    df.drop(columns=[col],inplace=True)


# In[18]:


extract_hour(train_data,'Dep_Time')
extract_min(train_data,'Dep_Time')
drop_col(train_data,'Dep_Time')


# In[19]:


train_data.head()


# In[ ]:





# In[20]:


extract_hour(train_data,'Arrival_Time')
extract_min(train_data,'Arrival_Time')
drop_col(train_data,'Arrival_Time')


# In[21]:


train_data.head()


# In[22]:


train_data['Duration']


# In[23]:


duration=list(train_data['Duration'])

for i in range(len(duration)):
    if(len(duration[i].split(' '))==2):
        pass
    else:
        if 'h' in duration[i]:
            duration[i]=duration[i]+' 0m'
        else:
            duration[i]='0h '+duration[i]
    


# In[24]:


train_data['Duration']=duration


# In[25]:


def hour(x):
    return x.split(' ')[0][0:-1]

def min(x):
    return x.split(' ')[1][0:-1]


# In[ ]:





# In[26]:


'2h 40m'.split(' ')[0][0:-1]


# In[27]:


train_data['Duration_hours']=train_data['Duration'].apply(hour)
train_data['Duration_mins']=train_data['Duration'].apply(min)


# In[28]:


train_data.drop('Duration',axis=1,inplace=True)


# In[29]:


train_data.dtypes


# In[30]:


train_data.head()


# In[31]:


train_data['Duration_hours']=train_data['Duration_hours'].astype(int)
train_data['Duration_mins']=train_data['Duration_mins'].astype(int)


# In[32]:


train_data.dtypes


# In[33]:


# to find which column is int and which is object

cat_col=[col for col in train_data.columns if train_data[col].dtype=='O']
cat_col


# In[34]:


cant_col=[col for col in train_data.columns if train_data[col].dtype!='O']
cant_col


# In[ ]:





# ## Handling Categorical Data

# Nominal data --> data are not in any order --> OneHotEncoder is used in this case
# 
# 
# Ordinal data --> data are in order -->       LabelEncoder is used in this case

# In[35]:


categorical=train_data[cat_col]
categorical


# In[36]:


train_data['Airline'].value_counts()


# In[37]:


plt.figure(figsize=(20,10))
sns.boxplot(x='Airline',y='Price',data=train_data.sort_values('Price',ascending=False))


# In[38]:


plt.figure(figsize=(20,10))
sns.boxplot(x='Total_Stops',y='Price',data=train_data.sort_values('Price',ascending=False))


# In[39]:


len(categorical['Airline'].unique())


# ##  As Airline is Nominal Categorical data we will perform OneHotEncoding
# 
# 

# In[40]:


Airline = pd.get_dummies(categorical['Airline'],drop_first=True)
Airline.head()


# In[41]:


train_data['Airline'].unique()


# In[42]:


categorical['Source'].value_counts()


# # Source Vs Price

# In[43]:


plt.figure(figsize=(20,10))
sns.catplot(x='Source',y='Price',data=train_data.sort_values('Price',ascending=False),kind='boxen')


# In[53]:


# As Source is Nominal Categorical data we will perform OneHotEncoding


Source=pd.get_dummies(categorical['Source'],prefix='s_',drop_first=True)
Source.head()


# In[54]:


categorical['Destination'].value_counts()


# In[55]:


plt.figure(figsize=(20,10))
sns.catplot(x='Destination',y='Price',data=train_data.sort_values('Price',ascending=False),kind='boxen')


# In[56]:


# As Destination is Nominal Categorical data we will perform OneHotEncoding

Destination=pd.get_dummies(categorical['Destination'],prefix='d_',drop_first=True)
Destination.head()


# In[57]:


categorical['Route'].value_counts()


# In[58]:


plt.figure(figsize=(30,20))
sns.catplot(x='Route',y='Price',data=train_data.sort_values('Price',ascending=False))


# In[59]:


categorical['Route_1']=categorical['Route'].str.split('→').str[0]
categorical['Route_2']=categorical['Route'].str.split('→').str[1]
categorical['Route_3']=categorical['Route'].str.split('→').str[2]
categorical['Route_4']=categorical['Route'].str.split('→').str[3]
categorical['Route_5']=categorical['Route'].str.split('→').str[4]


# In[60]:


import warnings 
from warnings import filterwarnings
filterwarnings('ignore')


# In[61]:


categorical['Route_1'].fillna('None',inplace=True)
categorical['Route_2'].fillna('None',inplace=True)
categorical['Route_3'].fillna('None',inplace=True)
categorical['Route_4'].fillna('None',inplace=True)
categorical['Route_5'].fillna('None',inplace=True)


# In[62]:


categorical.head()


# In[63]:


for i in categorical.columns:
     print(categorical[i].value_counts())
     print("\n")


# In[ ]:





# ### as we will see we have lots of features in Route , one hot encoding will not be a better option lets appply Label Encoding

# In[64]:


from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()


# In[65]:


for i in ['Route_1', 'Route_2', 'Route_3', 'Route_4','Route_5']:
    categorical[i]=encoder.fit_transform(categorical[i])


# In[66]:


categorical.head()


# In[ ]:





# In[67]:


drop_col(categorical,'Route')
drop_col(categorical,'Additional_Info')


# In[68]:


categorical['Total_Stops'].value_counts()


# In[69]:


categorical['Total_Stops'].unique()


# In[70]:


dict={'non-stop':0, '2 stops':2, '1 stop':1, '3 stops':3, '4 stops':4}


# In[71]:


categorical['Total_Stops']=categorical['Total_Stops'].map(dict)


# In[72]:


categorical.head()


# In[ ]:





# In[ ]:





# In[73]:


train_data[cant_col]


# In[74]:


Airline


# In[75]:


Source


# In[76]:


data_train = pd.concat([categorical,Airline,Source,Destination,train_data[cant_col]],axis=1)


# In[77]:


data_train


# In[78]:


drop_col(data_train,'Airline')
drop_col(data_train,'Source')
drop_col(data_train,'Destination')


# In[79]:


drop_col(data_train,'Date_of_Journey')


# In[80]:


data_train.head()


# In[81]:


data_train.columns


# In[82]:


data_train.info()


# In[ ]:





# ## Outlier Detection

# In[83]:


def plot(df,col):
    fig,(ax1,ax2)=plt.subplots(2,1)
    sns.distplot(df[col],ax=ax1)
    sns.boxplot(df[col],ax=ax2)


# In[84]:


plt.figure(figsize=(30,20))
plot(data_train,'Price')


# In[85]:


data_train['Price']=np.where(data_train['Price']>=40000,data_train['Price'].median(),data_train['Price'])


# In[86]:


plt.figure(figsize=(30,20))
plot(data_train,'Price')


# ## Seperate dependent and independent variable

# In[87]:


#Dependent variables

x = data_train.drop('Price',axis=1)
x.head()


# In[88]:


y=data_train['Price']
y


# In[89]:


type(x)


# In[90]:


x.isnull().sum()


# In[91]:


y.isnull().sum()


# In[92]:


type(y)


# In[93]:


np.array(x)


# In[94]:


np.array(y)


# ## Feature selection

#  Feature selection is done to find out the best independent fetures which contributes most to the target value(price or dependent feature)

# In[95]:


from sklearn.feature_selection import mutual_info_classif


# In[96]:


mutual_info_classif(x,y)


# In[97]:


imp=pd.DataFrame(mutual_info_classif(x,y),index=x.columns)
imp


# In[98]:


imp.columns=['importance']
imp.sort_values(by='importance',ascending=False)


# In[99]:


from sklearn.model_selection import train_test_split


# In[100]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[ ]:





# In[101]:


from sklearn import metrics
##dump your model using pickle so that we will re-use
import pickle
def predict(ml_model,dump):
    model=ml_model.fit(x_train,y_train)
    print('Training score : {}'.format(model.score(x_train,y_train)))
    y_prediction=model.predict(x_test)
    print('predictions are: \n {}'.format(y_prediction))
    print('\n')
    r2_score=metrics.r2_score(y_test,y_prediction)
    print('r2 score: {}'.format(r2_score))
    print('MAE:',metrics.mean_absolute_error(y_test,y_prediction))
    print('MSE:',metrics.mean_squared_error(y_test,y_prediction))
    print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test,y_prediction)))
    sns.distplot(y_test-y_prediction)


# In[102]:


from sklearn.ensemble import RandomForestRegressor


# In[103]:


predict(RandomForestRegressor(),1)


# In[104]:


from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor


# In[105]:


predict(DecisionTreeRegressor(),0)


# In[106]:


predict(LinearRegression(),0)


# In[107]:


predict(KNeighborsRegressor(),0)


# In[108]:


from sklearn.model_selection import RandomizedSearchCV


# In[109]:


# Number of trees in random forest
n_estimators=[int(x) for x in np.linspace(start=100,stop=1200,num=6)]

# Number of features to consider at every split
max_features=['auto','sqrt']

# Maximum number of levels in tree
max_depth=[int(x) for x in np.linspace(5,30,num=4)]

# Minimum number of samples required to split a node
min_samples_split=[5,10,15,100]


# In[110]:



random_grid={
    'n_estimators':n_estimators,
    'max_features':max_features,
'max_depth':max_depth,
    'min_samples_split':min_samples_split
}


# In[111]:


reg_rf=RandomForestRegressor()


# In[112]:



rf_random=RandomizedSearchCV(estimator=reg_rf,param_distributions=random_grid,cv=3,verbose=2,n_jobs=-1)


# In[113]:


rf_random.fit(x_train,y_train)


# In[114]:


rf_random.best_params_


# In[115]:


prediction=rf_random.predict(x_test)


# In[116]:


sns.distplot(y_test-prediction)


# In[117]:


metrics.r2_score(y_test,prediction)


# In[118]:


print('MAE',metrics.mean_absolute_error(y_test,prediction))
print('MSE',metrics.mean_squared_error(y_test,prediction))
print('RMSE',np.sqrt(metrics.mean_squared_error(y_test,prediction)))


# In[120]:


get_ipython().system('pip install pickle')


# In[121]:


import pickle


# In[122]:


file=open('flight_price_pred.pkl','wb')


# In[123]:


pickle.dump(rf_random,file)


# In[124]:


model=open('flight_price_pred.pkl','rb')
forest=pickle.load(model)
y_prediction=forest.predict(x_test)
result=metrics.r2_score(y_test,y_prediction)
print(result)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




