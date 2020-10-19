#!/usr/bin/env python
# coding: utf-8

# ## Telecom Churn Prediction of High Value Customers

# Only 4 months of data considered for the training

# In[1]:


# Suppressing Warnings
import warnings
warnings.filterwarnings('ignore')


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# In[3]:


#Loading the csv into Data Frame
telecom_df = pd.read_csv("./datasets/telecom_churn_data.csv")
telecom_df.head()


# In[4]:


telecom_df.info(verbose=1, null_counts=True, memory_usage=True)


# In[5]:


telecom_df.shape


# #### Handling Missing Values:

# In[6]:


init_rows, init_cols = telecom_df.shape
print("Number of rows", init_rows)
print("Number of columns", init_cols)


# In[7]:


# Checking the percentage of missing values
missing_percent = pd.DataFrame(round(100*(telecom_df.isnull().sum()/len(telecom_df.index)), 2), columns=['Percentage'])
missing_percent.reset_index(inplace=True)
missing_percent.rename(columns={'index':'features'}, inplace=True)

print("missing percentage < 10% ", missing_percent[(missing_percent['Percentage'] > 0.00) & (missing_percent['Percentage'] < 10.00)].count())
print("missing percentage 10-60% ", missing_percent[(missing_percent['Percentage'] >= 10.00) & (missing_percent['Percentage'] < 60.00)].count())
print("missing percentage >= 60% ", missing_percent[(missing_percent['Percentage'] >= 60.00)].count())


# In[8]:


# Considering attributes with more than 60% missing value
missing_percent[(missing_percent['Percentage'] > 60.00)]


# Here for each user when the date of last recharge data is blank, corresponding recharge data related attributes like count_rech_3g, total_rech_data, arpu_2g, are also blank with same percentage of missing as month wise last recharge date column which implies that user didn't make any data recharge for that month.
# - So we can replace above recharge data related attributes NAN values with 0 

# In[9]:


telecom_df.fb_user_8.value_counts()


# In[10]:


no_data_rech_cols = list(missing_percent[(missing_percent['Percentage'] > 60.00)].features)
no_data_rech_cols


# In[11]:


# Replacing those missing values with 0 for above columns
telecom_df[no_data_rech_cols] = telecom_df[no_data_rech_cols].fillna(value=0)


# In[12]:


# reinitializing missing_percent data frame with HVC_telecom
# Checking the percentage of missing values
missing_percent = pd.DataFrame(round(100*(telecom_df.isnull().sum()/len(telecom_df.index)), 2), columns=['Percentage'])
missing_percent.reset_index(inplace=True)
missing_percent.rename(columns={'index':'features'}, inplace=True)

print("missing percentage > 60% ", missing_percent[(missing_percent['Percentage'] > 60.00)].count())


# Other features are having less than **10** % of missing data.

# In[13]:


# Considering attributes with less than 6% missing value
missing_percent[(missing_percent['Percentage'] > 0.00) & (missing_percent['Percentage'] < 10.00)]


# In[14]:


# Validating data in loc_og_t2o_mou attribute
telecom_df['loc_og_t2o_mou'].value_counts()


# In[15]:


# Validating data in std_og_t2o_mou attribute
telecom_df['std_og_t2o_mou'].value_counts()


# In[16]:


# Validating data in loc_ic_t2o_mou attribute
telecom_df['loc_ic_t2o_mou'].value_counts()


# The above three columns `loc_ic_t2o_mou, loc_og_t2o_mou, std_og_t2o_mou` have most of values are zero and having 1 % missing values, so we can drop these attributes

# In[17]:


# Dropping above mentioned attributes
telecom_df.drop(['std_og_t2o_mou', 'loc_og_t2o_mou', 'loc_ic_t2o_mou'], axis=1, inplace=True)
telecom_df.head()


# `last_date_of_month_7` `last_date_of_month_8` and `last_date_of_month_9` null values should be replaced by 7/31/2014, 8/31/2014, 9/30/2014.

# In[18]:


# Filling last date of month columns with last date
telecom_df['last_date_of_month_7'] = telecom_df['last_date_of_month_7'].fillna(value='7/31/2014')
telecom_df['last_date_of_month_8'] = telecom_df['last_date_of_month_8'].fillna(value='8/31/2014')
telecom_df['last_date_of_month_9'] = telecom_df['last_date_of_month_9'].fillna(value='9/30/2014')


# In[19]:


# Verifying recharge date distribution in date_of_last_rech_6
telecom_df['date_of_last_rech_6'].value_counts()


# In[20]:


# Verifying recharge date distribution in date_of_last_rech_7
telecom_df['date_of_last_rech_7'].value_counts()


# In[21]:


# Verifying recharge date distribution in date_of_last_rech_8
telecom_df['date_of_last_rech_8'].value_counts()


# In[22]:


# Verifying recharge date distribution in date_of_last_rech_9
telecom_df['date_of_last_rech_9'].value_counts()


# In[23]:


# Missing data in last date of recharge columns filled up with mode of the same columns
telecom_df['date_of_last_rech_6'] = telecom_df['date_of_last_rech_6'].fillna(
    value=telecom_df['date_of_last_rech_6'].mode()[0])

telecom_df['date_of_last_rech_7'] = telecom_df['date_of_last_rech_7'].fillna(
    value=telecom_df['date_of_last_rech_7'].mode()[0])

telecom_df['date_of_last_rech_8'] = telecom_df['date_of_last_rech_8'].fillna(
    value=telecom_df['date_of_last_rech_8'].mode()[0])

telecom_df['date_of_last_rech_9'] = telecom_df['date_of_last_rech_9'].fillna(
    value=telecom_df['date_of_last_rech_9'].mode()[0])


# In[24]:


# Checking the percentage of missing values
missing_percent = pd.DataFrame(round(100*(telecom_df.isnull().sum()/len(telecom_df.index)), 2), columns=['Percentage'])
missing_percent.reset_index(inplace=True)
missing_percent.rename(columns={'index':'features'}, inplace=True)

print("missing percentage < 10% ", missing_percent[(missing_percent['Percentage'] > 0.00) & (missing_percent['Percentage'] < 10.00)].count())
print("missing percentage 10-60% ", missing_percent[(missing_percent['Percentage'] >= 10.00) & (missing_percent['Percentage'] < 60.00)].count())
print("missing percentage >= 60% ", missing_percent[(missing_percent['Percentage'] >= 60.00)].count())


# In[25]:


missing_percent[(missing_percent['Percentage'] > 0.00) & (missing_percent['Percentage'] < 10.00)]


# All month wise attributes missing percentage is similar like 3.94 % in month 6, 3.86 % in month 7 and 5.38 % in month 8 and 7.75 % in month 9 .
# 

# where Onnet and offnet minute of usages i.e. Call Usages are blank for months other call usage related attributes are also blank that implies that the user didn't make any call.
# - So we can replace attribute NAN values with 0 

# In[26]:


# Filling null columns with zero for month 6, 7, 8, 9
missing_cols = missing_percent[(missing_percent['Percentage'] == 3.94) | (missing_percent['Percentage'] == 3.86) |
                                (missing_percent['Percentage'] == 5.38) | (missing_percent['Percentage'] == 7.75)].features

telecom_df[missing_cols] = telecom_df[missing_cols].fillna(value=0)


# In[27]:


# Checking the percentage of missing values
missing_percent = pd.DataFrame(round(100*(telecom_df.isnull().sum()/len(telecom_df.index)), 2), columns=['Percentage'])
missing_percent.reset_index(inplace=True)
missing_percent.rename(columns={'index':'features'}, inplace=True)

print("missing percentage < 10% ", missing_percent[(missing_percent['Percentage'] > 0.00) & (missing_percent['Percentage'] < 10.00)].count())
print("missing percentage 10-60% ", missing_percent[(missing_percent['Percentage'] >= 10.00) & (missing_percent['Percentage'] < 60.00)].count())
print("missing percentage >= 60% ", missing_percent[(missing_percent['Percentage'] >= 60.00)].count())


# ###### All missing values are handled in the telecom data set

# In[28]:


# Filtering high value customer data set from telecom data
telecom_df['av_total_rech_goodPhase'] = (telecom_df['total_rech_amt_6'] + telecom_df['total_rech_amt_7'])/(
    telecom_df['total_rech_num_6'] + telecom_df['total_rech_num_7'])

var = round(telecom_df['av_total_rech_goodPhase'].quantile(0.7), 2)
print("70% of average Recharge Amount of month 6 and 7", var)


HV_Cust_df = telecom_df[(telecom_df.av_total_rech_goodPhase >= var)]
HV_Cust_df.info()


# In[29]:


# Dropping the average total recharge good phase column
HV_Cust_df.drop(['av_total_rech_goodPhase'], axis=1, inplace=True)


# #### Tag Churn/Non-Churn based on ninth month attributes

# In[30]:


var_list = ['total_ic_mou_9', 'total_og_mou_9', 'vol_2g_mb_9', 'vol_3g_mb_9']
churn_info = HV_Cust_df[var_list]
churn_info.head()


# In[31]:


churn_info.isnull().sum()


# There are no null values in above attributes

# In[32]:


def churn_detect(df):
    if((df['total_ic_mou_9'] == 0.0) & (df['total_og_mou_9'] == 0.0)
      & (df['vol_2g_mb_9'] == 0.0) & (df['vol_3g_mb_9'] == 0.0)):
        return 1
    else:
        return 0

# Applying User defined Churn detection function HV_Cust_df and creating new column Churn    
HV_Cust_df['Churn'] = churn_info.apply(churn_detect, axis=1)       


# In[33]:


# Verifying the Churn column with 9th month attributes
churn_info['Churn'] = HV_Cust_df['Churn']
churn_info.head(10)


# In[34]:


# Removing all attributes corresponding to Churn phase
Nonchurn_phase_cols = [i  for i in HV_Cust_df.columns if "_9" not in i]
len(Nonchurn_phase_cols)


# In[35]:


# Keeping the dataset with non-churn phase attributes only
HVC_telecom = HV_Cust_df[Nonchurn_phase_cols]
HVC_telecom.head()


# In[36]:


HVC_telecom.shape


# In[37]:


HVC_telecom.info(verbose=True, null_counts=True)


# All missing values are handled in the data set. Most of the attributes missing values are imputed with zero to avoid loss of information in class imbalance data set

# #### Exploratory data analysis

# In[38]:


# Verifying recharge date distribution in std_ic_t2o_mou_6
HVC_telecom['std_ic_t2o_mou_6'].value_counts()


# In[39]:


# Verifying recharge date distribution in std_ic_t2o_mou_7
HVC_telecom['std_ic_t2o_mou_7'].value_counts()


# In[40]:


# Verifying recharge date distribution in std_ic_t2o_mou_8
HVC_telecom['std_ic_t2o_mou_8'].value_counts()


# In[41]:


# Verifying recharge date distribution in std_og_t2c_mou_6
HVC_telecom['std_og_t2c_mou_6'].value_counts()


# In[42]:


# Verifying recharge date distribution in std_og_t2c_mou_7
HVC_telecom['std_og_t2c_mou_7'].value_counts()


# In[43]:


# Verifying recharge date distribution in std_og_t2c_mou_8
HVC_telecom['std_og_t2c_mou_8'].value_counts()


# We have also observed that T2O/T2C minute usage in incoming & outgoing STD calls attributes don't have any information.
# The attributes are
# - std_ic_t2o_mou_6
# - std_ic_t2o_mou_7
# - std_ic_t2o_mou_8
# - std_og_t2c_mou_6
# - std_og_t2c_mou_7
# - std_og_t2c_mou_8
# 
# So we can drop these columns in the data set

# In[44]:


drop_cols = ['std_ic_t2o_mou_6', 'std_ic_t2o_mou_7', 'std_ic_t2o_mou_8', 'std_og_t2c_mou_6', 'std_og_t2c_mou_7', 'std_og_t2c_mou_8']
HVC_telecom.drop(columns=drop_cols, axis=1, inplace=True)


# In[45]:


# Dropping the phone number attribute as its no use in model
HVC_telecom.drop(['mobile_number'], axis=1, inplace=True)


# In[46]:


HVC_telecom.head()


# In[47]:


HVC_telecom.circle_id.value_counts()


# In[48]:


# The cirle_id attribute column also has only one value for the whole data set. Hence dropping the column
HVC_telecom.drop(['circle_id'], axis=1, inplace=True)


# In[49]:


# Changing type of date related attributes
date_cols = [cols for cols in HVC_telecom.columns if "date" in cols]
date_cols


# In[50]:


HVC_telecom[date_cols].head()


# In[51]:


HVC_telecom[date_cols] = HVC_telecom[date_cols].apply(pd.to_datetime)


# In[52]:


HVC_telecom[date_cols].head()


# In[53]:


# finding the last recharge data/call for 6, 7, 8 months in days for each user

def find_last_rech(df, str_month_num):
    diff1 = df['last_date_of_month_'+str_month_num] - df['date_of_last_rech_'+str_month_num]
    diff2 = df['last_date_of_month_'+str_month_num] - df['date_of_last_rech_data_'+str_month_num]
    if(diff1.dt.days <= diff2.dt.days).any():
        return diff1.dt.days
    else:
        return diff2.dt.days
    
    
# Applying the function for all months
HVC_telecom['last_rech_day_6'] = find_last_rech(HVC_telecom, "6")
HVC_telecom['last_rech_day_7'] = find_last_rech(HVC_telecom, "7")
HVC_telecom['last_rech_day_8'] = find_last_rech(HVC_telecom, "8")


# In[54]:


# Dropping date columns
HVC_telecom.drop(columns=date_cols, axis=1, inplace=True)


# In[55]:


HVC_telecom.head()


# In[56]:


HVC_telecom.info(verbose=True, memory_usage=True, null_counts=True)


# Converting some attribute types from float64 to int64
# - Count of recharge attributes should be in int format
# - User have Facebook account or not these attributes should be in int format
# - Total recharge count for call and data should also be in integer

# In[57]:


cols_int = [i for i in HVC_telecom.columns if ("count" in i) or ("fb" in i)]

# total 6 columns are there for recharge count
cols_int.extend(['total_rech_num_6','total_rech_num_7', 'total_rech_num_8', 
                'total_rech_data_6','total_rech_data_7', 'total_rech_data_8'])

cols_int


# In[58]:


HVC_telecom[cols_int] = HVC_telecom[cols_int].astype('int64')

HVC_telecom.info(verbose=True, memory_usage=True, null_counts=True)


# In[59]:


HVC_telecom.head(10)


# In[60]:


# Validating Class imbalance in dataset
100*(HVC_telecom.Churn.value_counts(normalize=True))


# **8**  % data is churned in the data set. So the dataset is highly imbalanced

# In[61]:


# Validating Minute of Usage related columns distribution
mou_cols = [i for i in HVC_telecom.columns if "mou" in i]
print("Total MOU attributes", len(mou_cols))

HVC_telecom[mou_cols].describe()


# In[62]:


# Looking into maximum recharge data and average recharge data attributes
HVC_telecom[['max_rech_data_6', 'max_rech_data_7', 'max_rech_data_8', 
             'av_rech_amt_data_6', 'av_rech_amt_data_7', 'av_rech_amt_data_8']].describe(percentiles=[.25,.5,.75,.90,.95,.99])


# In[63]:


sns.boxplot(data=HVC_telecom, y='max_rech_data_8')


# In[64]:


sns.boxplot(data=HVC_telecom, y='max_rech_data_6')


# In[65]:


sns.boxplot(data=HVC_telecom, y='max_rech_data_7')


# In[66]:


cols=['total_rech_data_6', 'total_rech_data_7', 'total_rech_data_8', 'max_rech_data_6', 'max_rech_data_7', 'max_rech_data_8', 
             'av_rech_amt_data_6', 'av_rech_amt_data_7', 'av_rech_amt_data_8']

HVC_telecom[(HVC_telecom['max_rech_data_6'] > 1200) & (HVC_telecom['max_rech_data_7'] > 1200)
           & (HVC_telecom['max_rech_data_8'] > 1200)][cols]


# Clearly above values are outliers for these columns so we can remove these rows

# In[67]:


HVC_telecom = HVC_telecom[(HVC_telecom['max_rech_data_6'] < 1200) & (HVC_telecom['max_rech_data_7'] < 1200)
           & (HVC_telecom['max_rech_data_8'] < 1200)]


# In[68]:


sns.boxplot(data=HVC_telecom, y='max_rech_data_8')


# In[69]:


cols=['total_rech_data_6', 'max_rech_data_6', 'av_rech_amt_data_6']

HVC_telecom[(HVC_telecom['total_rech_data_6'] > 1) & 
            (HVC_telecom['total_rech_data_6'] * HVC_telecom['max_rech_data_6'] == HVC_telecom['av_rech_amt_data_6'])][cols]


# In[70]:


cols=['total_rech_data_7', 'max_rech_data_7', 'av_rech_amt_data_7']

HVC_telecom[(HVC_telecom['total_rech_data_7'] > 1) & 
            (HVC_telecom['total_rech_data_7'] * HVC_telecom['max_rech_data_7'] == HVC_telecom['av_rech_amt_data_7'])][cols]


# In[71]:


cols=['total_rech_data_8', 'max_rech_data_8', 'av_rech_amt_data_8']

HVC_telecom[(HVC_telecom['total_rech_data_8'] > 1) & 
            (HVC_telecom['total_rech_data_8'] * HVC_telecom['max_rech_data_8'] == HVC_telecom['av_rech_amt_data_8'])][cols]


# The above average attributes are somehow wrongly displayed like just the multiplication of number of recharge and maximum recharge.
# We need to fix these average values for all three months considering the number of recharge and maximum recharge values are correct.

# In[72]:


# Lets check together which users average miscalculated and what the Churn count
cols = ['total_rech_data_6', 'total_rech_data_7', 'total_rech_data_8', 'max_rech_data_6', 'max_rech_data_7', 'max_rech_data_8', 
             'av_rech_amt_data_6', 'av_rech_amt_data_7', 'av_rech_amt_data_8']

wrong_cal = HVC_telecom[(HVC_telecom['total_rech_data_6'] > 1) & 
            (HVC_telecom['total_rech_data_6'] * HVC_telecom['max_rech_data_6'] == HVC_telecom['av_rech_amt_data_6']) & 
            (HVC_telecom['total_rech_data_7'] > 1) & 
            (HVC_telecom['total_rech_data_7'] * HVC_telecom['max_rech_data_7'] == HVC_telecom['av_rech_amt_data_7']) & 
            (HVC_telecom['total_rech_data_8'] > 1) & 
            (HVC_telecom['total_rech_data_8'] * HVC_telecom['max_rech_data_8'] == HVC_telecom['av_rech_amt_data_8'])]

print(wrong_cal.shape)
print(wrong_cal.Churn.sum())


# Here couple of rows are there which all month average are miscalculated and Churn rate for these rows are only 5. So we can drop these rows

# In[73]:


# Dropping above rows
HVC_telecom.drop(wrong_cal.index, inplace=True)


# Now considering individual month wise  wrongly calculated averages

# In[74]:


HVC_telecom.Churn.sum()


# In[75]:


# Validating miscalculated average valued datapoints Churn count of month 6 
cols=['total_rech_data_6', 'max_rech_data_6', 'av_rech_amt_data_6', 'Churn']

HVC_telecom[(HVC_telecom['total_rech_data_6'] > 1) & 
            (HVC_telecom['total_rech_data_6'] * HVC_telecom['max_rech_data_6'] == HVC_telecom['av_rech_amt_data_6'])].Churn.sum()


# In[76]:


# Validating miscalculated average valued datapoints Churn count of month 7
cols=['total_rech_data_7', 'max_rech_data_7', 'av_rech_amt_data_7', 'Churn']

HVC_telecom[(HVC_telecom['total_rech_data_7'] > 1) & 
            (HVC_telecom['total_rech_data_7'] * HVC_telecom['max_rech_data_7'] == HVC_telecom['av_rech_amt_data_7'])].Churn.sum()


# In[77]:


# Validating miscalculated average valued datapoints Churn count of month 8
cols=['total_rech_data_8', 'max_rech_data_8', 'av_rech_amt_data_8', 'Churn']

HVC_telecom[(HVC_telecom['total_rech_data_8'] > 1) & 
            (HVC_telecom['total_rech_data_8'] * HVC_telecom['max_rech_data_8'] == HVC_telecom['av_rech_amt_data_8'])].Churn.sum()


# As the above rows Churn information has almost 10% data we should not drop rows for these. We approximate the average value by dividing the max data by recharge count

# In[78]:


# Creating user defined function for calculating average recharge data for above rows
def cal_av_rech_data(df, col):
    if((df['total_rech_data_'+col] > 1) & 
            (df['total_rech_data_'+col] * df['max_rech_data_'+col] == df['av_rech_amt_data_'+col])).all():
        return df['max_rech_data_'+col]/df['total_rech_data_'+col]
    else:
        return df['av_rech_amt_data_'+col]

# Applying the function for all months
HVC_telecom['av_rech_amt_data_6'] = cal_av_rech_data(HVC_telecom, "6")
HVC_telecom['av_rech_amt_data_7'] = cal_av_rech_data(HVC_telecom, "7")
HVC_telecom['av_rech_amt_data_8'] = cal_av_rech_data(HVC_telecom, "8")


# In[79]:


HVC_telecom[['av_rech_amt_data_6', 'av_rech_amt_data_7', 'av_rech_amt_data_8']]


# We have observed that total recharge data count is distributed in 2G & 3G data count attributes. So we can drop total rech data attributes 

# In[80]:


# Dropping the total rech data attributes
HVC_telecom.drop(columns=['total_rech_data_6', 'total_rech_data_7', 'total_rech_data_8'], axis=1, inplace=True)

HVC_telecom.head()


# As per requirement month 6 and 7 are treated as Good phase. We can merge 6, 7 month attributes together.

# In[81]:


cols_6_7 = [col for col in HVC_telecom.columns if ("_6" in col) or ("_7" in col)]

for i,v in enumerate(cols_6_7):
    if("_6" in v):
        col = "av_"+v.split("6")[0]+"good"
        HVC_telecom[col] = (HVC_telecom[v] + HVC_telecom[cols_6_7[i+1]])/2

# Dropping the previous 6 and 7 attributes
HVC_telecom.drop(columns=cols_6_7, axis=1, inplace=True)

HVC_telecom.head()


# In[82]:


# Rechecking the percentage of missing values
missing_percent = pd.DataFrame(round(100*(HVC_telecom.isnull().sum()/len(HVC_telecom.index)), 2), columns=['Percentage'])
missing_percent.reset_index(inplace=True)
missing_percent.rename(columns={'index':'features'}, inplace=True)

print("missing percentage < 10% ", missing_percent[(missing_percent['Percentage'] > 0.00) & (missing_percent['Percentage'] < 10.00)].count())
print("missing percentage 10-60% ", missing_percent[(missing_percent['Percentage'] >= 10.00) & (missing_percent['Percentage'] < 60.00)].count())
print("missing percentage >= 60% ", missing_percent[(missing_percent['Percentage'] >= 60.00)].count())


# In[83]:


# Storing the above data set to csv
HVC_telecom.to_csv("./datasets/HVC_telecom_cleaned.csv")


# In[84]:


# Loading the last stored data set from csv
HVC_telecom = pd.read_csv("./datasets/HVC_telecom_cleaned.csv")
HVC_telecom.drop("Unnamed: 0", axis=1, inplace=True)

HVC_telecom.head()


# In[85]:


# divide the data set in to X & y
X = HVC_telecom.drop('Churn', axis = 1)
y = HVC_telecom['Churn']

# Now will split the data into train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, random_state=100)


# In[86]:


# Apply scaling on the train and test data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[87]:


# We will apply PCA on train data set
X_train.shape


# #### Considering PCA to reduce the features then we will apply model on that.

# In[88]:


#Initializing the PCA module
pca = PCA(svd_solver='randomized', random_state=50)

#Doing the PCA on the train data
pca.fit(X_train)


# In[89]:


#Making the screeplot - plotting the cumulative variance against the number of components
get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.grid("grey")
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()


# From the scree plot we can see that **95%** variance of dataset explained by **50** components

# In[90]:


#Using incremental PCA for efficiency - saves a lot of time on larger datasets
pca_final = IncrementalPCA(n_components=50)

# Transforming the X_train final 50 components PCA
df_train_pca = pca_final.fit_transform(X_train)
df_train_pca.shape


# In[91]:


# Validating minimum and maximum correlations between components
corrmat = np.corrcoef(df_train_pca.transpose())
corrmat_nodiag = corrmat - np.diagflat(corrmat.diagonal())
print("max corr:",corrmat_nodiag.max(), ", min corr: ", corrmat_nodiag.min(),)
# we see that correlations are indeed very close to 0


# The correlations between components are almost 0. So we can used these components in our model.

# In[92]:


#Applying selected components to the test data - 50 components
df_test_pca = pca_final.transform(X_test)
df_test_pca.shape


# In[93]:


get_ipython().run_line_magic('matplotlib', 'inline')
fig = plt.figure(figsize = (8,8))
plt.scatter(df_train_pca[:,0], df_train_pca[:,1], c = y_train.map({0:'green',1:'red'}))
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.tight_layout()
plt.show()


# #### Building the model:

# - We are considered class_weight parameter as `balanced` in each model due class imbalance.
# - `Sensitivity` or `Recall` score is considered primarily for evaluating models as we are more focused to capture Churned customers than non-churn.

# Applying PCA components on simple **Logistic Regression model**

# In[94]:


# First will start with Logistic regression
logReg = LogisticRegression(class_weight='balanced')
model = logReg.fit(df_train_pca, y_train)

# predicting the Churn/Non-Churn using the model
y_pred = model.predict(df_test_pca)

print("Accuracy Score {0}:".format(metrics.accuracy_score(y_test, y_pred)))
print("Sensitivity/Recall {0}:".format(metrics.recall_score(y_test, y_pred)))
metrics.confusion_matrix(y_test, y_pred)


# Applying PCA components on **Advanced Regression** with GridSearchCV

# In[95]:


# We will proceed with tuning the hyperparamters and advanced regression using GridSearch
logR = LogisticRegression(class_weight='balanced')
params = {'penalty': ['l1', 'l2'],
             'C':[0.0001, 0.001, 0.01, 0.1, 0.2, 0.5, 1]}

# Applying the GridSearch using scoring as recall
grid = GridSearchCV(estimator= logR, param_grid=params, cv=5, verbose=1, scoring='recall',return_train_score=True)
grid.fit(df_train_pca, y_train)

print("Best train Score: ",grid.best_score_)
print("Best Parameter: ",grid.best_params_)

# Applying GridSearch Parameters on LogisticRegression
logReg = LogisticRegression(penalty=grid.best_params_['penalty'], 
                            C=grid.best_params_['C'], class_weight='balanced')
model = logReg.fit(df_train_pca, y_train)

# predicting the Churn/Non-Churn using the model
y_pred = model.predict(df_test_pca)

# Validating the Accuracy, Recall and Confusion matrix
print("Accuracy Score {0}:".format(metrics.accuracy_score(y_test, y_pred)))
print("Sensitivity/Recall {0}:".format(metrics.recall_score(y_test, y_pred)))
metrics.confusion_matrix(y_test, y_pred)


# Considering **RandomForestClassifier**

# In[96]:


# Applying the RandomForest on this dataset

rfc = RandomForestClassifier(class_weight='balanced')
model2 = rfc.fit(df_train_pca, y_train)

# predicting the Churn/Non-Churn using the model
y_pred = model2.predict(df_test_pca)

# Validating the Accuracy, Recall and Confusion matrix
print("Accuracy Score {0}:".format(metrics.accuracy_score(y_test, y_pred)))
print("Sensitivity/Recall {0}:".format(metrics.recall_score(y_test, y_pred)))
metrics.confusion_matrix(y_test, y_pred)


# Considering **Advanced Regression with Elastic net**

# In[97]:


# considering elastic net for this data set
logR = LogisticRegression(class_weight='balanced')
params = {'penalty': ['l1', 'l2'],
            'C':[0.0001, 0.001, 0.01, 0.1, 0.2, 0.5, 1],
            'l1_ratio':[0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 1]}
grid = GridSearchCV(estimator= logR, param_grid=params, cv=5, verbose=1, scoring='recall',return_train_score=True)
grid.fit(df_train_pca, y_train)

print("Elastic Net Train score: {0}".format(grid.best_score_))
print("Elastic Net params: {0}".format(grid.best_params_))

logReg = LogisticRegression(penalty=grid.best_params_['penalty'], C=grid.best_params_['C'],
                            class_weight='balanced', l1_ratio= grid.best_params_['l1_ratio'])
model = logReg.fit(df_train_pca, y_train)

# predicting the Churn/Non-Churn using the model
y_pred = model.predict(df_test_pca)

# Validating the Accuracy, Recall and Confusion matrix
print("Accuracy Score {0}:".format(metrics.accuracy_score(y_test, y_pred)))
print("Sensitivity/Recall {0}:".format(metrics.recall_score(y_test, y_pred)))
metrics.confusion_matrix(y_test, y_pred)


# ###### Selecting best model for prediction

# Now from above models Accuracy and sensitivity metrics we choose Ridge (L2) regression model is suitable for the prediction. The L2 regression model is applied on the PCA components.
# 
# As per the business requirement we have chosen Sensitivity over Accuracy. it is important to predict Churn users as they will take action to retain those customers. In that case if some of the non-churn user gets those perks/offers mistakenly it will not be much affect business.
# 
# The simple Logistic regression model with class weight balanced parameter overall performance also good but sensitivity is more on Ridge Regression model.

# In[98]:


# The final model for Churn/Non-Churn prediction
logReg = LogisticRegression(penalty='l2', 
                            C=0.0001, class_weight='balanced')
classifier = logReg.fit(df_train_pca, y_train)

# predicting the Churn/Non-Churn using the model
y_pred = classifier.predict(df_test_pca)

# Validating the Accuracy, Recall and Confusion matrix
print("Accuracy Score {0}:".format(metrics.accuracy_score(y_test, y_pred)))
print("Sensitivity/Recall {0}:".format(metrics.recall_score(y_test, y_pred)))
metrics.confusion_matrix(y_test, y_pred)


# In[99]:


# Saving the model in pkl file
pickle.dump(classifier, open('./model/model.pkl', 'wb'))

