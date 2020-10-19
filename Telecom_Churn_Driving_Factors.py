#!/usr/bin/env python
# coding: utf-8

# ## Important Variables thats drives Churn in Telecom Industry

# ###### Cleaning the datasets already done in Churn prediction ML model py files. We have referred the same datasets here.

# ##### **Building Second Model** for important predictor attributes

# For predicting the important feature best model we choose Simple Logistic Regression & Decision Tree.
# 
# - RFE need to apply for Feature reduction before applying Logistic model and then we will reduce feature by checking VIFs.
# - For Decision Tree model Tree depth will be chosen appropriately so that based on important feature tree branch will divide.

# ##### Feature Selection Using RFE

# Considering Logistic Regression with RFE (15 features)

# In[1]:


# Suppressing Warnings
import warnings
warnings.filterwarnings('ignore')


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(class_weight='balanced')


# Applying modelling on actual X_train & X_test data as these are actual attributes not the principal components

# In[3]:


# Loading the Cleaned High Value Customers (HVC) telecom dataset
HVC_telecom = pd.read_csv('datasets\HVC_telecom_cleaned.csv')
HVC_telecom.head()


# In[27]:


# divide the data set in to X & y
X = HVC_telecom.drop('Churn', axis = 1)
y = HVC_telecom['Churn']

# Now will split the data into train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, random_state=100)


# In[28]:


# Apply scaling on the train and test data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.values)
X_test_scaled = scaler.transform(X_test.values)

print(type(X_train_scaled))
print(type(X_test_scaled))


# In[29]:


# Converting X_train_scaled to dataframe
X_train = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)
X_train.describe()

# Converting X_test_scaled to dataframe
X_test = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)


# In[30]:


# Applying RFE on Logistic Regression
rfe = RFE(logreg, 15)
rfe = rfe.fit(X_train, y_train)


# In[31]:


list(zip(X_train.columns, rfe.support_, rfe.ranking_))


# In[32]:


col = X_train.columns[rfe.support_]
col


# ##### Assessing the model with StatsModels

# In[33]:


import statsmodels.api as sm


# In[34]:


X_train_sm = sm.add_constant(X_train[col])
logm = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm.fit()
res.summary()


# In[35]:


# Create a VIF dataframe
vif = pd.DataFrame()
vif['Features'] = X_train[col].columns
vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# Dropping `av_total_og_mou_good` feature first as it has high VIF

# In[36]:


col = col.drop('av_total_og_mou_good', 1)
col


# In[37]:


# Let's re-run the model using the selected variables
X_train_sm = sm.add_constant(X_train[col])
logm = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm.fit()
res.summary()


# In[38]:


# Create a VIF dataframe
vif = pd.DataFrame()
vif['Features'] = X_train[col].columns
vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# All VIFs are less than 4 which is good but `av_std_og_mou_good` feature P value increased to 0.31. So we need to drop these variable.

# In[39]:


col = col.drop('av_std_og_mou_good', 1)
col


# In[40]:


# Let's re-run the model using the selected variables
X_train_sm = sm.add_constant(X_train[col])
logm = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm.fit()
res.summary()


# In[41]:


# Create a VIF dataframe
vif = pd.DataFrame()
vif['Features'] = X_train[col].columns
vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# Now we can see that All features `P value are almost zero and VIFs are less than 4`. So we can go with these features.
# 
# Now we will verify the accuracy and recall score of the model

# In[42]:


# Storing the probabiities to a dataframe
y_train_pred= pd.DataFrame({'Churn_actual':y_train.values,'Churn_Prob': res.predict(X_train_sm)})

# Creating new column 'predicted' with 1 if Churn_Prob > 0.5 else 0
y_train_pred['predicted'] = y_train_pred.Churn_Prob.map(lambda x: 1 if x > 0.5 else 0)


y_train_pred.head()

print(metrics.accuracy_score(y_train_pred.Churn_actual, y_train_pred.predicted))


# Now try to reduce some more feature from the model and will check the accuracy

# In[43]:


# Lets drop the `fb_user_8` feature as business nothing to do with. Business can't take any action based on this
col = col.drop('fb_user_8', 1)
col


# In[44]:


# Let's re-run the model using the selected variables
X_train_sm = sm.add_constant(X_train[col])
logm = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm.fit()
res.summary()


# In[45]:


# Create a VIF dataframe
vif = pd.DataFrame()
vif['Features'] = X_train[col].columns
vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[46]:


# Storing the probabiities to a dataframe
y_train_pred= pd.DataFrame({'Churn_actual':y_train.values,'Churn_Prob': res.predict(X_train_sm)})

# Creating new column 'predicted' with 1 if Churn_Prob > 0.5 else 0
y_train_pred['predicted'] = y_train_pred.Churn_Prob.map(lambda x: 1 if x > 0.5 else 0)


y_train_pred.head()

print(metrics.accuracy_score(y_train_pred.Churn_actual, y_train_pred.predicted))


# Now we will find optimul cutoff for the Churn probability and will verify the evaluation metrics

# In[47]:


# Let's create columns with different probability cutoffs 
numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred[i]= y_train_pred.Churn_Prob.map(lambda x: 1 if x > i else 0)
y_train_pred.head()


# In[48]:


# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.
cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])
from sklearn.metrics import confusion_matrix

# TP = confusion[1,1] # true positive 
# TN = confusion[0,0] # true negatives
# FP = confusion[0,1] # false positives
# FN = confusion[1,0] # false negatives

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred.Churn_actual, y_train_pred[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)

# Let's plot accuracy sensitivity and specificity for various probabilities.
cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.grid("dark_whitegrid")
plt.show()


# #### From the curve above, 0.15 is the optimum point to take it as a cutoff probability.

# In[49]:


y_train_pred['final_predicted'] = y_train_pred.Churn_Prob.map( lambda x: 1 if x > 0.15 else 0)

print("Train Accuracy Score: ",metrics.accuracy_score(y_train_pred.Churn_actual, y_train_pred.final_predicted))
print("Train Recall Score: ",metrics.recall_score(y_train_pred.Churn_actual, y_train_pred.final_predicted))
print(metrics.confusion_matrix(y_train_pred.Churn_actual, y_train_pred.final_predicted))


# In[50]:


# Applying previous model on test set
X_test_sm = sm.add_constant(X_test[col])
y_test_pred = res.predict(X_test_sm)

# Defining probabilities on Dataframe based on optimul cutoff
y_pred_df = pd.DataFrame({'y_test_prob':y_test_pred, 'y_test_actual': y_test})
y_pred_df['final_predicted'] = y_pred_df.y_test_prob.map( lambda x: 1 if x > 0.15 else 0)

# Validating evaluation metric
print("Train Accuracy Score: ",metrics.accuracy_score(y_pred_df.y_test_actual, y_pred_df.final_predicted))
print("Train Recall Score: ",metrics.recall_score(y_pred_df.y_test_actual, y_pred_df.final_predicted))
print(metrics.confusion_matrix(y_pred_df.y_test_actual, y_pred_df.final_predicted))


# The above same model is also pretty good for Testset with 86 % accuracy and 80% precision. So we can refer selected features of the model and important driving factors of Churn.

# - `total_og_mou_8`
# - `loc_ic_mou_8`
# - `std_ic_t2f_mou_8`
# - `spl_ic_mou_8`
# - `last_day_rch_amt_8`
# - `aon`
# - `sep_vbc_3g`
# - `av_arpu_good`
# - `av_onnet_mou_good`
# - `av_offnet_mou_good`
# - `av_loc_ic_mou_good`
# - `av_total_rech_num_good`

# Applying **DecisionTreeClassifier** model to the dataset

# In[70]:


from sklearn.tree import DecisionTreeClassifier

# Create the parameter grid 
param_grid = {
    'max_depth': range(3, 25, 5),
    'min_samples_leaf': range(50, 150, 50),
    'min_samples_split': range(50, 150, 50),
    'criterion': ["entropy", "gini"]
}

n_folds = 5

# Instantiate the grid search model
dtree = DecisionTreeClassifier(class_weight='balanced')
grid_search = GridSearchCV(estimator = dtree, param_grid = param_grid, 
                          cv = n_folds, scoring='recall', verbose = 1, return_train_score= True,)

# Fit the grid search to the data
grid_search.fit(X_train,y_train)

# printing the optimal accuracy score and hyperparameters
print("DecisionTree score: {0}".format(grid_search.best_score_))
print("DecisionTree params: {0}".format(grid_search.best_estimator_))


# In[127]:


# model with optimal hyperparameters
dtree = DecisionTreeClassifier(criterion = 'gini', 
                                  random_state = 100,
                                  max_depth=8, 
                                  min_samples_leaf=100,
                                  min_samples_split=50,
                              class_weight='balanced')
dtree.fit(X_train, y_train)

y_pred = dtree.predict(X_test)

# Validating the Accuracy, Recall and Confusion matrix
print("Accuracy Score {0}:".format(metrics.accuracy_score(y_test, y_pred)))
print("Sensitivity/Recall {0}:".format(metrics.recall_score(y_test, y_pred)))
metrics.confusion_matrix(y_test, y_pred)


# In[128]:


# Visualization of the decision tree
from IPython.display import Image  
from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz
import pydotplus, graphviz

# Putting features
features = list(X.columns[0:])


# plotting the tree
dot_data = StringIO()  
export_graphviz(dtree, out_file=dot_data,feature_names=features,filled=True,rounded=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())


# In[129]:


# model with optimal hyperparameters and max_depth as 3
dtree = DecisionTreeClassifier(criterion = 'gini', 
                                  random_state = 100,
                                  max_depth=3, 
                                  min_samples_leaf=100,
                                  min_samples_split=50,
                              class_weight='balanced')
dtree.fit(X_train, y_train)

y_pred = dtree.predict(X_test)

# Validating the Accuracy, Recall and Confusion matrix
print("Accuracy Score {0}:".format(metrics.accuracy_score(y_test, y_pred)))
print("Sensitivity/Recall {0}:".format(metrics.recall_score(y_test, y_pred)))
metrics.confusion_matrix(y_test, y_pred)


# In[130]:


# Visualization of the decision tree
from IPython.display import Image  
from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz
import pydotplus, graphviz

# Putting features
features = list(X.columns[0:])


# plotting the tree
dot_data = StringIO()  
export_graphviz(dtree, out_file=dot_data,feature_names=features,filled=True,rounded=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())


# Here by decreasing the depth of Decision Tree to **3** we found the important of attributes which drives the Churn.
# 
# - loc_ic_mou_8
# - av_rech_amt_data_8
# - total_og_mou_8
# - av_monthly_2g_good
# - roam_og_mou_8
# - last_day_rch_amt_8

# In[75]:


HVC_telecom.head()


# In[105]:


# Plotting each attribute to see the behavior
plt.figure(figsize=(10,8))
plt.grid()
sns.countplot(x= HVC_telecom[HVC_telecom['Churn'] == 1].av_monthly_2g_good)
plt.show()


# Maximum Churned user average monthly 2G data recharge is 0.0 on Good Phase. So we should focus on monthly 2G data recharge scheme to decrease Churn count

# In[104]:


# Plotting each attribute to see the behavior
plt.figure(figsize=(10,8))
plt.grid()
sns.countplot(x= HVC_telecom[HVC_telecom['Churn'] == 1].last_day_rch_amt_8)
plt.show()


# Here we can see that Churned user's last day is at day 0. Implies that the User who are recharged at day 0 at 8th month and further didn't make recharge on are Churned.

# Based on above two models most of the important driving factors of Churn are attributes,
# 
# - `Local Incoming call minute of usages`
# - `Outgoing call minute of usages`
# - `Average mobile data recharge, last data recharge amount`
# - `2G mobile Pack recharges on 6th & 7th months`

# ##### Corrective Actions based on Driving factors

# 1. There are couple attributes are present regarding `Incoming Usage` like overall call MOU of 3 months, average On & Off network minute usage of Good Phase, STD and Special call incoming usages.
# 
#     Incoming MOU attrubites are having negative coefficient, i.e. User experience somehow not good in that case. Business should improve Service quantity of incoming calls, networking as Offers/benefits can't be provide on incoming calls.

# 2. `Outgoing` usage related attribute Total Outgoing Calls MOU also a important parameter.
# 
#     Useful Offers or benefit can be provided for Users to increase the overall outgoing MOU

# 3. Another important parameter is `Mobile data`. Average data recharge on Action phase, Monthly recharge amount, number of recharges are important attributes.
# 
#     Business should more focused on decreasing Mobile Data recharge price or extra benefits on data recharges which may lower existing customers Churn probability.
#     
#     Another attribute is last day of recharge amount on Action phase. Business should track these users and at the end/middle of validity business should provide benefit Offers to those user to make user recharge again.
