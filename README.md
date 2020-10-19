# Telecom Churn Prediction Case Study 

## Table of Content
  * [Overview](#overview)
  * [Motivation](#motivation)
  * [Understanding of Churn](#understanding-of-churn)
  * [Business Aspect](#business-aspect)
  * [Technical Aspect](#technical-aspect)
  * [Installation](#installation)
  * [Directory Tree](#directory-tree)  
  * [Technologies Used](#technologies-used)
  * [Team](#team)
  * [Credits](#credits)


## Overview
This is a vanilla classification model for a most common dataset, Telecom Churn predictions in Indian and South Asian Market. This project has two parts. A Telecom Churn prediction model for High Value Customers with maximum accuracy and another approach to find driving factors that influencing Churn in the Telecom company.

## Motivation
In a Telecom industry, customers can choose from multiple service providers and actively switch from one operator to another. In this highly competitive market, the telecommunications industry experiences an average of 15-25% annual churn rate. Given the fact that it costs 5-10 times more to acquire a new customer than to retain an existing one, customer retention has now become even more important than customer acquisition.

<kbd><img src= "https://miro.medium.com/max/300/1*ajR-SIQzi8Reb6VaL5x2Gg.png"></kbd>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<kbd><img src= "https://miro.medium.com/max/300/1*A-6PzYAEzHbXejMOoOYYHg.jpeg"></kbd>

For many incumbent operators, retaining high profitable customers is the number one business goal.

To reduce customer churn, telecom companies need to predict which customers are at high risk of churn.

## Understanding of Churn
There are two main models of payment in the telecom industry - postpaid and prepaid.

In the **Postpaid model**, when customers want to switch to another operator, they usually inform the existing operator, and you directly know that this is an instance of churn.

However, in the **Prepaid model**, customers who want to switch to another network can simply stop using the services without any notice, and it is hard to know whether someone has actually churned or is simply not using the services temporarily.

Prepaid is the most common model in India and southeast Asia, while postpaid is more common in Europe in North America. This project is based on the Indian and Southeast Asian market.

**Revenue-based churn:** Customers who have not utilized any revenue-generating facilities such as mobile internet, outgoing calls, SMS etc. over a given period of time. One could also use aggregate metrics such as ‘customers who have generated less than INR 4 per month in total/average/median revenue’.

The main shortcoming of this definition is that there are customers who only receive calls/SMSs from their wage-earning counterparts.

**Usage-based churn:** Customers who have not done any usage, either incoming or outgoing - in terms of calls, internet etc. over a period.
A potential shortcoming of this definition is that when the customer has stopped using the services for a while, it may be too late to take any corrective actions to retain them. For e.g., if you define churn based on a ‘two-months zero usage’ period, predicting churn could be useless since by that time the customer would have already switched to another operator.

In this project, the usage-based definition has been considered.

High-value Churn: Approximately 80% of revenue comes from the top 20% customers (called high-value customers). Thus, if we can reduce churn of the high-value customers, we will be able to reduce significant revenue leakage.

Here, high-value customer-level data has been analyzed of a leading telecom firm, built predictive models to identify customers at high risk of churn and identify the main indicators of churn.

## Business Aspects
The dataset contains customer-level information for a span of four consecutive months - June, July, August and September.

The business objective is to predict the churn in the last (i.e. 9th) month using the data from the first three months.

Customers usually do not decide to switch to another competitor instantly, but rather over a period (especially applicable to high-value customers). In churn prediction, we assume that there are three phases of customer lifecycle:

The **‘good’** phase: In this phase, the customer is happy with the service and behaves as usual.

The **‘action’** phase: The customer experience starts to sore in this phase, for e.g. he/she gets a compelling offer from a competitor, faces unjust charges, becomes unhappy with service quality etc. In this phase, the customer usually shows different behavior than the ‘good’ months. Also, it is crucial to identify high-churn-risk customers in this phase, since some corrective actions can be taken at this point 

The **‘churn’** phase: In this phase, the customer is said to have churned. You define churn based on this phase. Also, it is important to note that at the time of prediction, this data is not available to you for prediction. Thus, after tagging churn as 1/0 based on this phase, you discard all data corresponding to this phase.

## Technical Aspect
This project is divided into two part:
1. Training a classification model using [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) to predict telecom churn as accurate as possible.
	- Cleaning the datasets, fixing all features
	- Applying all [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) to reduce the datasets features
	- Apply Classification ML model
2. Considering previous Cleaned dataset have to find important variables that are strong predictors of Churn.
    - Apply Recursive Feature Elimination with Logistic Regression
    - Eliminate features by checking p-values, VIFs of each feature using statsmodel API.

## Installation
The Code is written in Python 3.7. If you don't have Python installed you can find it [here](https://www.python.org/downloads/). If you are using a lower version of Python you can upgrade using the pip package, ensuring you have the latest version of pip. To install the required packages and libraries, run this command in the project directory after [cloning](https://www.howtogeek.com/451360/how-to-clone-a-github-repository/) the repository:
```bash
pip install -r requirements.txt
```

## Directory Tree 
```
├── data dictionary 
│   └── Data_Dictionary_Telecom_Churn_Case_Study.xlsx
├── datasets
│   ├── telecom_churn_data.csv
│   └── HVC_telecom_cleaned.csv
├── model
│   └── model.pkl
├── Telecom_Churn_Driving_Factors.py
├── Telecom_Churn_Prediction.py
├── requirements.txt
└── README.md
```

## Technologies Used

![](https://forthebadge.com/images/badges/made-with-python.svg)

[<img target="_blank" src="https://numpy.org/images/logos/numpy.svg" width=100>](https://numpy.org)    [<img target="_blank" src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/ed/Pandas_logo.svg/450px-Pandas_logo.svg.png" width=150>](https://pandas.pydata.org)    [<img target="_blank" src="https://scikit-learn.org/stable/_static/scikit-learn-logo-small.png" width=150>](https://scikit-learn.org/stable)   [<img target="_blank" src="https://www.statsmodels.org/stable/_images/statsmodels-logo-v2-horizontal.svg" width=170>](https://www.statsmodels.org)

[<img target="_blank" src="https://matplotlib.org/_static/logo2_compressed.svg" width=170>](https://matplotlib.org)      [<img target="_blank" src="https://seaborn.pydata.org/_static/logo-wide-lightbg.svg" width=150>](https://seaborn.pydata.org)      [<img target="_blank" src="https://jupyter.org/assets/nav_logo.svg" width=150>](https://jupyter.org)

## Team
[![Debmalya Ghosal](https://avatars2.githubusercontent.com/u/60285205?s=144&u=45fc55fc21b66ed5ea26153766e3d8e1cc3f4449&v=4)](https://github.com/debmalya92) |
-|
[Debmalya Ghosal](https://github.com/debmalya92) |)

## Credits
- The datasets & data dictionary has been provided by [UpGrad](https://www.upgrad.com/). This project wouldn't have been possible without this dataset.
