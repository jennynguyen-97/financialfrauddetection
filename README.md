# Fraud Detection in Financial Payment Services using Paysim Synthetic Dataset of Mobile Money Transactions

The [World Economic Forum](https://www.mckinsey.com/business-functions/risk-and-resilience/our-insights/financial-crime-and-fraud-in-the-age-of-cybersecurity) noted that fraud and financial crime was a trillion-dollar industry, reporting that private companies spent approximately $8.2 billion on anti–money laundering (AML) controls alone in 2017. In the age of automation and digitization, along with a massive growth in transaction volumes, financial crimes themselves, detected and undetected, have become more numerous and costly than ever.

![frauds-in-audit](https://user-images.githubusercontent.com/93355594/147002073-cf91f2f9-8f92-4ca1-9525-276e329314b3.jpeg)

## I. Business Understanding
The cost of fraud pre-COVID-19 among U.S. financial services and lending firms rose 12.8% over the previous reporting period, which covered the first halves of 2018 and 2019 respectively. For every dollar of fraud lost in the pre-COVID period, U.S. financial services and lending companies incurred an average of $3.78 in costs, up from $3.35. These losses include the transaction face value for which firms are held liable, plus fees and interest incurred, fines and legal fees, labor and investigation costs and external recovery expenses. The COVID-19 pandemic has had a significantly negative impact on financial services and lending firms. The volume of successful attacks has risen across segments during COVID-19, most dramatically among larger institutions, causing a spike in the cost of fraud.

Due to the increasingly sophisticated and costly frauds, the financial services industry must continue to accelerate and innovate how it prevent, detect, and investigate fraudulent activities. One of the most promising solutions to achieve this goal is through machine learning algorithms.

## II. Data Understanding
### A. Data Overview
The dataset used in this project is from [Kaggle - Paysim synthetic financial dataset for fraud detection](https://www.kaggle.com/arjunjoshua/predicting-fraud-in-financial-payment-services/data). Paysim provided synthetic data for mobile money transactions in a one month period. This dataset contains 11 attributes and roughly 6.3 million observations. There is no null value in the dataset so it does not require data cleaning. A summary of variable characteristics is provided below:

<p align="center"><img width="579" alt="Screen Shot 2021-12-21 at 6 45 03 PM" src="https://user-images.githubusercontent.com/93355594/147011796-eef8ab41-77e3-471a-b680-3ae85b54dfa6.png">

<p align="center"><img width="911" alt="Screen Shot 2021-12-21 at 6 56 01 PM" src="https://user-images.githubusercontent.com/93355594/147013424-718daaf1-fd1b-4ff4-bfca-7b20eb0299a8.png">
  
The main technical challenge in any financial fraud dataset is the highly imbalanced distribution between 0 and 1 (or legitimate and fraudulent transactions). This project will show how imbalanced dataset is resolved and choose a suitable machine learning algorithm to deal with the skew.
  
### B. Exploratory Data Analysis
First, I visualize the frequency of fraud, meaning which percentage of total transactions is fraudulent. The dataset is highly imbalanced with fraudulent transactions only account for 0.13%.

<p align="center"><img width="729" alt="Screen Shot 2021-12-21 at 7 13 34 PM" src="https://user-images.githubusercontent.com/93355594/147013778-247b1675-bc1e-4f7c-8947-c94a0b3d80d7.png">
  
Second, I want to know which types make up the majority of fraudulent transactions. Interestingly, only two transaction types out of five - CASH-OUT and TRANSFER have fraudulent activities. TRANSFER means money is sent to a customer/fraudster and CASH-OUT means money is sent to a merchant who pays the customer/fraudster in cash. Remarkably, the number of fraudulent TRANSFERs almost equals the number of fraudulent CASH-OUTs. These observations make sense in which frauds are committed by first transferring out funds to another account which subsequently cashes it out.

<p align="center"><img width="723" alt="Screen Shot 2021-12-21 at 7 24 11 PM" src="https://user-images.githubusercontent.com/93355594/147014501-e65d28b4-ac4c-40a3-8636-569e4875d652.png">

Third, it would be helpful to know the distribution of amount of fraudulent transactions. According to this histogram, fraudulent amount has positive skew with most transactions between 0 - 625,000 in local currency. There is also a low peak around 10,000,000. In total, these fraudulent transactions account for a loss of 12,056,415,428 in local currency. 

<p align="center"><img width="706" alt="Screen Shot 2021-12-21 at 7 36 22 PM" src="https://user-images.githubusercontent.com/93355594/147015349-b783359c-075a-4183-96ca-bcbe44481691.png">
  
Fourth, I incorporate a time element associated with fradulent activities in the analysis by taking the modulo (or the remainder) of a division between the step variable and 24. Since each step represents 1 hour of real world and there is a total of 743 steps for 30 days of data, I convert them into 24 hours where each day has 0 to 23 hours. As the graph depicts, most fraudulent transactions occur between 3 to 6 a.m. when there is a lack of real-time human monitoring. Representing time as hours instead of steps, therefore, would add predictive power to the model and represent a pattern that machine learning algorithms can detect.

<p align="center"><img width="710" alt="Screen Shot 2021-12-21 at 7 37 30 PM" src="https://user-images.githubusercontent.com/93355594/147016244-04e44ce2-01d7-4eab-b1f8-bd4e69fa8718.png">

## III. Data Preparation
There is not much cleaning required in this dataset since the data is already clean. However, I decide to drop four columns: step, nameOrig, nameDest, and isFlaggedFraud. step is used in creating the hour variable so I remove it to avoid the multicollinearity issue. nameOrig and nameDest have too many unique levels to create dummy variables. isFlaggedFraud, according to the data description, will be set when an attempt is made to TRANSFER an amount greater than 200,000; however, only 16 entries (out of more than 6 million) are set to 1 and instances where conditions are met but isFlaggedFraud is not set. Consequently, isFlaggedFraud is not consistent with the data description and will be dropped.
  
In addition, I have to determine an effective way to split the train and test sets so that the train dataset is balanced and appropriate for the modeling step. First, I split the dataset into 70:30 with 70% for train dataset and 30% for test dataset. Next, I apply under-sampling majority class method due to the highly skewed dataset. This method balances classes in the train dataset so that there are 5725 observations in each class. Balancing dataset is important in machine learning as feeding imbalanced data to classifier model can make it biased in favor of the majority class, simply because it did not have enough data to learn about the minority.
  



