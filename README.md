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
  
In addition, I have to determine an effective way to split the train and test sets so that the train dataset is balanced and appropriate for the modeling step. First, I split the dataset into 70:30 with 70% for train dataset and 30% for test dataset. Next, I apply under-sampling majority class method due to the highly skewed dataset. This method balances classes in the train dataset so that there are 5725 observations in each class or 50:50 split among classes 0 and 1. Balancing dataset is important in machine learning as feeding imbalanced data to classifier model can make it biased in favor of the majority class, simply because it did not have enough data to learn about the minority.

## IV. Modeling
### A. Logistic Regression
Logistic Regression is a classification technique used in machine learning. It uses a sigmoid function to model the dependent variable and is bounded between 0 and 1. The objective of the sigmoid function is to minimize the loss function which is a measure of how wrong the model is in terms of its ability to estimate the relationship between independent and dependent variables.

I use the Logistic Regression to model the relationship between the dependent variable - isFraud and the independent variables - type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, and hour. The result of the logistic regression model is shown below:

<p align="center"><img width="912" alt="Screen Shot 2021-12-22 at 3 57 25 PM" src="https://user-images.githubusercontent.com/93355594/147154239-865068cd-4687-451d-a29c-03a6f2ca621f.png">
  
Logistic Regression is easy to implement, interpret, and very efficient to train. It is very fast at classifying unknown records. Additionally, the model coefficients can be interpreted as indicators of feature importance. However, this machine learning algorithm has the tendency to underfit and requires an assumption of linearity between the dependent variable and the independent variables and between the independent variables and the log odds. Since it is important to accurately predict fraudulent transactions as incorrect predictions will create customer friction and produce uncessary constraints for legitimate transactions, I will utilize machine learning models that do not assume linearity between variables and complex models that are more powerful.
  
### B. Decision Tree
Decision Tree is also a type of supervised machine learning where the data is continuously split according to a certain parameter. The tree can be explained by two entities, namely decision nodes and leaves. Decision nodes are where the data is split and leaves are the decisions or the final outcomes. As the Decision Tree depicted, the financial fraud dataset is first splitted by type, then other variables such as amount, oldbalanceOrg, and newbalanceOrig are considered in subsequent splits.

<p align="center"><img width="629" alt="Screen Shot 2021-12-23 at 9 52 12 AM" src="https://user-images.githubusercontent.com/93355594/147256973-3595e590-8e11-4afe-a4f5-3908dc851a00.png">
  
Decision Tree is beneficial in that it requires minimal data cleaning and scaling in the pre-processing step and takes into consideration non-linear parameters. Another added benefit of this model is that its result is clearly visualized and can be easily interpreted by both technical and non-technical audience. While Decision Tree is robust to outliers and missing data, it is highly sensitive to new data points. Adding a new data point can lead to re-generation of the overall tree and all nodes need to be recalculated and recreated. Additionally, Decision Tree is prone to overfitting as in the process of fitting the data (even noisy data), it keeps generating new nodes and ultimately the model becomes too complex to interpret. In this way, it loses its generalization capabilities, performing well on the train dataset but starts making a lot of mistakes on the unseen data.

### C. Random Forest
Next, we will consider Random Forest which is a bagging technique that combines many decision trees. Random Forest reduces the complexity of Decision Tree by taking the average (in case of regression problems) or the majority (in case of classification problems) of the output from various trees. As a result, overfitting is reduced and precision is increased. Random Forest also inherits advantages of Decision Tree such as less susceptible to outliers and missing values, effeciently handle non-linearity, and minimal pre-processing requirements. Despite these advantages, Random Forest is typically more computationally intense than Decision Tree and can not be easily interpreated. The interpretability of individual trees is lost when they are combined together in a Random Forest model.
  
### D. XGBoost
While both XGBoost and Random Forest are ensemble learning methods in machine learning, each has their own differences. Random Forest aims to reduce the complexity of models that overfit the training data. In contrast, XGBoost is an approach to increase the complexity of models that suffer from high bias, that is, models that underfit the training data. Consequently, while Random Forest reduces overfitting, XGBoost may increase it. Additionally, tree models in Random Forest are built independently, tree models are not independent in XGBoost where the machine learning algorithm tries to add new models that do well where previous models fail. Both methods are weighted averages of a combination of trees; however, Random Forest assigns equal weighting while XGBoost assigns higher weights to models that perform well on the train dataset.
  
There are more data pre-processing in XGBoost than in Random Forest. Before modeling, I have to specifically divide the train dataset into data (independent variables) and label (dependent variable). In order to avoid overfitting, I set a parameter named early_stopping_rounds to 50 in which XGBoost will stop early if there is no improvement in learning. Additionally, I use xgmodel$best_iteration to find the optimal nrounds which is the number of decision trees in the final model to be 2290.
  
## V. Evaluation
Since the train dataset is balanced, I do not need to consider metrics that take into account the imbalance of the dataset. I will use four metrics for model evaluation:
  1. Accuracy: measures the number of classifications a model correctly predicts
  2. Area under the curve (AUC): measures the ability of a classifier to distinguish between classes  
  3. False Positive Rate (FPR): measures the percentage of false positives in each model. I do not want a high FPR as it will reduce customer satisfaction and increase customer fraction in using our mobile money app.
  4. False Negative Rate (FNR): measures the percentage of false negatives in each model. Since the cost of undetected fraud is high, I want my model to correctly detect fraudulent transactions in order to implement preventive methods.

The performance of four above-mentioned models according to four performance criterias is listed below:

<p align="center"><img width="581" alt="Screen Shot 2021-12-23 at 11 59 52 AM" src="https://user-images.githubusercontent.com/93355594/147271213-7f9089d1-13e3-486a-aa41-affc8c32dde6.png">

XGBoost outperforms other models on all four metrics, suggesting its high precision and generalization on future unseen data. Consequently, I choose XGBoost as the final model to perform predictions on the train dataset. I also rank feature importance based on XGBoost. As the graph indicates,  oldbalanceOrg, newbalanceOrig, amount, and type provide high informational gains; this finding is consistent with decision nodes in the decision tree model.
  
<p align="center"><img width="603" alt="Screen Shot 2021-12-23 at 12 04 27 PM" src="https://user-images.githubusercontent.com/93355594/147271629-4819779d-2449-4551-a3d0-dc5c408daf54.png">

Banks or mobile money transaction apps can implement this machine learning algorithm in their daily operations and measure its effectiveness and efficiency in detecting fraudulent activities. These financial institutions can implement control charts to track the number or the proportion of fraud transactions missed or legitimate transactions wrongly classified as frauds. In addition, companies can measure the cost-saving from using machine learning to detect frauds. The benefits gained from using machine learning can help organizations increase and enhance their analytics initiatives. Additionally, machine learning can lift sales by frictionlessly approving more legitimate transactions, while taking on fraudulent transactions to reduce operating costs.
  
## VI. Deployment  
When used successfully, machine learning removes heavy burden of data analysis from the fraud detection team. The results help the team with investigation, insights, and reporting. Machine learning doesn’t replace the fraud analyst team, but gives them the ability to reduce the time spent on manual reviews and data analysis. This means analysts can focus on the most urgent cases and assess alerts faster with more accuracy, and also reduce the number of genuine customers declined.

Machines are less good at dealing with uncertainty; there are cases that are new, or that are difficult, or somehow different. Edge cases are those that require more attention and may be difficult to determine - this is where the human insight comes in and provides massive value. The expert human intervention here is not just at the point of approving a transaction; it’s more a case of analysis after the event and labelling the data in a way that gives rapid feedback to a machine. The more confirmed behaviour labels it can receive, the more accurate a result there is likely to be.

There are, however, points in which the machine learning algorithm can be improved. First, while undersampling helps reduce the risk of machine learning algorithms skewing toward the majority class and offers less storage requirements and better run times for analyses, this method may drop potentially important information or cause the sample of the majority class chosen to be biased and not representative of real world data. In order to improve the sampling method, a combination of both undersampling and oversampling can be implemented to obtain the most lifelike dataset and accurate results. Second, unsupervised machine learning methods such as K-means or Support Vector Machine (SVM) can be implemented to capture normal data distribution in unlabeled data sets when they’re being trained. Unsupervised methods can prove particularly helpful in cases when labeling data is expensive and time-consuming. Third, machine learning methods will differ based on companies' processes and particular setting. Companies will need to conduct research and experimentation to assess what data and features they have readily available to figure out which model can help detect fraud efficiently. This XGB can serve as a base model and can be upgraded based on companies' features, characteristics, and needs.
  
It can be argued that one ethical problem that arises from the use of detection techniques to predict fraudulent and genuine customers is that a technique may predict some customers as genuine, when actually they are fraudulent, and other customers as fraudulent, when actually they are genuine. In terms of justice, these errors should be minimized. However, from the banks' or money transaction apps’ own perspective, the cost of predicting as genuine a customer who is actually fraudulent is much higher than the cost of predicting as fraudulent a customer who is actually genuine. In the latter case, companies lose the opportunity cost of the associated profit margin that would have been earned. However, in the former case, these financial institutions lose the capital value of the money transaction as well as the interest. To operate in the best interests of shareholders, their objective should be to minimize the misclassification costs rather than to minimize the propensity to incorrectly classify customers as fraudulent or genuine. Yet, it would be unethical to reject genuine customers that happened to have the same array of characteristics as those of fraudulent customers. Real world application of machine learning algorithms requires companies to balance the interests of multiple stakeholder groups.
  




