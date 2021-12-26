## ----setup, include=FALSE------------------------------------------------------------------------------------
knitr::opts_chunk$set(echo = TRUE)


## ------------------------------------------------------------------------------------------------------------
rm(list=ls()) # clear workspace
cat("\014")  # clear console
graphics.off() # shuts down all open graphics devices 


## ------------------------------------------------------------------------------------------------------------
financialfraud <- read.csv("/Users/thuhuong/Desktop/R Working Directory/FinancialFraud.csv")
head(financialfraud)


## ------------------------------------------------------------------------------------------------------------
dim(financialfraud) #dimension of the dataset
summary(financialfraud) #summary of the dataset
sum(is.na(financialfraud)) #check null values in the dataset
library(dplyr)
financialfraud %>% count(isFraud) #number of fraud vs. non-fraud transactions
library(ggplot2)
g <- ggplot(financialfraud, aes(isFraud))
g + geom_bar(fill="lightblue") + geom_label(stat='count', aes(label = paste0(round(((..count..) /sum(..count..)), 4)*100, "%"))) + labs(x = "Fraud vs Not Fraud", y = "Frequency", title = "Frequency of Fraud", subtitle = "Labels as Percent of Total Observations") + scale_y_continuous(labels = scales::comma)


## ------------------------------------------------------------------------------------------------------------
fraud_trans_type <- financialfraud %>% group_by(type) %>% summarize(fraud_count = sum(isFraud)) #type of fraud transactions
ggplot(fraud_trans_type, aes(x=type, y=fraud_count)) + geom_col(fill="lightblue") + labs(title = 'Fraud transactions per Type', x = 'Transaction Type', y = 'Number of Fraud Transactions') + geom_label(aes(label=fraud_count)) + theme_classic()


## ------------------------------------------------------------------------------------------------------------
ggplot(data=financialfraud[financialfraud$isFraud==1,], aes(x=amount)) + geom_histogram(fill="lightblue") + labs(title = 'Fraud transactions Amount distribution', x = 'Amount', y = 'Number of Fraud Transactions') + theme_classic() + scale_x_continuous(labels = scales::comma)


## ------------------------------------------------------------------------------------------------------------
modulo <- function(y, x) {
  q <- y/x
  q <- floor(q)
  p <- q*x
  mod <- y-p
  return(mod)
}
financialfraud$hour <- modulo(y=financialfraud$step,x=24)
fraud_hour <- financialfraud %>%
group_by(hour) %>%
summarise(cnt = n(),sum=sum(isFraud)) %>%
mutate(fraud_percentage = round((sum/cnt)*100,2))
ggplot(fraud_hour, aes(x=hour,y=fraud_percentage)) + geom_col(fill="lightblue") + labs(title = 'Fraud Percentage per Hour', x = 'Hour', y = 'Percentage of Fraud Transactions') + theme_classic()


## ------------------------------------------------------------------------------------------------------------
financialfraud<- financialfraud %>% 
                  select( -one_of('step','nameOrig', 'nameDest', 'isFlaggedFraud'))
library(caret)
set.seed(1234)
splitindex <- createDataPartition(financialfraud$isFraud, p=0.7, list=FALSE, times=1)
train <- financialfraud[splitindex,]
table(train$isFraud)
test <- financialfraud[-splitindex,]
table(test$isFraud)
# Use under-sampling majority class method for inbalanced dataset
library(unbalanced)
inputs <- train[,-which(names(train) %in% "isFraud")]
target <- as.factor(train[,which(names(train) %in% "isFraud")])
under_sam <- ubUnder(X = inputs, Y = target)
train <- cbind(under_sam$X, under_sam$Y)
train$isFraud <- train$`under_sam$Y`
train$`under_sam$Y` <- NULL
table(train$isFraud)


## ------------------------------------------------------------------------------------------------------------
logistic <- glm(isFraud~., data=train, family="binomial")
summary(logistic)
logisticprediction <- predict(logistic, test, type="response")
logisticprediction <- ifelse(logisticprediction>0.5, 1, 0)
table(logisticprediction)
table(test$isFraud)
confusionMatrix(as.factor(logisticprediction),as.factor(test$isFraud))
library("pROC")
auc1 <- roc(test$isFraud, logisticprediction)
auc1


## ------------------------------------------------------------------------------------------------------------
library(rpart)
library(rpart.plot)
tree <- rpart(isFraud ~ ., data = train)
prp(tree)
treeprediction <- predict(tree, test, type = "class")
table(treeprediction)
table(test$isFraud)
confusionMatrix(as.factor(treeprediction),as.factor(test$isFraud))
auc2 <- roc(test$isFraud, as.numeric(treeprediction))
auc2


## ------------------------------------------------------------------------------------------------------------
library(randomForest)
rf <- randomForest(isFraud ~ ., data = train, importance = TRUE)
rfprediction <- predict(rf, test)
table(rfprediction)
confusionMatrix(as.factor(rfprediction),as.factor(test$isFraud))
auc3 <- roc(test$isFraud, as.numeric(rfprediction))
auc3


## ------------------------------------------------------------------------------------------------------------
library(xgboost)
train_x = data.matrix(train[,-which(names(train) %in% "isFraud")])
train_y = as.numeric(levels(train$isFraud))[train$isFraud]
test_x = data.matrix(test[,-which(names(test) %in% "isFraud")])
test$isFraud <- as.factor(test$isFraud)
test_y = as.numeric(levels(test$isFraud))[test$isFraud]
xgb_train = xgb.DMatrix(data=train_x, label=train_y)
xgb_test = xgb.DMatrix(data=test_x, label=test_y)
xgmodel <- xgboost(data=xgb_train, nrounds=10000, max_depth=3, early_stopping_rounds=50, objective="binary:logistic", verbose=0)
xgmodel$best_iteration #tune for best nround
xgmodel <- xgboost(data=xgb_train, nrounds=2290, max_depth=3, early_stopping_rounds=50, objective="binary:logistic", verbose=0)
xgboostprediction <- predict(xgmodel, xgb_test)
xgboostprediction <- ifelse(xgboostprediction>0.5, 1, 0)
table(xgboostprediction)
confusionMatrix(as.factor(xgboostprediction),as.factor(test$isFraud))
auc4 <- roc(test$isFraud, as.numeric(xgboostprediction))
auc4


## ------------------------------------------------------------------------------------------------------------
importance <- xgb.importance(model=xgmodel)
importance
xgb.plot.importance(importance_matrix = importance, xlab = "Relative importance based on Gain")

