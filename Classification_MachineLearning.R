## Clear The Enviornment
rm(list = ls(all = T))

### Set The Current Working Directory
setwd("C:\\Users\\NY 5211\\Downloads\\CUTE4")

## Read The Train and Test Data into Enviornment
train_data = read.csv(file = "traindata.csv", header = T)
test_data = read.csv(file = "testdata.csv", header = T)
 
### Looking at the structre of both train and test file
str(train_data)
str(test_data)

## Checking Total Missing Values into both files
sum(is.na(train_data))
sum(is.na(test_data))

## Identifying Column wise Missing Values into both train and test file
colSums(is.na(train_data))
colSums(is.na(test_data))

## Remove those column which contain same length missing values of dataset
train_data = train_data[, !(colSums(is.na(train_data)) == nrow(train_data))]
test_data = test_data[, !(colSums(is.na(test_data)) == nrow(test_data))]

### Remove those column which contains 10% Missing values from the both files
train_data = train_data[,!(colSums(is.na(train_data)) > nrow(train_data) * 0.1)]
test_data = test_data[,!(colSums(is.na(test_data)) > nrow(test_data) * 0.1)]

### COnvert Target variable into Factor
train_data$target = as.factor(train_data$target)

## Remove Unnecessory Column
train_data$ID = NULL
test_data$ID = NULL

### Impute The Missing Values
library(DMwR)
train_data = knnImputation(train_data,k = 5)
test_data = knnImputation(test_data, k = 5)

### Normalization Using PreProcess
preProc = preProcess(train_data[, setdiff(names(train_data),"target")])
train_data = predict(preProc, train_data) 
test_data = predict(preProc, test_data)

library(doParallel)
registerDoParallel(4)

### Make the sample train and test
library(caret)
set.seed(5211)
rows = createDataPartition(train_data$target, p = 0.7, list = F)
train = train_data[rows,]
test = train_data[-rows,]

### Dataset is a highly imbalanced dataset so we need to require Balance dataset
## For balance dataset we can use ROSE library and doing oversampling.
library(ROSE)
set.seed(5211)
balance_train = ovun.sample(target ~ ., data = train, N = 10000)
balance_train <- balance_train$data

### Check The Correlation between Independent variables using corrplot
library(corrplot)
c = cor(balance_train[,-63])
corrplot(c , method = "color",order = "AOE", addrect = 2)

####################### LOGISTIC MODEL ##################################################################################
### Building a Logistic MOdel
glm_model = glm(balance_train$target~., family = "binomial", data = balance_train)
summary(glm_model)

### Outliers Detection
library(car)
outlierTest(glm_model)

### Set the cutoff using ROCR curve and ggplot
library(ggplot2)
library(ROCR)

### Predict on GLM Train data
pred_train = predict(glm_model, newdata = balance_train,type = "response")

## Convert into probablity
prob = prediction(pred_train, balance_train$target)

## Getting Tru Positive and False Negative 
tprfpr = performance(prob, "tpr","fpr")

## Plot the graph and select the cutoff values
plot(tprfpr, col = rainbow(10), colorize = T, print.cutoffs.at=seq(0,1,0.05))

### Check the value of AUC
pred_auc = performance(prob, measure = "auc")
auc = pred_auc@y.values[[1]]

## Predict on Train
pred_train_class = ifelse(pred_train > 0.5,1,0)
confusionMatrix(pred_train_class, balance_train$target, positive = "1")

## Predict on Validation Data
pred_test = predict(glm_model, test, type = "response")
pred_valid = ifelse(pred_test>0.5,1,0)
confusionMatrix(pred_valid,test$target,positive = "1") ## Sensitivity : 0.60
##########################################################################################################

############## STEP AIC ###########################################################################################
### Step AIC
library(MASS)
step_model = stepAIC(glm_model)
summary(step_model)

## Predict on Step model
pred_step_train = predict(step_model, balance_train,type = "response")
step_train = ifelse(pred_step_train > 0.5,1,0)
confusionMatrix(step_train, balance_train$target, positive = "1")

## Predict on Step Validate Data
pred_step_valid = predict(step_model, test, type = "response")
step_valid = ifelse(pred_step_valid > 0.5, 1, 0)
confusionMatrix(step_valid, test$target, positive = "1") ## Sensitivity: 0.60302
##########################################################################################################

############ DECISION TREE WITHOUT CP #############################################################################################
library(rpart)

#### Building a cart model without Cost-Complexity Parameter
rpart_model = rpart(balance_train$target~., data = balance_train, method = "class")
summary(rpart_model)

## Print and plot the cost complexity parameter
printcp(rpart_model)
plotcp(rpart_model)

### Predict on cart train data
pred_cart_train = predict(rpart_model, balance_train, type = "class")
confusionMatrix(pred_cart_train, balance_train$target)

### Predict on cart validation data
pred_cart_validation = predict(rpart_model, test, type = "class")
confusionMatrix(pred_cart_validation, test$target, positive = "1") ## Sensitivity : 0.7584
###################################################################################################################

############ DECISION TREE WITH CP #############################################################################################
#### Building a cart model with Cost-Complexity Parameter
rpart_model = rpart(balance_train$target~., data = balance_train, method = "class",  control = rpart.control(cp = 0.0001))
summary(rpart_model)

## Print and plot the cost complexity parameter
printcp(rpart_model)
plotcp(rpart_model)

### Predict on cart train data
pred_cart_train = predict(rpart_model, balance_train, type = "class")
confusionMatrix(pred_cart_train, balance_train$target)

### Predict on cart validation data
pred_cart_validation = predict(rpart_model, test, type = "class")
confusionMatrix(pred_cart_validation, test$target, positive = "1") ## Sensitivity : 0.7584

############ RANDOM FOREST #############################################################################################
## Building A Random Forest Model

library(randomForest)

random_model = randomForest(balance_train$target~., data = balance_train,keep.forest=TRUE,ntree=100)
print(random_model)

pred_random_train = predict(random_model, newdata = balance_train, type = "response", norm.votes = T)
confusionMatrix(pred_random_train, balance_train$target, positive = "1")

pred_random_validation = predict(random_model, newdata = test, type = "response", norm.votes = T)
confusionMatrix(pred_random_validation, test$target, positive = "1")
