#### Clear The Enviornment
rm(list = ls(all = T))

### Set The Directory
setwd("C:\\Users\\NY 5211\\Downloads\\MITH")

### Read The Both File
file1 = read.csv(file = "train.csv", header = T, na.strings = "NA")
file2 = read.csv(file = "test.csv", header = T, na.strings = "NA")

str(file1)
str(file2)

head(file1)
head(file2)

sum(is.na(file1))
sum(is.na(file2))

data = rbind(file1, file2)

### Remove The Unnecessary Column
data$NumberOfPictures = NULL
data$ZipCode = NULL
data$DateOfAdCreation = NULL
data$DateOfAdLastSeen = NULL
data$DataCollectedDate = NULL
data$VehicleID = NULL
data$NameOfTheVehicle = NULL
str(data)

## Remove Those factor Columns WHich contain only 1 level
data$SellerType = NULL
data$OfferType = NULL

### Handling The Missing Values
data = data[, !(colSums(is.na(data)) == nrow(data))]
data = data[,!(colSums(is.na(data)) > nrow(data) * 0.3)]
summary(data)

## Missing Value Imputation
library(DMwR)
imputed_data = centralImputation(data)

## Convert the data into numeric
library(dummies)
imputed_data$YearOfVehicleRegistration = as.factor(imputed_data$YearOfVehicleRegistration)
factor_data = imputed_data[, sapply(imputed_data, is.factor)]
factor_columns = factor_data

dummy_data = sapply(factor_data, dummy)

#num_data = imputed_data[, setdiff(names(imputed_data), names(factor_columns))]

## Remove Original Factor Columns and add those column which are dummifying
vehicle_main = imputed_data[, setdiff(names(imputed_data), names(factor_data))]
vehicle_data = data.frame(vehicle_m ain, dummy_data)

### Create Partiton the data
train_data = vehicle_data[1:58857,]
test_data = vehicle_data[58858:78466,]
test_data$Price = NULL

### Spliting the data into train and test
library(caret)
set.seed(5211)
rows = createDataPartition(train_data$Price, p = 0.7, list = F)
train = train_data[rows,]
test = train_data[-rows,]

### Building Liner Model
lm_model = lm(train$Price~., data = train)
summary(lm_model)

## Making The Prediction on Train data
pred_train = predict(lm_model, newdata = train)
regr.eval(train$Price, pred_train)

## Making Prediction On Validation Data
pred_test = predict(lm_model, newdata = test)
regr.eval(test$Price, pred_test)

## Prediction on ACtual Data
pred_actual = predict(lm_model, newdata = test_data)
write.csv(pred_actual, "prediction.csv", row.names = F)

#### Regularization
library(doParallel)
preProc = preProcess(train[,setdiff(names(train),"Price")])
# Depending on the pre-processing standaidizing object, we update the train and 
# test data sets
train = predict(preProc , train)
test = predict(preProc , test)

### For Increase the processing speed, we can use registerDoParallel
registerDoParallel(8)
x = model.matrix(train$Price~., train)
head(x)

### Build LASSO Model
library(glmnet)
fit.lasso = glmnet(x, train$Price, family = "gaussian", alpha = 1)

### Perform  Grid Search to find optimum Lambda
fit.lasso.cv = cv.glmnet(x, train$Price, type.measure = "mse", alpha = 1, family = "gaussian",
                         nfolds = 10, parallel = T)

### Ploting the Lambda Values
plot(fit.lasso)
plot(fit.lasso.cv)

###Build a new Model with Minimum Lambda Value
newmodel_lasso = glmnet(x, train$Price, family = "gaussian",lambda = fit.lasso.cv$lambda.min, alpha = 1)
head(newmodel_lasso$classnames)

### Converting Validation Data Into Proper Formate
x.test = model.matrix(test$Price~., test)
y.test = test$Price

pred_lasso_csv = predict(newmodel_lasso, x.test, s = newmodel_lasso$lambda.min)
regr.eval(y.test, pred_lasso_csv)

### Predict On Actual Test Data
test_data$Price = 0
rgr_test = predict(preProc, test_data)
rgr_test = as.matrix(rgr_test)
predTest_lasso = predict(newmodel_lasso, rgr_test, family = "gaussian" ,s = newmodel_lasso$lambda.min, alpha = 1)
write.csv(predTest_lasso, "prediction.csv", row.names = F)

#### DECISION TREE ****************************************************************
### Using CART build Decision Tree For Linear Regression
library(rpart)
rpart_model = rpart(train$Price~., data = train)
summary(rpart_model)

#### Print the Value of CP values and Ploting it
#printcp(rpart_model)
#plotcp(rpart_model)

### Finding Some Important Variable from the Tree
a = rpart_model$variable.importance

## Making The Prediction on  CART Train data
cart_train = predict(rpart_model, newdata = train)
regr.eval(train$Price, cart_train)

## Making Prediction On Validation Data
cart_test = predict(rpart_model, newdata = test)
regr.eval(test$Price, cart_test)

## Prediction on ACtual Data
cart_actual = predict(rpart_model, newdata = test_data)
write.csv(cart_actual, "prediction.csv", row.names = F)


