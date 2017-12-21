#### Clear The Enviornment
rm(list = ls(all = T))

### Set The Directory
setwd("C:\\Users\\NY 5211\\Downloads\\MITH")

### Read The Both File
file1 = read.csv(file = "train.csv", header = T, na.strings = "NA")
file2 = read.csv(file = "new_test.csv", header = T, na.strings = "NA")
file2$Price = NA

### Combine the Both Dataset
new_data = rbind(file1,file2)

### Remove the columns
new_data$NumberOfPictures = NULL
new_data$ZipCode = NULL
new_data$DateOfAdCreation = NULL
new_data$DateOfAdLastSeen = NULL
new_data$DataCollectedDate = NULL
new_data$VehicleID = NULL
new_data$NameOfTheVehicle = NULL

## Remove Those factor Columns WHich contain only 1 level
new_data$SellerType = NULL
new_data$OfferType = NULL

### Handling The Missing Values
new_data = new_data[, !(colSums(is.na(new_data)) == nrow(new_data))]
new_data = new_data[,!(colSums(is.na(new_data)) > nrow(new_data) * 0.3)]
summary(new_data)

## Missing Value Imputation
library(DMwR)
imputed_data = centralImputation(new_data[,-1])
imputed_data$Price = new_data$Price

## Convert categorical into numeric data
library(dummies)
imputed_data$YearOfVehicleRegistration = as.factor(imputed_data$YearOfVehicleRegistration)
factor_type = imputed_data[, sapply(imputed_data, is.factor)]
dummy_data = sapply(factor_type, dummy)

vehicle = imputed_data[, setdiff(names(imputed_data), names(factor_type))]
vehicle_data = data.frame(vehicle, dummy_data)

### create Data Partition
train_data = vehicle_data[1:58857,]
test_data = vehicle_data[58858:70829,]
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
regr.eval(train$Price, pred_test)

## Prediction on ACtual Data
pred_actual = predict(lm_model, newdata = test_data)
write.csv(pred_actual, "prediction.csv", row.names = F)

#******************************************************************************************

### PERFORM STEPAIC *****************************************************
library(MASS)
step_model = stepAIC(lm_model)
summary(step_model)

## Making The Prediction on Train data
pred_train = predict(step_model, newdata = train)
regr.eval(train$Price, pred_train)

## Making Prediction On Validation Data
pred_test = predict(step_model, newdata = test)
regr.eval(train$Price, pred_test)

## Prediction on ACtual Data
pred_actual = predict(step_model, newdata = test_data)
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
fit.ridge = glmnet(x, train$Price, family = "gaussian", alpha = 0)

### Perform  Grid Search to find optimum Lambda
fit.lasso.cv = cv.glmnet(x, train$Price, type.measure = "mse", alpha = 1, family = "gaussian",
                         nfolds = 10, parallel = T)
fit.ridge.cv = cv.glmnet(x, train$Price, type.measure = "mse", alpha = 0, family = "gaussian",
                         nfolds = 10, parallel = T)


fit.lasso.cv$lambda.min
fit.ridge.cv$lambda.min
### Ploting the Lambda Values
plot(fit.lasso)
plot(fit.lasso.cv)

plot(fit.ridge)
plot(fit.ridge.cv)


###Build a new Model with Minimum Lambda Value
newmodel_lasso = glmnet(x, train$Price, family = "gaussian",lambda = fit.lasso.cv$lambda.min, alpha = 1)
newmodel_ridge = glmnet(x, train$Price, family = "gaussian",lambda = fit.ridge.cv$lambda.min, alpha = 0)

### Converting Validation Data Into Proper Formate
x.test = model.matrix(test$Price~., test)
y.test = test$Price

pred_lasso_csv = predict(newmodel_lasso, x.test, s = newmodel_lasso$lambda.min, alpha = 1)
regr.eval(y.test, pred_lasso_csv)

pred_ridge_csv = predict(newmodel_ridge, x.test, s = newmodel_ridge$lambda.min, alpha = 0)
regr.eval(y.test, pred_ridge_csv)

### Predict On Actual Test Data
test_data$Price = 0
rgr_test = predict(preProc, test_data)
rgr_test = as.matrix(rgr_test)
predTest_lasso = predict(newmodel_lasso, rgr_test, family = "gaussian" ,s = newmodel_lasso$lambda.min, alpha = 1)
write.csv(predTest_lasso, "prediction.csv", row.names = F)

predTest_ridge = predict(newmodel_ridge, rgr_test, family = "gaussian" ,s = newmodel_ridge$lambda.min, alpha = 0)
write.csv(predTest_ridge, "prediction.csv", row.names = F)

#*************************************************************************************************************************

#### Decision Tree #########################################################################
library(rpart)
rpart_model = rpart(train$Price~., data = train)
summary(rpart_model)
printcp(rpart_model)

### Finding Some Important Variable from the Tree
a = rpart_model$variable.importance

# Making The Prediction on  CART Train data
cart_train = predict(rpart_model, newdata = train)
regr.eval(train$Price, cart_train)

## Making Prediction On Validation Data
cart_test = predict(rpart_model, newdata = test)
regr.eval(test$Price, cart_test)

## Prediction on ACtual Data
cart_actual = predict(rpart_model, newdata = test_data)
write.csv(cart_actual, "prediction.csv", row.names = F)


imp_train = c("PowerOfTheEngine","DistranceTravelled","VehicleType.factor_typeSmall.Car",
             "TypeOfTheFuelUsed.factor_typediesel","TypeOfTheFuelUsed.factor_typepetrol",
             "GearBoxType.factor_typeautomatic","GearBoxType.factor_typemanual","ModelOfTheVehicle.factor_type5er",
             "YearOfVehicleRegistration.factor_type2013","YearOfVehicleRegistration.factor_type2014",
             "YearOfVehicleRegistration.factor_type2012","BrandOfTheVehicle.factor_typebmw","YearOfVehicleRegistration.factor_type2011",
             "ModelOfTheVehicle.factor_typea6","VehicleType.factor_typelimousine","VehicleType.factor_typecabrio",
             "VehicleType.factor_typeCombi","YearOfVehicleRegistration.factor_type2015","BrandOfTheVehicle.factor_typeporsche",
             "ModelOfTheVehicle.factor_typecayenne","ModelOfTheVehicle.factor_type3er","ModelOfTheVehicle.factor_typea8",
             "ModelOfTheVehicle.factor_typec_klasse","ModelOfTheVehicle.factor_types_klasse","ModelOfTheVehicle.factor_typeq7",
             "VehicleType.factor_typesuv","VehicleType.factor_typebus","ModelOfTheVehicle.factor_type7er","ModelOfTheVehicle.factor_typesl",
             "ModelOfTheVehicle.factor_type6er","YearOfVehicleRegistration.factor_type2016","YearOfVehicleRegistration.factor_type2010",
             "ModelOfTheVehicle.factor_typea1","ModelOfTheVehicle.factor_typescirocco","ModelOfTheVehicle.factor_typekuga","Price")

imp_train = train[,imp_train]

imp_test= c("PowerOfTheEngine","DistranceTravelled","VehicleType.factor_typeSmall.Car",
              "TypeOfTheFuelUsed.factor_typediesel","TypeOfTheFuelUsed.factor_typepetrol",
              "GearBoxType.factor_typeautomatic","GearBoxType.factor_typemanual","ModelOfTheVehicle.factor_type5er",
              "YearOfVehicleRegistration.factor_type2013","YearOfVehicleRegistration.factor_type2014",
              "YearOfVehicleRegistration.factor_type2012","BrandOfTheVehicle.factor_typebmw","YearOfVehicleRegistration.factor_type2011",
              "ModelOfTheVehicle.factor_typea6","VehicleType.factor_typelimousine","VehicleType.factor_typecabrio",
              "VehicleType.factor_typeCombi","YearOfVehicleRegistration.factor_type2015","BrandOfTheVehicle.factor_typeporsche",
              "ModelOfTheVehicle.factor_typecayenne","ModelOfTheVehicle.factor_type3er","ModelOfTheVehicle.factor_typea8",
              "ModelOfTheVehicle.factor_typec_klasse","ModelOfTheVehicle.factor_types_klasse","ModelOfTheVehicle.factor_typeq7",
              "VehicleType.factor_typesuv","VehicleType.factor_typebus","ModelOfTheVehicle.factor_type7er","ModelOfTheVehicle.factor_typesl",
              "ModelOfTheVehicle.factor_type6er","YearOfVehicleRegistration.factor_type2016","YearOfVehicleRegistration.factor_type2010",
              "ModelOfTheVehicle.factor_typea1","ModelOfTheVehicle.factor_typescirocco","ModelOfTheVehicle.factor_typekuga","Price")

imp_test = test[,imp_test]

lm_model_imp = lm(imp_train$Price~., data = imp_train)
summary(lm_model_imp)

## Making The Prediction on Train data
pred_train = predict(lm_model_imp, newdata = imp_train)
regr.eval(imp_train$Price, pred_train)

## Making Prediction On Validation Data
pred_test = predict(lm_model_imp, newdata = imp_test)
regr.eval(imp_test$Price, pred_test)

## Prediction on ACtual Data
pred_actual = predict(lm_model_imp, newdata = test_data)
write.csv(pred_actual, "prediction.csv", row.names = F)
