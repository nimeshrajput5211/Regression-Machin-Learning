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

### Standardization
library(vegan)
vehicle_data = decostand(vehicle_data, "range", na.rm = T)

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

library(xgboost)
### Fit the model
dtrain = xgb.DMatrix(data = as.matrix(train[,-4]),
                     label = train$Price)

### Building a model
xg_model = xgboost(data = dtrain, max.depth = 2, 
               eta = 1, nthread = 2, nround = 2, 
               objective = "reg:linear", verbose = 1)

#Use watchlist parameter. It is a list of xgb.DMatrix, each of them tagged with a name.

### Convert the test data and fit the model
dtest = xgb.DMatrix(data = as.matrix(test[,-4]),
                    label = test$Price)

watchlist = list(train=dtrain, test=dtest)

model = xgb.train(data=dtrain, max.depth=2,
                  eta=1, nthread = 2, nround=5, 
                  watchlist=watchlist,
                  eval.metric = "error", 
                  objective = "reg:linear",
                  verbose=1)

# eval.metric allows us to monitor two new metrics for each round, logloss and error.

## MAking Prediction on Train data
xg_train <- predict(model, as.matrix(train[,-4]))
regr.eval(train$Price, xg_train)

### Prediction on Validation
xg_test = predict(model, as.matrix(test[,-4]))
regr.eval(test$Price, xg_test)

### Prediction on Actual New Test Data
xg_actual = predict(model, as.matrix(test_data))
write.csv(xg_actual, "prediction.csv", row.names = F)

#************************ GRID SEARCH *********************************************************

ctrl <- trainControl(method = "repeatedcv",   # n fold cross validation
                     number = 2,							# do 2 repititions of cv
                    allowParallel = TRUE)

xgb.grid <- expand.grid(nrounds = c(2,5), #the maximum number of iterations
                        eta = c(0.1,1), # shrinkage
                        max_depth = c(2,5,10),
                        subsample = 1,
                        gamma = c(0,1),               #default=0
                        colsample_bytree = c(1,0.5),    #default=1
                        min_child_weight = c(1,2))

xgb.tune <-train(x=train[,-4],
                 y=train$Price,
                 method="xgbTree",
                 metric="RMSE",
                 trControl=ctrl,
                 tuneGrid=xgb.grid)


#This will give best tuning parameters from given set of parameters                 
xgb.tune$bestTune

plot(xgb.tune)  		# Plot the performance of the training models
res <- xgb.tune$results
head(res)

## MAking Prediction on test data
pred_test = predict(xgb.tune, test)
regr.eval(test$Price, pred_test)

### MAkin Prediction on Actual Test Data
xg_actual_test = predict(xgb.tune, test_data)
write.csv(xg_actual_test, "prediction.csv", row.names = F)
