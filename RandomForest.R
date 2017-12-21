#### Clear The Enviornment
rm(list = ls(all = T))

### Set The Directory
setwd("C:\\Users\\NY 5211\\Downloads\\MITH")

### Read The Both File
file1 = read.csv(file = "train.csv", header = T, na.strings = "NA")
file2 = read.csv(file = "test.csv", header = T, na.strings = "NA")

data = rbind(file1, file2)

### Remove The Unnecessary Column
data$NumberOfPictures = NULL
data$ZipCode = NULL
data$DateOfAdCreation = NULL
data$DateOfAdLastSeen = NULL
data$DataCollectedDate = NULL
data$VehicleID = NULL
data$NameOfTheVehicle = NULL

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

imputed_data$YearOfVehicleRegistration = as.factor(imputed_data$YearOfVehicleRegistration)
imputed_data$ModelOfTheVehicle = NULL

### Create Partiton the data
train_data = imputed_data[1:58857,]
test_data = imputed_data[58858:78466,]
test_data$Price = NULL

### Spliting the data into train and test
library(caret)
set.seed(5211)
rows = createDataPartition(train_data$Price, p = 0.7, list = F)
train = train_data[rows,]
test = train_data[-rows,]

### Building A Random Forest Model
library(randomForest)

random_model = randomForest(train$Price~., train, keep.forest=TRUE, ntree=200)
print(random_model)
random_model$importance

### MAking Prediction on Train Data
random_train = predict(random_model, train, norm.votes=TRUE)
regr.eval(train$Price, random_train)

## Making Prediction On Validation Data
random_test = predict(random_model, test, norm.votes=TRUE)
regr.eval(test$Price, random_test)

### Making Prediction on actual Test Data
random_actual = predict(random_model, test_data)
write.csv(random_actual, "prediction.csv", row.names = F)

