library(readr)
train <- read.csv("data/assign3_train.csv")
test <- read.csv("data/assign3_test.csv")

# Overview of dataset
str(train)
table(train$y)

# Convert blank strings "" to NA
train[train == ""] <- NA

# See character columns
char_cols <- names(train)[sapply(train, is.character)]
print(char_cols)

# Convert group ID columns to factors 
cols_to_factor <- c("x4", "x5", "x16", "x18", "x20")
train[cols_to_factor] <- lapply(train[cols_to_factor], as.factor)

# Drop x5 due to 50% of missingness
train$x5 <- NULL

# Assess how much of the dataset is missing, and whether existing missing variables are highly correlated 

# MCAR

# KNN imputation for missingness (464)
library(VIM)

# Impute using KNN with k=5 neighbors (default)
train_imputed <- kNN(train, k = 5)

sum((is.na(train_imputed)))

# Load required libraries
library(randomForest)
library(caret)
library(pROC)

# Step 1: Relabel the target variable to valid R variable names
train_imputed$y <- factor(train_imputed$y, levels = c(0, 1), labels = c("No", "Yes"))

# Step 2: Split data into 80% train and 20% test
set.seed(123)  # for reproducibility
train_index <- createDataPartition(train_imputed$y, p = 0.8, list = FALSE)
train_set <- train_imputed[train_index, ]
test_set <- train_imputed[-train_index, ]

# Step 3: Train a Random Forest model
rf_model <- randomForest(y ~ ., data = train_set, ntree = 100, mtry = 7, importance = TRUE)

# Step 4: Predict on the test set
test_predictions_prob <- predict(rf_model, test_set, type = "prob")[, "Yes"]
test_predictions <- predict(rf_model, test_set, type = "response")

# Step 5: Generate confusion matrix
conf_matrix <- confusionMatrix(test_predictions, test_set$y)
print(conf_matrix)

# Step 6: Compute AUC and plot ROC
roc_curve <- roc(test_set$y, test_predictions_prob, levels = c("No", "Yes"), direction = "<")
print(roc_curve)
plot(roc_curve, main = "ROC Curve", col = "blue")

# Step 7: Cross-validation (5-fold CV)
train_control <- trainControl(method = "cv", number = 5, 
                              summaryFunction = twoClassSummary,
                              classProbs = TRUE,
                              savePredictions = TRUE)

cv_model <- train(y ~ ., data = train_set, method = "rf",
                  trControl = train_control, metric = "ROC")

# Show cross-validation results
print(cv_model)

# Step 8: Compute precision, recall, F1-score, and accuracy
accuracy <- conf_matrix$overall['Accuracy']
precision <- conf_matrix$byClass['Pos Pred Value']
recall <- conf_matrix$byClass['Sensitivity']
f1_score <- 2 * (precision * recall) / (precision + recall)

# Print evaluation metrics
cat("Accuracy: ", accuracy, "\n")
cat("Precision: ", precision, "\n")
cat("Recall: ", recall, "\n")
cat("F1-score: ", f1_score, "\n")

# Process test data

# Check structure of the test data
str(test)

# Identify character columns
char_cols <- names(test)[sapply(test, is.character)]
print(char_cols)

# Convert group ID columns to factors
cols_to_factor <- c("x4", "x5", "x16", "x18", "x20")
test[cols_to_factor] <- lapply(test[cols_to_factor], as.factor)

# Drop x5 due to 50% missingness
test$x5 <- NULL

# Impute missing values using KNN with k=5 neighbors (default)
test_imputed <- kNN(test, k = 5)

# Check if any missing values remain
sum(is.na(test_imputed))

# Ensure the factor levels in the test set match the training set
for (col in names(train_set)) {
  if (is.factor(train_set[[col]])) {
    # Check if the column is present in the test_imputed dataset
    if (col %in% names(test_imputed)) {
      # Set the factor levels in the test set to match the levels in the training set
      test_imputed[[col]] <- factor(test_imputed[[col]], levels = levels(train_set[[col]]))
    }
  }
}

# Ensure that the columns in test_imputed match those in the training set
# Align columns by removing the target variable 'y' if it's present
test_imputed <- test_imputed[, names(train_set)[-which(names(train_set) == "y")]]  # Remove target column 'y'

# Step 1: Make predictions on the test dataset using the trained random forest model
test_predictions <- predict(rf_model, test_imputed)  # Use the trained model to predict on the test set

# Step 2: Convert the factor predictions to numeric (0 and 1)
test_predictions_numeric <- as.numeric(test_predictions) - 1  # This will convert "Yes" to 1 and "No" to 0

# Step 3: Create a dataframe with numeric predictions
predictions_df <- data.frame(y = test_predictions_numeric)

head(predictions_df)