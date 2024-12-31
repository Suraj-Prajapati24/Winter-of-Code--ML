import pandas as pd
import numpy as np

# Load the training dataset
train_data = pd.read_csv('/home/suraj-prajapati07/my_ml_lib/datasets/binary_classification_train.csv')

# Load the testing dataset
test_data = pd.read_csv('/home/suraj-prajapati07/my_ml_lib/datasets/binary_classification_test.csv')

# Extract features and target from the training dataset
X_train = train_data.iloc[:, 1:-1].values  # Features (exclude ID and Class columns)
y_train = train_data['Class'].values       # Target column (Class)

# Extract features and IDs from the testing dataset
X_test = test_data.iloc[:, 1:].values      # Features (exclude ID column)
ids_test = test_data['ID'].values          # IDs for the test dataset



# Import and test Logistic Regression
from logistic_regression import LogisticRegression

regressor = LogisticRegression(learning_rate=0.0001, n_iters=1000)
regressor.fit(X_train, y_train)


# Predict for the test set
predictions = regressor.predict(X_test)


# Define accuracy function
def accuracy(y_true, y_pred):
    acc = np.sum(y_true == y_pred) / len(y_true)
    return acc

# # Calculate and print accuracy
# acc = accuracy(y_test, predictions)
# print("LR classification accuracy:", acc)






# Save training predictions with IDs to a CSV file
predicted_df = pd.DataFrame({
    "ID": ids_test,                 # IDs from the training set
    "Predicted_Train": predictions  # Predicted values for X_train
})
predicted_df.to_csv("/home/suraj-prajapati07/my_ml_lib/outputs/logistic_predicted.csv", index=False)

print("Predictions for X_train with IDs saved to logistic_predicted.csv")
