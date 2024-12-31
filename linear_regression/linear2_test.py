import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('/home/suraj-prajapati07/my_ml_lib/datasets/linear_regression_train.csv')



# Extract ID, features and target
ids = data.iloc[:, 0].values  
X = data.iloc[:, 1:-1].values  
y = data['Target'].values      



# Normalize the features (standardization: mean = 0, std = 1)
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X = (X - X_mean) / X_std


# Set the test size to 20%; and make a split index till where to split the data
test_size = 0.2
num_samples = X.shape[0]  
split_index = int(num_samples * (1 - test_size))  



# Spliting the training dataset
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]
ids_test = ids[split_index:]  # Extract IDs for the test set---to be added in the file of predicted values;


#impot the manuallly created linear regression and feed data
from linear2 import LinearRegression
regressor = LinearRegression(learning_rate=0.01, n_iters=1000)
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)


#what's the mean squared error
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

mse = mean_squared_error(y_test, predictions)
print("MSE:", mse)


#frrame the values as ID, Predicted_Target_values
predicted_df = pd.DataFrame({
    "ID": ids_test,
    "Predicted_Target_values": predictions
})
predicted_df.to_csv("/home/suraj-prajapati07/my_ml_lib/outputs/linear_predicted.csv", index=False)
print("Predictions for X_train saved to linear_predicted.csv")


