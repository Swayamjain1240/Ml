import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# 1. Create dataset
data = {
    "Area": [500, 750, 1000, 1250, 1500,
             1750, 2000, 2250, 2500, 3000],

    "Price": [100000, 150000, 200000, 250000, 300000,
              350000, 400000, 450000, 500000, 600000]
}

# 2. Convert dictionary into DataFrame
df = pd.DataFrame(data)

# 3. Display dataset
print("Dataset:")
print(df)

# 4. Basic data exploration
print("\nFirst 5 rows:")
print(df.head())

print("\nShape:")
print(df.shape)

print("\nInformation:")
print(df.info())

print("\nStatistics:")
print(df.describe())

# 5. Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# 6. Define features and target
X = df[["Area"]]
y = df["Price"]

# 7. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 8. Create Linear Regression model
model = LinearRegression()

# 9. Train model
model.fit(X_train, y_train)

# 10. Make predictions
y_pred = model.predict(X_test)

# 11. Display actual and predicted values
print("\nActual Prices:")
print(y_test.values)

print("\nPredicted Prices:")
print(y_pred)

# 12. Evaluate model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R² Score:", r2)

# 13. Predict price for a new house
new_house = pd.DataFrame({
    "Area": [1800]
})

predicted_price = model.predict(new_house)

print("\nPredicted price for 1800 sq ft:")
print(predicted_price[0])

# 14. Visualize data and regression line
plt.scatter(X, y)

plt.plot(
    X,
    model.predict(X)
)

plt.xlabel("Area (sq ft)")
plt.ylabel("Price ($)")
plt.title("House Price Prediction using Linear Regression")

plt.show()