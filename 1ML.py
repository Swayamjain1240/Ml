import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Sample dataset
data = {
    "size_sqft": [1000, 1200, 1500, 1800, 2000, 2200, 2500, 2700, 3000, 3200],
    "bedrooms": [2, 2, 3, 3, 4, 4, 4, 5, 5, 5],
    "location_score": [5, 6, 7, 8, 8, 9, 9, 9, 10, 10],
    "price": [120000, 140000, 180000, 220000, 260000, 290000, 320000, 350000, 390000, 420000]
}

df = pd.DataFrame(data)


X = df[["size_sqft", "bedrooms", "location_score"]]
y = df["price"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("Model Performance:")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

# Predict new house
new_house = pd.DataFrame({
    "size_sqft": [1700],
    "bedrooms": [3],
    "location_score": [7]
})

predicted_price = model.predict(new_house)
print("\nPredicted House Price:", predicted_price[0])