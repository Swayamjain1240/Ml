import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

X = np.array([
    [1],
    [2],
    [3],
    [4],
    [5],
    [7],
    [8],
    [10]
])

Y = np.array([
    32000,
    35000,
    39000,
    45000,
    52000,
    65000,
    72000,
    85000
])


x_train, x_text, y_train, y_test = train_test_split(X,Y,random_state=42, test_size=0.2)

model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_text)

mae =  mean_absolute_error(y_test,y_pred)
mse =  mean_squared_error(y_test,y_pred)
rmse =  np.sqrt(mse)

r2 = r2_score(y_test,y_pred)


predict = model.predict([[6]])
print("\nPredicted salary for 6 years of experience:", predict[0])

